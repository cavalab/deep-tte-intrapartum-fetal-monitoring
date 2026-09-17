"""Streaming trainer for standard, marked, and competing DeepHit models."""
import importlib
import hashlib
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys
import uuid

import fire
import numpy as np
import pandas as pd
import torch
import torchtuples as tt
from pycox.models import DeepHitSingle
from pycox.preprocessing.label_transforms import LabTransDiscreteTime
from tqdm.auto import tqdm

try:
    import psutil
except ImportError:  # pragma: no cover - optional runtime telemetry dependency
    psutil = None

import utils
from methods.competing_risks_survival import (
    CauseSpecificCompetingDeepHitNet,
    CompetingDeepHitNet,
    PartialLabelDeepHit,
    delivery_pmf_from_competing,
    threshold_cumulative_risks,
)
from time_to_event_data import RandomChunkLoader, lab_metadata
from methods.marked_survival import (
    MarkedDeepHitNet,
    MarkedDeepHitSingle,
    corn_cumulative_mark_probabilities,
    conditional_mark_probabilities,
    delivery_pmf,
    discover_ph_threshold_columns,
    joint_cumulative_risks,
)
from time_to_event_evaluation import (
    EVALUATION_HORIZON_SECONDS,
    evaluate_landmark_suite,
)


def _threshold_name(threshold):
    """Format a numeric pH threshold for result-table labels.

    Parameters
    ----------
    threshold : float
        pH cutoff.

    Returns
    -------
    str
        Canonical label such as ``"pH < 7.2"``.
    """
    return f"pH < {float(threshold):g}"


def _resolve_event_label(columns, requested_label):
    """Resolve a requested threshold label to an available label column.

    Parameters
    ----------
    columns : Iterable[str]
        Available data-frame column names.
    requested_label : str
        Requested label, for example ``"pH Cord < 7.20"``.

    Returns
    -------
    str
        Matching available column name. Numeric matching makes formatting such
        as ``7.20`` and ``7.2`` equivalent.

    Raises
    ------
    ValueError
        If the requested label is malformed or has no unique matching column.
    """
    if requested_label in columns:
        return requested_label
    try:
        requested_threshold = float(str(requested_label).rsplit(" ", 1)[-1])
    except ValueError as error:
        raise ValueError(f"Invalid pH threshold event_label: {requested_label}") from error
    matches = [
        column for threshold, column in discover_ph_threshold_columns(columns)
        if np.isclose(threshold, requested_threshold)
    ]
    if len(matches) != 1:
        raise ValueError(f"Unknown event_label: {requested_label}")
    return matches[0]


class ElapsedTimeDeepHitNet(torch.nn.Module):
    """Attach a DeepHit output layer to an encoder and optional elapsed time.

    The elapsed scalar is concatenated to the pooled encoder representation,
    so it cannot be mistaken for a physiological waveform channel.
    """

    def __init__(self, encoder, n_time_bins, elapsed_time_feature=False):
        """Configure an encoder, DeepHit head, and optional scalar feature."""
        super().__init__()
        if not hasattr(encoder, "forward_features"):
            raise TypeError("encoder must expose forward_features(x)")
        feature_dim = getattr(encoder, "feature_dim", None)
        if feature_dim is None and hasattr(encoder, "fc"):
            feature_dim = encoder.fc.in_features
        if feature_dim is None:
            raise TypeError("encoder must expose feature_dim or an fc layer")
        self.encoder = encoder
        self.elapsed_time_feature = bool(elapsed_time_feature)
        self.head = torch.nn.Linear(
            feature_dim + int(self.elapsed_time_feature), int(n_time_bins)
        )

    def forward(self, x, elapsed_time=None):
        """Return DeepHit logits from trace features and optional elapsed time."""
        features = self.encoder.forward_features(x)
        if self.elapsed_time_feature:
            if elapsed_time is None:
                raise ValueError("elapsed_time is required when elapsed_time_feature is enabled")
            elapsed_time = elapsed_time.reshape(features.shape[0], 1).to(features.dtype)
            features = torch.cat([features, elapsed_time], dim=1)
        return self.head(features)


def _prediction_function(net, *, mode, device, feature_stats, cuts, sample_rate_hz,
                         target_names, inference_batch_size=1024,
                         elapsed_time_feature=False, elapsed_time_scale_seconds=43200,
                         elapsed_time_channel=False, n_physiological_features=None,
                         history_window_count=1):
    """Build batched landmark inference for one trained time-to-event network.

    Parameters
    ----------
    net : torch.nn.Module
        Trained standard, marked, or competing-risk DeepHit network.
    mode : {"survival", "marked", "marked_corn", "competing", "competing_threshold"}
        Determines how logits are converted to cumulative risks.
    device : torch.device
        Device on which inference is run.
    feature_stats : list[dict] | None
        Training-set feature min/max values used to scale chunks.
    cuts : array-like
        Discrete delivery-time cuts in samples.
    sample_rate_hz : float
        Trace sampling rate used to convert cuts to seconds.
    target_names : list[str]
        Output names for pH threshold risk curves.
    elapsed_time_feature : bool, default=False
        Pass normalized elapsed chunk-end time to a compatible network.
    elapsed_time_scale_seconds : float, default=43200
        Fixed denominator used to normalize elapsed seconds.
    elapsed_time_channel : bool, default=False
        Append sample-level elapsed-time values to chunks before applying
        feature scaling.
    n_physiological_features : int | None, default=None
        Number of raw physiological channels at the beginning of each chunk.
        Required with both elapsed-time and missingness-mask input channels so
        elapsed time is inserted before the masks.
    inference_batch_size : int, default=1024
        Number of chunks evaluated in each forward pass.

    Returns
    -------
    Callable
        Function accepting chunks shaped ``[N, T, C]`` and, when enabled,
        aligned elapsed endpoint seconds. Returns a seconds ``time_grid`` plus
        cumulative-risk curves shaped ``[N, n_cuts]``.
    """

    grid_seconds = np.asarray(cuts, dtype=float) / float(sample_rate_hz)

    if elapsed_time_scale_seconds <= 0:
        raise ValueError("elapsed_time_scale_seconds must be positive")

    def predict(chunks, elapsed_seconds=None):
        """Scale chunks, run the network, and return named cumulative risks.

        Parameters
        ----------
        chunks : numpy.ndarray
            Raw landmark chunks shaped ``[N, T, C]``.
        elapsed_seconds : numpy.ndarray | None
            Chunk-end seconds since tracing start, required when the feature is
            enabled.

        Returns
        -------
        dict[str, numpy.ndarray]
            Seconds time grid and mode-specific cumulative-risk curves.
        """
        chunks = np.asarray(chunks, dtype=np.float32)
        if elapsed_time_feature or elapsed_time_channel:
            if elapsed_seconds is None:
                raise ValueError("elapsed_seconds is required when an elapsed-time option is enabled")
            elapsed_seconds = np.asarray(elapsed_seconds, dtype=np.float32).reshape(-1, 1)
            if len(elapsed_seconds) != len(chunks):
                raise ValueError("elapsed_seconds must align with chunks")
        if elapsed_time_channel:
            channel_insert_index = (
                chunks.shape[-1]
                if n_physiological_features is None
                else int(n_physiological_features)
            )
            sample_count = chunks.shape[-2]
            sample_offsets = np.arange(sample_count, dtype=np.float32) / float(sample_rate_hz)
            if chunks.ndim == 3:
                chunk_start_seconds = elapsed_seconds - sample_count / float(sample_rate_hz)
                elapsed_channels = chunk_start_seconds[:, None, :] + sample_offsets[None, :, None]
            elif chunks.ndim == 4:
                window_offsets = np.arange(chunks.shape[1], dtype=np.float32)
                window_starts = elapsed_seconds[:, None, :] - (
                    chunks.shape[1] - window_offsets[None, :, None]
                ) * sample_count / float(sample_rate_hz)
                elapsed_channels = window_starts[:, :, None, :] + sample_offsets[None, None, :, None]
            else:
                raise ValueError("landmark chunks must be [N,T,C] or [N,W,T,C]")
            chunks = np.concatenate([
                chunks[..., :channel_insert_index],
                elapsed_channels,
                chunks[..., channel_insert_index:],
            ], axis=-1)
        if chunks.ndim == 3:
            scaled = np.stack([utils._apply_feature_stats(chunk, feature_stats) for chunk in chunks])
        elif chunks.ndim == 4:
            scaled = np.stack([
                np.stack([utils._apply_feature_stats(window, feature_stats) for window in history])
                for history in chunks
            ])
        else:
            raise ValueError("landmark chunks must be [N,T,C] or [N,W,T,C]")
        was_training = net.training
        net.eval()
        outputs = []
        if elapsed_time_feature:
            elapsed_seconds = elapsed_seconds / float(elapsed_time_scale_seconds)
        with torch.no_grad():
            for start in range(0, len(scaled), int(inference_batch_size)):
                batch = torch.from_numpy(scaled[start:start + int(inference_batch_size)])
                batch = batch.transpose(1, 2) if scaled.ndim == 3 else batch.permute(0, 1, 3, 2)
                if elapsed_time_feature:
                    elapsed_batch = torch.from_numpy(
                        elapsed_seconds[start:start + int(inference_batch_size)]
                    ).to(device)
                    outputs.append(net(batch.to(device), elapsed_batch))
                else:
                    outputs.append(net(batch.to(device)))
        if was_training:
            net.train()

        result = {"time_grid": grid_seconds}
        if mode == "survival":
            logits = torch.cat(outputs, dim=0)
            result[target_names[0]] = delivery_pmf(logits).cumsum(1).cpu().numpy()
        elif mode in {"marked", "marked_corn"}:
            delivery_logits = torch.cat([item[0] for item in outputs], dim=0)
            mark_logits = torch.cat([item[1] for item in outputs], dim=0)
            pmf = delivery_pmf(delivery_logits)
            result["delivery"] = pmf.cumsum(1).cpu().numpy()
            mark_probabilities = (
                corn_cumulative_mark_probabilities(mark_logits)
                if mode == "marked_corn"
                else conditional_mark_probabilities(mark_logits)
            )
            risks = joint_cumulative_risks(pmf, mark_probabilities).cpu().numpy()
            for index, name in enumerate(target_names):
                result[name] = risks[:, :, index]
        else:
            logits = torch.cat(outputs, dim=0)
            result["delivery"] = delivery_pmf_from_competing(logits).cumsum(1).cpu().numpy()
            for index, name in enumerate(target_names):
                result[name] = threshold_cumulative_risks(logits, index).cpu().numpy()
        return result

    predict.uses_elapsed_time_feature = bool(elapsed_time_feature or elapsed_time_channel)
    return predict


class PeriodicHistoryWriter(tt.callbacks.Callback):
    """Atomically snapshot the accumulated training history at fixed epochs."""

    def __init__(self, path, every_epochs=10):
        self.path = Path(path)
        self.every_epochs = int(every_epochs)
        if self.every_epochs < 1:
            raise ValueError("every_epochs must be positive")
        self.completed_epochs = 0

    def write(self):
        """Write all metrics observed through the current completed epoch."""
        frame = self.model.log.to_pandas()
        temporary = self.path.with_name(f".{self.path.name}.tmp")
        frame.to_csv(temporary, index=False)
        os.replace(temporary, self.path)

    def on_epoch_end(self):
        self.completed_epochs += 1
        if self.completed_epochs % self.every_epochs == 0:
            self.write()


class TrainingProgress(tt.callbacks.Callback):
    """Display batch-level training progress and live resource telemetry."""

    def __init__(self, epochs, train_batches, validation_batches, gpu_index=0,
                 training_landmark_monitor=None, optimizer_name="optimizer",
                 progress_bar=False):
        """Create a batch progress bar suitable for batch-job logs.

        Parameters
        ----------
        epochs : int
            Maximum configured epoch count.
        train_batches : int
            Number of streaming batches per training epoch.
        validation_batches : int
            Number of streaming batches in validation.
        gpu_index : int, default=0
            CUDA device index queried through ``nvidia-smi``.
        """
        self.epochs = int(epochs)
        self.train_batches = int(train_batches)
        self.validation_batches = int(validation_batches)
        self.gpu_index = int(gpu_index)
        self.epoch = 0
        self.progress_bar = None
        self.last_validation_loss = None
        self.training_landmark_monitor = training_landmark_monitor
        self.optimizer_name = str(optimizer_name)
        self.progress_bar_enabled = bool(progress_bar)

    def _resource_postfix(self):
        """Return current GPU, VRAM, system-RAM, and process-RSS telemetry."""
        telemetry = {"gpu": "n/a", "vram": "n/a", "ram": "n/a", "rss": "n/a"}
        try:
            result = subprocess.run(
                [
                    "nvidia-smi",
                    f"--id={self.gpu_index}",
                    "--query-gpu=utilization.gpu,memory.used,memory.total",
                    "--format=csv,noheader,nounits",
                ],
                check=True,
                capture_output=True,
                text=True,
                timeout=2,
            )
            utilization, used, total = result.stdout.strip().splitlines()[0].split(", ")
            telemetry["gpu"] = f"{utilization}%"
            telemetry["vram"] = f"{used}/{total}MiB"
        except (FileNotFoundError, subprocess.CalledProcessError, subprocess.TimeoutExpired, IndexError, ValueError):
            pass
        if psutil is not None:
            memory = psutil.virtual_memory()
            rss_gib = psutil.Process().memory_info().rss / 1024**3
            telemetry["ram"] = f"{memory.percent:.0f}%"
            telemetry["rss"] = f"{rss_gib:.1f}GiB"
        return telemetry

    def on_epoch_start(self):
        """Open a progress bar for the freshly sampled training epoch."""
        if not self.progress_bar_enabled:
            return
        description = f"epoch {self.epoch + 1}/{self.epochs}"
        self.progress_bar = tqdm(
            total=self.train_batches,
            desc=description,
            unit="batch",
            dynamic_ncols=True,
            file=sys.stdout,
            mininterval=5,
        )
        self.progress_bar.set_postfix(
            val_loss=(f"{self.last_validation_loss:.4g}" if self.last_validation_loss is not None else "pending"),
            **self._resource_postfix(),
        )

    def on_batch_end(self):
        """Advance the training bar and periodically refresh resource telemetry."""
        if self.progress_bar is None:
            return
        self.progress_bar.update(1)
        if self.progress_bar.n == self.train_batches or self.progress_bar.n % 10 == 0:
            self.progress_bar.set_postfix(
                val_loss=(f"{self.last_validation_loss:.4g}" if self.last_validation_loss is not None else "pending"),
                **self._resource_postfix(),
            )

    def on_epoch_end(self):
        """Finalize the bar with completed losses and optimizer telemetry."""
        train_loss = float(self.model.train_metrics.scores["loss"]["score"][-1])
        validation_loss = float(self.model.val_metrics.scores["loss"]["score"][-1])
        self.last_validation_loss = validation_loss
        if self.progress_bar is not None:
            postfix = {
                "train_loss": f"{train_loss:.4g}",
                "val_loss": f"{validation_loss:.4g}",
                "lr": f"{self.model.optimizer.param_groups[0]['lr']:.3g}",
                **self._resource_postfix(),
            }
            if self.training_landmark_monitor is not None:
                postfix["train_landmark_panel_loss"] = (
                    f"{self.training_landmark_monitor.last_loss:.4g}"
                )
            postfix["optimizer"] = self.optimizer_name
            self.progress_bar.set_postfix(**postfix)
            self.progress_bar.close()
            self.progress_bar = None
        self.epoch += 1

    def on_fit_end(self):
        """Close a partially completed bar when fitting exits early or errors."""
        if self.progress_bar is not None:
            self.progress_bar.close()
            self.progress_bar = None


class TrainingLandmarkMonitor(tt.callbacks.Callback):
    """Record loss on a small, fixed training landmark panel each epoch.

    The panel is deliberately separate from the sampled training batches: it
    distinguishes an improving stochastic training loss from improvement at
    reproducible landmarks. With terminal-balanced sampling, the panel also
    contains deterministic terminal landmarks. It is inference-only and does
    not influence gradients or early stopping.
    """

    def __init__(self, dataloader):
        self.dataloader = dataloader
        self.monitor = tt.callbacks.MonitorMetrics()
        self.last_loss = None

    def give_model(self, model):
        super().give_model(model)
        # TrainingLogger writes every named monitor to the normal history CSV.
        self.model.log.monitors["train_landmark_panel_"] = self.monitor

    def on_epoch_end(self):
        scores = self.model.score_in_batches_dataloader(self.dataloader)
        self.last_loss = float(scores["loss"])
        self.monitor.epoch += 1
        self.monitor.append_score("loss", self.last_loss)


class WarmupCosineSchedule(tt.callbacks.Callback):
    """Linearly warm up, then cosine-decay an optimizer's learning-rate cap.

    The schedule advances after each optimizer update, rather than per epoch,
    so it remains comparable when the number of chunks per epoch changes.
    """

    def __init__(self, total_steps, initial_lr, warmup_fraction=0.05,
                 min_fraction=0.01):
        self.total_steps = int(total_steps)
        self.initial_lr = float(initial_lr)
        self.warmup_fraction = float(warmup_fraction)
        self.min_fraction = float(min_fraction)
        if self.total_steps < 1:
            raise ValueError("total_steps must be positive for cosine scheduling")
        if self.initial_lr <= 0:
            raise ValueError("initial_lr must be positive for cosine scheduling")
        if not 0 <= self.warmup_fraction < 1:
            raise ValueError("lr_warmup_fraction must be in [0, 1)")
        if not 0 < self.min_fraction <= 1:
            raise ValueError("lr_min_fraction must be in (0, 1]")
        self.warmup_steps = min(
            self.total_steps - 1,
            int(round(self.total_steps * self.warmup_fraction)),
        )
        self.step_count = 0

    def _lr_for_update(self, update_index):
        """Return the cap used by one-indexed optimizer update ``update_index``."""
        update_index = max(1, min(int(update_index), self.total_steps))
        if self.warmup_steps and update_index <= self.warmup_steps:
            return self.initial_lr * update_index / self.warmup_steps
        cosine_steps = self.total_steps - self.warmup_steps
        progress = (update_index - self.warmup_steps - 1) / max(cosine_steps - 1, 1)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return self.initial_lr * (
            self.min_fraction + (1.0 - self.min_fraction) * cosine
        )

    def _set_lr(self, lr):
        for group in self.model.optimizer.param_groups:
            group["lr"] = float(lr)

    def on_fit_start(self):
        self._set_lr(self._lr_for_update(1))

    def on_batch_end(self):
        self.step_count += 1
        # Set the cap for the next update. The final scheduled cap is reached
        # on the last optimizer step, not after training has ended.
        if self.step_count < self.total_steps:
            self._set_lr(self._lr_for_update(self.step_count + 1))


def make_guarded_cocob(parameters, *, weight_decay, alpha,
                       gradient_clip_norm, max_update_norm,
                       max_parameter_abs, max_rejected_steps):
    """Build parameterfree's COCOB with guards for non-convex deep learning.

    COCOB retains parameterfree's native ``lr=1.0`` gradient multiplier rather
    than accepting this trainer's learning-rate argument. Gradient clipping,
    rollback of non-finite/oversized updates, and a rejection limit avoid
    silently continuing after a catastrophic update.
    """
    try:
        from parameterfree import COCOB
    except ImportError as exc:  # pragma: no cover - depends on optional install
        raise ImportError(
            "optimizer='cocob' requires parameterfree; install project dependencies "
            "with `uv sync`."
        ) from exc

    class GuardedCOCOB(COCOB):
        def __init__(self, params):
            super().__init__(params, alpha=alpha, eps=1e-8, weight_decay=weight_decay)
            self.gradient_clip_norm = float(gradient_clip_norm)
            self.max_update_norm = float(max_update_norm)
            self.max_parameter_abs = float(max_parameter_abs)
            self.max_rejected_steps = int(max_rejected_steps)
            self.rejected_steps = 0

        @torch.no_grad()
        def step(self, closure=None):
            params_with_grad = [
                parameter for group in self.param_groups for parameter in group["params"]
                if parameter.grad is not None
            ]
            if not params_with_grad:
                return None
            grad_norm = torch.nn.utils.clip_grad_norm_(
                params_with_grad, max_norm=self.gradient_clip_norm,
            )
            if not torch.isfinite(grad_norm):
                self._reject("non-finite gradient")
                return None
            previous = [parameter.detach().clone() for parameter in params_with_grad]
            loss = super().step(closure)
            update_norm_sq = sum(
                (parameter.detach() - before).square().sum()
                for parameter, before in zip(params_with_grad, previous)
            )
            update_norm = torch.sqrt(update_norm_sq)
            valid = (
                torch.isfinite(update_norm)
                and float(update_norm) <= self.max_update_norm
                and all(
                    torch.isfinite(parameter).all()
                    and float(parameter.detach().abs().max()) <= self.max_parameter_abs
                    for parameter in params_with_grad
                )
            )
            if valid:
                self.rejected_steps = 0
                return loss
            for parameter, before in zip(params_with_grad, previous):
                parameter.copy_(before)
            # COCOB's wealth/statistics are no longer consistent after a
            # rejected update. Restarting its per-parameter state is safer
            # than allowing corrupted state to influence the next step.
            self.state.clear()
            self._reject("non-finite or oversized parameter update")
            return None

        def _reject(self, reason):
            self.rejected_steps += 1
            if self.rejected_steps > self.max_rejected_steps:
                raise RuntimeError(
                    f"COCOB rejected {self.rejected_steps} consecutive updates ({reason}); "
                    "aborting before catastrophic training failure."
                )

    return GuardedCOCOB(parameters)


class Trainer:
    """Train and evaluate one standard, marked, or competing DeepHit experiment.

    The Fire CLI exposes the constructor parameters as command-line arguments.
    ``run`` samples according to the configured training regime, trains the
    selected model, and writes model, history, risk, and evaluation artifacts
    to ``savedir``.
    """
    def __init__(self, trace_file, label_trainfile, label_testfile=None, mode="marked", event_label="pH Cord < 7.1",
                 savedir="results_time_to_event", features='["toco", "fecg"]',
                 ml="Inception_classifier", splits_file="data/train_val_test_splits_extended.csv",
                 chunk_window_size=900, overlap=0, batch_size=256, epochs=200,
                 weight_decay=0.0, lab_order_delay=30, horizon=0, random_state=42,
                 gpu=0, deephit_alpha=0.2, deephit_sigma=0.1, mark_loss_weight=1.0,
                 missing_delivery_loss_weight=1.0, durations='[0,600,1200,2400,3600,7200,14400,28800,86400,172800]',
                 competing_head="joint", competing_head_hidden_dims="[128,64]",
                 competing_head_batch_norm=True, competing_head_dropout=0.2,
                 early_stopping_patience=12, bootstrap_samples=100,
                 bootstrap_confidence_level=0.95, elapsed_time_feature=False,
                 history_write_interval=10,
                 elapsed_time_scale_seconds=43200, elapsed_time_channel=False,
                 missingness_indicator_channels=False,
                 missing_data_method="ffill",
                 optimizer="adafactor", lr_schedule="constant", learning_rate=None,
                 lr_warmup_fraction=0.05, lr_min_fraction=0.01,
                 cocob_alpha=100.0, cocob_gradient_clip_norm=1.0,
                 cocob_max_update_norm=100.0, cocob_max_parameter_abs=1e4,
                 cocob_max_rejected_steps=8,
                 history_seconds=8 * 60 * 60,
                 validation_landmarks_per_patient=4,
                 training_sampling_strategy="uniform",
                 training_chunks_per_patient_per_epoch="auto",
                 progress_bar=False,
                 training_landmark_diagnostic_patients=1024,
                 chunk_missingness_max_fraction=None,
                 chunk_eligibility_cache_dir="cache/chunk_eligibility",
                 fit_kwargs="{}", eval_only=False, checkpoint_path="auto",
                 checkpoint_cuts=None,
                 evaluation_name="evaluation",
                 skip_missing_checkpoint=True):
        """Store data, model, optimization, and reporting configuration.

        Parameters
        ----------
        trace_file : str
            Variable-length tracing store.
        label_trainfile, label_testfile : str | None
            Train/validation and optional held-out test lab tables.
        mode : {"survival", "marked", "marked_corn", "competing", "competing_threshold", "all"}
            Model formulation to fit.
        competing_head : {"joint", "cause_specific"}, default="joint"
            Competing-risk output architecture. ``"joint"`` uses one linear
            cause-by-time head; ``"cause_specific"`` uses one independent MLP
            head per cause.
        competing_head_hidden_dims : JSON list[int] | list[int], default=[128, 64]
            Strictly decreasing hidden widths for ``cause_specific`` heads.
        competing_head_batch_norm : bool, default=True
            Use batch normalization in ``cause_specific`` hidden layers.
        competing_head_dropout : float, default=0.2
            Dropout probability in ``cause_specific`` hidden layers.
        event_label : str
            pH-threshold column for ``survival`` mode.
        savedir : str
            Directory for model and report artifacts.
        features : JSON list[str] | list[str]
            Signal channels presented to the encoder.
        chunk_window_size, batch_size, epochs, weight_decay : int or float
            Core training settings.
        optimizer : {"adafactor", "adamw", "cocob"}, default="adafactor"
            Optimization algorithm. COCOB uses parameterfree's native
            gradient multiplier and requires ``learning_rate`` to be omitted.
        lr_schedule : {"constant", "cosine"}, default="constant"
            ``"cosine"`` uses a linear per-update warmup followed by cosine
            decay for Adafactor or AdamW. COCOB uses ``"constant"`` only.
        learning_rate : float | None, default=None
            Adafactor/AdamW scale constrained to ``[1e-7, 1e-2]``; defaults
            to 0.01. Omit for COCOB.
        lr_warmup_fraction, lr_min_fraction : float, default=0.05, 0.01
            Cosine-only fractions of total optimizer updates used for linear
            warmup and the final cap relative to ``learning_rate``.
        durations : JSON list[float] | list[float]
            Delivery-time cuts in seconds.
        bootstrap_samples, bootstrap_confidence_level : int, float
            Landmark metric bootstrap configuration.
        history_write_interval : int, default=10
            Write an atomic snapshot of the accumulated training/validation
            history after this many completed epochs. The final history is
            always written when fitting ends.
        elapsed_time_feature : bool, default=False
            Concatenate normalized chunk-end time since tracing start to the
            encoder representation before the outcome heads.
        elapsed_time_scale_seconds : float, default=43200
            Fixed elapsed-time normalization denominator in seconds.
        elapsed_time_channel : bool, default=False
            Append elapsed time since tracing start as a synthetic input channel
            at every sample. Unlike ``elapsed_time_feature``, this is processed
            by the convolutional encoder.
        missingness_indicator_channels : bool, default=False
            Append one pre-imputation binary observed-value input channel for
            every physiological signal channel.
        missing_data_method : {"ffill", "zeros"}, default="ffill"
            Missing-data handling applied after selecting each landmark chunk.
        history_seconds : float, default=28800
            Maximum prior tracing history supplied to historical encoders.
        validation_landmarks_per_patient : int, default=4
            Four fixed delivery-relative validation endpoints per patient at
            0, 20, 40, and 60 minutes before delivery. Unlike randomly
            resampled validation chunks, this produces a stable loss for early
            stopping.
        training_sampling_strategy : {"uniform", "stratified_elapsed", "stratified_elapsed_plus_terminal", "terminal_balanced", "quartiles_plus_terminal"}, default="uniform"
            Training endpoint policy. ``"stratified_elapsed"`` rotates each
            patient across the elapsed-time strata defined by the validation
            landmarks, while sampling only from its eligible starts. It never
            uses time remaining to delivery to choose an input chunk.
            ``"terminal_balanced"`` alternates the elapsed-time strata with
            terminal strata centered on 0, 20, 40, and 60 minutes before
            delivery. This is appropriate only when explicitly optimizing
            retrospective end-landmark performance.
            ``"quartiles_plus_terminal"`` uses five chunks from every PID in
            every epoch: one random chunk from each elapsed-time quartile and
            one random chunk ending within 20 minutes of delivery. Validation
            always uses the common fixed delivery-relative endpoint panel.
            ``"stratified_elapsed_plus_terminal"`` uses the elapsed-time
            strata plus the final-20-minute terminal stratum.
        training_chunks_per_patient_per_epoch : {"auto", "all_strata"} or int, default="auto"
            Per-PID training chunk budget forwarded to ``RandomChunkLoader``.
        progress_bar : bool, default=False
            Show HDF5, chunk-enumeration, and training-epoch progress bars.
        training_landmark_diagnostic_patients : int, default=1024
            Number of deterministically selected training PIDs used for the
            fixed multi-landmark loss diagnostic after each epoch. Each PID
            contributes ``validation_landmarks_per_patient`` elapsed-time
            chunks, plus four terminal chunks when terminal-balanced sampling
            is selected or one terminal chunk when quartiles-plus-terminal
            sampling is selected. Set to zero to disable the diagnostic; it never
            affects gradients or early stopping.
        chunk_missingness_max_fraction : float | None, default=None
            Restrict training and validation sampling to chunks whose raw
            total missing-value fraction does not exceed this threshold.
            Values greater than one are interpreted as percentages. Each PID
            still contributes one uniformly sampled eligible chunk per epoch.
        chunk_eligibility_cache_dir : str, default="cache/chunk_eligibility"
            Directory for reusable compressed valid-chunk sidecar indexes.
        fit_kwargs : JSON dict | dict
            Encoder-construction keyword arguments.
        eval_only : bool, default=False
            Skip fitting, load ``checkpoint_path``, and regenerate only the
            landmark evaluation artifacts.
        checkpoint_path : str | None, default="auto"
            Saved ``.pt`` network state used when ``eval_only=True``. The
            default resolves the deterministic checkpoint key derived from
            this experiment's training configuration. The YAML must otherwise
            match the checkpoint's model, target, discretisation, and encoder
            settings.
        checkpoint_cuts : sequence[float] | None, default=None
            Evaluation-time delivery cuts in seconds from the checkpoint's
            training grid. This must be supplied when evaluating on a cohort
            whose data cannot reconstruct the checkpoint's training grid.
        evaluation_name : str, default="evaluation"
            Version label appended to evaluation-only artifacts. Change this
            when retaining results from multiple evaluator revisions.
        skip_missing_checkpoint : bool, default=True
            In evaluation-only mode, record a skipped-evaluation marker and
            exit successfully when the fitted checkpoint is absent.
        """
        self.__dict__.update(locals())
        self.features = json.loads(features) if isinstance(features, str) else features
        self.durations = json.loads(durations) if isinstance(durations, str) else durations
        self.fit_kwargs = json.loads(fit_kwargs) if isinstance(fit_kwargs, str) else fit_kwargs
        if isinstance(checkpoint_cuts, str):
            checkpoint_cuts = json.loads(checkpoint_cuts)
        self.checkpoint_cuts = checkpoint_cuts
        self.competing_head = str(self.competing_head).strip().lower()
        if self.competing_head not in {"joint", "cause_specific"}:
            raise ValueError("competing_head must be 'joint' or 'cause_specific'")
        if isinstance(self.competing_head_hidden_dims, str):
            self.competing_head_hidden_dims = json.loads(self.competing_head_hidden_dims)
        try:
            self.competing_head_hidden_dims = tuple(
                int(width) for width in self.competing_head_hidden_dims
            )
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "competing_head_hidden_dims must be a sequence of positive integers"
            ) from exc
        if (
            not self.competing_head_hidden_dims
            or any(width <= 0 for width in self.competing_head_hidden_dims)
            or any(
                next_width >= width
                for width, next_width in zip(
                    self.competing_head_hidden_dims,
                    self.competing_head_hidden_dims[1:],
                )
            )
        ):
            raise ValueError(
                "competing_head_hidden_dims must strictly decrease and contain positive widths"
            )
        if isinstance(competing_head_batch_norm, str):
            self.competing_head_batch_norm = (
                competing_head_batch_norm.strip().lower() in {"1", "true", "yes"}
            )
        else:
            self.competing_head_batch_norm = bool(competing_head_batch_norm)
        self.competing_head_dropout = float(competing_head_dropout)
        if not 0 <= self.competing_head_dropout < 1:
            raise ValueError("competing_head_dropout must be in [0, 1)")
        self.optimizer = str(self.optimizer).strip().lower()
        if self.optimizer not in {"adafactor", "adamw", "cocob"}:
            raise ValueError("optimizer must be 'adafactor', 'adamw', or 'cocob'")
        if isinstance(self.eval_only, str):
            self.eval_only = self.eval_only.strip().lower() in {"1", "true", "yes"}
        else:
            self.eval_only = bool(self.eval_only)
        self.checkpoint_path = "auto" if self.checkpoint_path is None else str(self.checkpoint_path)
        self.evaluation_name = str(self.evaluation_name).strip()
        if not self.evaluation_name or Path(self.evaluation_name).name != self.evaluation_name:
            raise ValueError("evaluation_name must be a non-empty filename label without path separators")
        if isinstance(self.skip_missing_checkpoint, str):
            self.skip_missing_checkpoint = (
                self.skip_missing_checkpoint.strip().lower() in {"1", "true", "yes"}
            )
        else:
            self.skip_missing_checkpoint = bool(self.skip_missing_checkpoint)
        if isinstance(self.elapsed_time_feature, str):
            self.elapsed_time_feature = self.elapsed_time_feature.strip().lower() in {"1", "true", "yes"}
        else:
            self.elapsed_time_feature = bool(self.elapsed_time_feature)
        self.elapsed_time_scale_seconds = float(self.elapsed_time_scale_seconds)
        if isinstance(self.elapsed_time_channel, str):
            self.elapsed_time_channel = self.elapsed_time_channel.strip().lower() in {"1", "true", "yes"}
        else:
            self.elapsed_time_channel = bool(self.elapsed_time_channel)
        if isinstance(self.missingness_indicator_channels, str):
            self.missingness_indicator_channels = (
                self.missingness_indicator_channels.strip().lower() in {"1", "true", "yes"}
            )
        else:
            self.missingness_indicator_channels = bool(self.missingness_indicator_channels)
        self.missing_data_method = str(self.missing_data_method).strip().lower()
        if self.missing_data_method not in {"ffill", "zeros"}:
            raise ValueError("missing_data_method must be 'ffill' or 'zeros'")
        self.lr_schedule = str(self.lr_schedule).strip().lower()
        if self.lr_schedule not in {"constant", "cosine"}:
            raise ValueError("lr_schedule must be 'constant' or 'cosine'")
        if self.learning_rate is None or (
            isinstance(self.learning_rate, str)
            and self.learning_rate.strip().lower() in {"", "none", "null"}
        ):
            self.learning_rate = None if self.optimizer == "cocob" else 1e-2
        else:
            self.learning_rate = float(self.learning_rate)
        if self.optimizer != "cocob" and not 1e-7 <= self.learning_rate <= 1e-2:
            raise ValueError("learning_rate must be between 1e-7 and 1e-2")
        if self.optimizer == "cocob" and self.learning_rate is not None:
            raise ValueError(
                "COCOB uses parameterfree's native gradient multiplier; omit learning_rate"
            )
        if self.optimizer == "cocob" and self.lr_schedule != "constant":
            raise ValueError("COCOB uses its native multiplier; lr_schedule must be 'constant'")
        self.lr_warmup_fraction = float(self.lr_warmup_fraction)
        self.lr_min_fraction = float(self.lr_min_fraction)
        if not 0 <= self.lr_warmup_fraction < 1:
            raise ValueError("lr_warmup_fraction must be in [0, 1)")
        if not 0 < self.lr_min_fraction <= 1:
            raise ValueError("lr_min_fraction must be in (0, 1]")
        self.cocob_alpha = float(self.cocob_alpha)
        self.cocob_gradient_clip_norm = float(self.cocob_gradient_clip_norm)
        self.cocob_max_update_norm = float(self.cocob_max_update_norm)
        self.cocob_max_parameter_abs = float(self.cocob_max_parameter_abs)
        self.cocob_max_rejected_steps = int(self.cocob_max_rejected_steps)
        if self.cocob_alpha <= 0 or self.cocob_gradient_clip_norm <= 0:
            raise ValueError("cocob_alpha and cocob_gradient_clip_norm must be positive")
        if self.cocob_max_update_norm <= 0 or self.cocob_max_parameter_abs <= 0:
            raise ValueError("COCOB update and parameter bounds must be positive")
        if self.cocob_max_rejected_steps < 0:
            raise ValueError("cocob_max_rejected_steps must be non-negative")
        self.history_seconds = float(self.history_seconds)
        if self.history_seconds <= 0:
            raise ValueError("history_seconds must be positive")
        self.history_write_interval = int(self.history_write_interval)
        if self.history_write_interval < 1:
            raise ValueError("history_write_interval must be positive")
        self.early_stopping_patience = int(self.early_stopping_patience)
        if self.early_stopping_patience < 0:
            raise ValueError("early_stopping_patience must be non-negative")
        self.validation_landmarks_per_patient = int(self.validation_landmarks_per_patient)
        self.progress_bar = bool(progress_bar)
        if self.validation_landmarks_per_patient != 4:
            raise ValueError(
                "validation_landmarks_per_patient must be 4 for the fixed "
                "0/20/40/60-minute delivery endpoints"
            )
        self.training_sampling_strategy = str(self.training_sampling_strategy).strip().lower()
        if self.training_sampling_strategy not in {
            "uniform", "stratified_elapsed", "stratified_elapsed_plus_terminal",
            "terminal_balanced", "quartiles_plus_terminal"
        }:
            raise ValueError(
                "training_sampling_strategy must be 'uniform', 'stratified_elapsed', "
                "'stratified_elapsed_plus_terminal', 'terminal_balanced', or "
                "'quartiles_plus_terminal'"
            )
        self.training_chunks_per_patient_per_epoch = training_chunks_per_patient_per_epoch
        self.training_landmark_diagnostic_patients = int(
            self.training_landmark_diagnostic_patients
        )
        if self.training_landmark_diagnostic_patients < 0:
            raise ValueError("training_landmark_diagnostic_patients must be non-negative")
        if self.chunk_missingness_max_fraction is None or (
            isinstance(self.chunk_missingness_max_fraction, str)
            and self.chunk_missingness_max_fraction.strip().lower() in {"", "none", "null"}
        ):
            self.chunk_missingness_max_fraction = None
        else:
            self.chunk_missingness_max_fraction = float(self.chunk_missingness_max_fraction)
            if self.chunk_missingness_max_fraction > 1:
                self.chunk_missingness_max_fraction /= 100.0
            if not 0 <= self.chunk_missingness_max_fraction <= 1:
                raise ValueError("chunk_missingness_max_fraction must be between 0 and 1 (or 0 and 100)")
        self.chunk_eligibility_cache_dir = str(self.chunk_eligibility_cache_dir)
        self.training_key = self._training_checkpoint_key()
        self.artifact_label = self._artifact_label()
        self.auto_checkpoint_path = str(
            Path(self.savedir) / "checkpoints" / f"{self.artifact_label}.pt"
        )
        self.result_stem = str(Path(self.savedir) / self.artifact_label)
        if self.eval_only:
            if self.checkpoint_path.strip().lower() == "auto":
                self.checkpoint_path = self.auto_checkpoint_path
        self.run_id = str(uuid.uuid4())

    def _training_checkpoint_key(self):
        """Return a stable identity for one fitted model configuration.

        Evaluation-only settings are intentionally omitted, so metric changes
        can replay the same learned weights. The short digest keeps paths safe
        for batch schedulers and common filesystems.
        """
        identity = {
            "trace_file": str(self.trace_file),
            "label_trainfile": str(self.label_trainfile),
            "label_testfile": None if self.label_testfile is None else str(self.label_testfile),
            "splits_file": str(self.splits_file),
            "mode": str(self.mode),
            "event_label": str(self.event_label),
            "features": list(self.features),
            "ml": str(self.ml),
            "chunk_window_size": int(self.chunk_window_size),
            "batch_size": int(self.batch_size),
            "epochs": int(self.epochs),
            "early_stopping_patience": int(self.early_stopping_patience),
            "random_state": int(self.random_state),
            "weight_decay": float(self.weight_decay),
            "lab_order_delay": float(self.lab_order_delay),
            "horizon": float(self.horizon),
            "deephit_alpha": float(self.deephit_alpha),
            "deephit_sigma": float(self.deephit_sigma),
            "mark_loss_weight": float(self.mark_loss_weight),
            "missing_delivery_loss_weight": float(self.missing_delivery_loss_weight),
            "durations": [float(value) for value in self.durations],
            "competing_head": self.competing_head,
            "competing_head_hidden_dims": list(self.competing_head_hidden_dims),
            "competing_head_batch_norm": bool(self.competing_head_batch_norm),
            "competing_head_dropout": float(self.competing_head_dropout),
            "elapsed_time_feature": bool(self.elapsed_time_feature),
            "elapsed_time_scale_seconds": float(self.elapsed_time_scale_seconds),
            "elapsed_time_channel": bool(self.elapsed_time_channel),
            "missingness_indicator_channels": bool(self.missingness_indicator_channels),
            "missing_data_method": str(self.missing_data_method),
            "lr_schedule": self.lr_schedule,
            "learning_rate": None if self.learning_rate is None else float(self.learning_rate),
            "lr_warmup_fraction": float(self.lr_warmup_fraction),
            "lr_min_fraction": float(self.lr_min_fraction),
            "cocob_alpha": float(self.cocob_alpha),
            "cocob_gradient_clip_norm": float(self.cocob_gradient_clip_norm),
            "cocob_max_update_norm": float(self.cocob_max_update_norm),
            "cocob_max_parameter_abs": float(self.cocob_max_parameter_abs),
            "cocob_max_rejected_steps": int(self.cocob_max_rejected_steps),
            "history_seconds": float(self.history_seconds),
            "validation_landmarks_per_patient": int(self.validation_landmarks_per_patient),
            "validation_panel": self._validation_panel_name(),
            "training_sampling_strategy": self.training_sampling_strategy,
            "training_chunks_per_patient_per_epoch": self.training_chunks_per_patient_per_epoch,
            "chunk_missingness_max_fraction": self.chunk_missingness_max_fraction,
            "fit_kwargs": self.fit_kwargs,
            "optimizer": self.optimizer,
        }
        encoded = json.dumps(identity, sort_keys=True, separators=(",", ":"), default=str)
        return hashlib.sha256(encoded.encode("utf-8")).hexdigest()[:16]

    def _validation_panel_name(self):
        """Name the fixed validation panel recorded with training artifacts."""
        return "delivery_endpoints_0_20_40_60m_v1"

    def _artifact_label(self):
        """Create a readable, deterministic artifact stem for one fitted model."""
        model_name = re.sub(r"_classifier$", "", str(self.ml), flags=re.IGNORECASE)
        model_name = re.sub(r"[^a-z0-9]+", "-", model_name.lower()).strip("-")
        parts = [str(self.mode), "deephit"]
        if self.mode in {"survival", "competing_threshold"}:
            try:
                threshold = float(str(self.event_label).rsplit(" ", 1)[-1])
                parts.append(f"ph-{int(round(threshold * 100)):03d}")
            except (ValueError, IndexError):
                parts.append("target-custom")
        parts.extend([
            f"impute-{self.missing_data_method}",
            model_name,
            f"s{int(self.random_state)}",
            self.training_key,
        ])
        return "_".join(parts)

    def _model_metadata(self):
        """Describe the learned encoder independently from input imputation."""
        model_name = str(self.ml)
        if "INCEPTION" in model_name.upper():
            encoder_family = "InceptionTime"
            architecture = (
                "InceptionTime_LSTM" if "LSTM" in model_name.upper() else "InceptionTime"
            )
        else:
            encoder_family = "other"
            architecture = model_name
        return {
            "model_class": model_name,
            "encoder_family": encoder_family,
            "architecture": architecture,
            "competing_head": (
                self.competing_head
                if self.mode in {"competing", "competing_threshold"}
                else None
            ),
            "competing_head_hidden_dims": (
                list(self.competing_head_hidden_dims)
                if self.mode in {"competing", "competing_threshold"}
                else None
            ),
            "competing_head_batch_norm": (
                bool(self.competing_head_batch_norm)
                if self.mode in {"competing", "competing_threshold"}
                else None
            ),
            "competing_head_dropout": (
                float(self.competing_head_dropout)
                if self.mode in {"competing", "competing_threshold"}
                else None
            ),
            "fit_kwargs": self.fit_kwargs,
        }

    def _imputation_metadata(self):
        """Describe preprocessing without conflating it with the encoder."""
        return {
            "method": self.missing_data_method,
        }

    def _resolve_evaluation_checkpoint(self, net, device):
        """Resolve the deterministic checkpoint for an evaluation-only run."""
        configured = Path(self.checkpoint_path)
        if configured.is_file():
            return str(configured)
        if str(configured) != self.auto_checkpoint_path:
            raise FileNotFoundError(f"evaluation checkpoint not found: {configured}")

        raise FileNotFoundError(
            f"Evaluation checkpoint not found: {self.auto_checkpoint_path}. "
            "Pass --checkpoint_path with an explicit checkpoint path."
        )

    def _record_skipped_evaluation(self, reason):
        """Persist an auditable marker for an evaluation-only run with no state."""
        stem = f"{self.result_stem}_{self.evaluation_name}_skipped"
        with open(stem + ".json", "w") as handle:
            json.dump(
                {
                    "status": "skipped",
                    "reason": str(reason),
                    "mode": self.mode,
                    "event_label": self.event_label,
                    "model": self.ml,
                    "random_state": int(self.random_state),
                    "eval_only": True,
                    "requested_checkpoint": self.checkpoint_path,
                    "evaluation_name": self.evaluation_name,
                },
                handle,
                indent=2,
            )
        print(f"[evaluation] skipped: {reason}", flush=True)
        print(f"[evaluation] wrote {stem}.json", flush=True)

    def run(self):
        """Fit one experiment and persist its model, metrics, scores, and metadata.

        Returns
        -------
        None
            Artifacts are written beneath ``savedir``. In ``all`` mode, this
            method recursively executes four threshold models and binary
            competing-risk experiments, plus marked and categorical
            competing-risk experiments.
        """
        if self.mode == "all":
            original_mode, original_event_label = self.mode, self.event_label
            for threshold in ("7.05", "7.10", "7.15", "7.20"):
                self.mode = "survival"
                self.event_label = f"pH Cord < {threshold}"
                self.run_id = str(uuid.uuid4())
                self.run()
            for threshold in ("7.05", "7.10", "7.15", "7.20"):
                self.mode = "competing_threshold"
                self.event_label = f"pH Cord < {threshold}"
                self.run_id = str(uuid.uuid4())
                self.run()
            for mode in ("marked", "marked_corn", "competing"):
                self.mode = mode
                self.event_label = original_event_label
                self.run_id = str(uuid.uuid4())
                self.run()
            self.mode, self.event_label = original_mode, original_event_label
            return
        if self.mode not in {"survival", "marked", "marked_corn", "competing", "competing_threshold"}:
            raise ValueError(
                "mode must be survival, marked, marked_corn, competing, competing_threshold, or all"
            )
        torch.manual_seed(self.random_state)
        np.random.seed(self.random_state)
        os.makedirs(self.savedir, exist_ok=True)
        Path(self.auto_checkpoint_path).parent.mkdir(parents=True, exist_ok=True)
        print(
            f"[setup] mode={self.mode} model={self.ml} seed={self.random_state} "
            f"chunk_samples={self.chunk_window_size} batch_size={self.batch_size} "
            f"optimizer={self.optimizer} "
            f"lr_schedule={self.lr_schedule} learning_rate="
            f"{self.learning_rate if self.learning_rate is not None else 'COCOB-default'} "
            f"eval_only={self.eval_only} "
            f"elapsed_time_feature={bool(self.elapsed_time_feature)} "
            f"elapsed_time_channel={self.elapsed_time_channel} "
            f"missingness_indicator_channels={self.missingness_indicator_channels} "
            f"validation_landmarks_per_patient={self.validation_landmarks_per_patient} "
            f"training_sampling_strategy={self.training_sampling_strategy} "
            f"training_chunks_per_patient_per_epoch="
            f"{self.training_chunks_per_patient_per_epoch} "
            f"missing_data_method={self.missing_data_method} "
            f"chunk_missingness_max_fraction={self.chunk_missingness_max_fraction}",
            flush=True,
        )
        print("[setup] loading trace and lab metadata", flush=True)
        frame, thresholds = lab_metadata(
            self.trace_file, self.label_trainfile, chunk_window_size=self.chunk_window_size,
            lab_order_delay=self.lab_order_delay, horizon=self.horizon,
        )
        def make_optimizer(parameters):
            """Construct the configured optimizer with its comparable LR scale."""
            if self.optimizer == "adafactor":
                return torch.optim.Adafactor(
                    parameters, lr=self.learning_rate, beta2_decay=-0.8,
                    eps=(None, 1e-3), d=1.0, weight_decay=self.weight_decay,
                    foreach=False,
                )
            if self.optimizer == "adamw":
                return torch.optim.AdamW(
                    parameters, lr=self.learning_rate, weight_decay=self.weight_decay,
                )
            return make_guarded_cocob(
                parameters, weight_decay=self.weight_decay, alpha=self.cocob_alpha,
                gradient_clip_norm=self.cocob_gradient_clip_norm,
                max_update_norm=self.cocob_max_update_norm,
                max_parameter_abs=self.cocob_max_parameter_abs,
                max_rejected_steps=self.cocob_max_rejected_steps,
            )
        if self.mode == "survival":
            resolved_event_label = _resolve_event_label(frame.columns, self.event_label)
            frame = frame[frame[resolved_event_label].notna()].copy()
            frame["event"] = (frame[resolved_event_label] > 0).astype(np.int64)
        elif self.mode == "competing_threshold":
            # Delivery remains observed for every eligible tracing. pH-missing
            # deliveries therefore retain an unknown cause (event_type 0),
            # while known pH produces one of two mutually exclusive causes.
            resolved_event_label = _resolve_event_label(frame.columns, self.event_label)
            frame["event_type"] = 0
            observed = frame["ph_observed"]
            frame.loc[observed & (frame[resolved_event_label] > 0), "event_type"] = 1
            frame.loc[observed & (frame[resolved_event_label] <= 0), "event_type"] = 2
        if self.splits_file is None:
            frame["fold"] = "Train"
        else:
            split_path = Path(self.splits_file)
            if not split_path.is_file():
                raise FileNotFoundError(f"splits_file does not exist: {split_path}")
            folds = pd.read_csv(split_path).set_index("PID")["fold"]
            frame["fold"] = frame.PID.map(folds).fillna("Train")
        train_frame = frame[frame.fold == "Train"].reset_index(drop=True)
        val_frame = frame[frame.fold == "Validation"].reset_index(drop=True)
        if val_frame.empty:
            val_frame = train_frame.copy()
        print(
            f"[setup] eligible PIDs: train={len(train_frame)} validation={len(val_frame)} "
            f"thresholds={thresholds.tolist()}",
            flush=True,
        )
        algorithm = importlib.import_module("methods." + self.ml)
        rate = utils.infer_sample_rate_hz(self.trace_file)
        uses_history = bool(getattr(algorithm, "history_encoder", False))
        history_window_count = (
            max(1, int(self.history_seconds * rate // self.chunk_window_size))
            if uses_history else 1
        )
        effective_elapsed_time_channel = bool(self.elapsed_time_channel or uses_history)
        if uses_history:
            print(
                f"[setup] historical encoder history={history_window_count} non-overlapping "
                f"windows ({history_window_count * self.chunk_window_size / rate / 3600:.2f}h)",
                flush=True,
            )
        max_duration = float((train_frame.trace_length - self.chunk_window_size).max())
        # Evaluation scores exact 1-hour, 2-hour, and 4-hour horizons in
        # addition to the delivery-relative endpoints. Keep every
        # requested score on the model grid regardless of the user duration
        # list; otherwise _value_at cannot select an exact curve column.
        required_eval_cuts_seconds = np.r_[
            np.asarray([0.0, 20 * 60, 40 * 60, 60 * 60]),
            EVALUATION_HORIZON_SECONDS,
        ]
        required_eval_cuts = required_eval_cuts_seconds * rate
        if self.checkpoint_cuts is not None:
            cuts = np.asarray(self.checkpoint_cuts, dtype=float) * rate
            if (
                cuts.ndim != 1
                or cuts.size == 0
                or not np.isfinite(cuts).all()
                or (cuts < 0).any()
                or not np.all(np.diff(cuts) > 0)
            ):
                raise ValueError(
                    "checkpoint_cuts must be a strictly increasing sequence of "
                    "finite, non-negative delivery cuts in seconds"
                )
            print(
                f"[setup] using checkpoint delivery grid with {len(cuts)} cuts",
                flush=True,
            )
        else:
            cuts = np.unique(
                np.r_[
                    np.asarray(self.durations, dtype=float) * rate,
                    required_eval_cuts,
                    max_duration,
                ]
            )
        labtrans = LabTransDiscreteTime(cuts)
        print(
            f"[setup] computing feature statistics from {len(train_frame)} training traces", flush=True
        )
        stats = utils.compute_hdf5_feature_stats(
            self.trace_file, train_frame.PID, features=self.features,
            missing_data_method="ffill", progress_bar=self.progress_bar,
        )
        if effective_elapsed_time_channel:
            max_elapsed_seconds = float(train_frame.trace_length.max()) / rate
            stats.append({"min": 0.0, "max": max_elapsed_seconds})
        print(f"[setup] feature statistics complete: {stats}", flush=True)
        target_kind = (
            "single" if self.mode == "survival"
            else "competing" if self.mode in {"competing", "competing_threshold"}
            else "marked"
        )
        train_loader = val_loader = None
        train_landmark_loader = None
        train_landmark_sample_count = 0
        if not self.eval_only:
            train_loader = RandomChunkLoader(
                self.trace_file, train_frame, features=self.features, chunk_window_size=self.chunk_window_size,
                batch_size=self.batch_size, labtrans=labtrans, target_kind=target_kind,
                random_state=self.random_state, training=True, feature_stats=stats,
                elapsed_time_feature=self.elapsed_time_feature, sample_rate_hz=rate,
                elapsed_time_scale_seconds=self.elapsed_time_scale_seconds,
                elapsed_time_channel=effective_elapsed_time_channel,
                missingness_indicator_channels=self.missingness_indicator_channels,
                missing_data_method=self.missing_data_method,
                chunk_missingness_max_fraction=self.chunk_missingness_max_fraction,
                chunk_eligibility_cache_dir=self.chunk_eligibility_cache_dir,
                history_window_count=history_window_count,
                validation_landmarks_per_patient=self.validation_landmarks_per_patient,
                training_sampling_strategy=self.training_sampling_strategy,
                training_chunks_per_patient_per_epoch=self.training_chunks_per_patient_per_epoch,
                progress_bar=self.progress_bar,
            )
            val_loader = RandomChunkLoader(
                self.trace_file, val_frame, features=self.features, chunk_window_size=self.chunk_window_size,
                batch_size=self.batch_size, labtrans=labtrans, target_kind=target_kind,
                random_state=self.random_state, training=False, feature_stats=stats,
                elapsed_time_feature=self.elapsed_time_feature, sample_rate_hz=rate,
                elapsed_time_scale_seconds=self.elapsed_time_scale_seconds,
                elapsed_time_channel=effective_elapsed_time_channel,
                missingness_indicator_channels=self.missingness_indicator_channels,
                missing_data_method=self.missing_data_method,
                chunk_missingness_max_fraction=self.chunk_missingness_max_fraction,
                chunk_eligibility_cache_dir=self.chunk_eligibility_cache_dir,
                history_window_count=history_window_count,
                validation_landmarks_per_patient=self.validation_landmarks_per_patient,
                training_sampling_strategy=self.training_sampling_strategy,
                training_chunks_per_patient_per_epoch=self.training_chunks_per_patient_per_epoch,
                progress_bar=self.progress_bar,
            )
            if self.training_landmark_diagnostic_patients:
                diagnostic_count = min(
                    self.training_landmark_diagnostic_patients, len(train_frame)
                )
                diagnostic_frame = train_frame.sample(
                    n=diagnostic_count, random_state=self.random_state,
                ).reset_index(drop=True)
                train_landmark_loader = RandomChunkLoader(
                    self.trace_file, diagnostic_frame, features=self.features,
                    chunk_window_size=self.chunk_window_size, batch_size=self.batch_size,
                    labtrans=labtrans, target_kind=target_kind,
                    random_state=self.random_state, training=False, feature_stats=stats,
                    elapsed_time_feature=self.elapsed_time_feature, sample_rate_hz=rate,
                    elapsed_time_scale_seconds=self.elapsed_time_scale_seconds,
                    elapsed_time_channel=effective_elapsed_time_channel,
                    missingness_indicator_channels=self.missingness_indicator_channels,
                    missing_data_method=self.missing_data_method,
                    chunk_missingness_max_fraction=self.chunk_missingness_max_fraction,
                    chunk_eligibility_cache_dir=self.chunk_eligibility_cache_dir,
                    history_window_count=history_window_count,
                    validation_landmarks_per_patient=self.validation_landmarks_per_patient,
                    training_sampling_strategy=self.training_sampling_strategy,
                    training_chunks_per_patient_per_epoch=self.training_chunks_per_patient_per_epoch,
                    progress_bar=self.progress_bar,
                )
                print(
                    "[setup] training landmark-panel diagnostic: "
                    f"{diagnostic_count:,} PIDs x "
                    f"{len(train_landmark_loader._validation_samples) // diagnostic_count} "
                    f"fixed landmarks = {len(train_landmark_loader):,} inference batches/epoch",
                    flush=True,
                )
                train_landmark_sample_count = len(train_landmark_loader._validation_samples)
        if train_loader is not None and train_loader.eligibility_summary is not None:
            train_summary = train_loader.eligibility_summary
            val_summary = val_loader.eligibility_summary
            print(
                "[setup] chunk eligibility: "
                f"train={train_summary['n_pids_after_filter']}/{train_summary['n_pids']} PIDs, "
                f"valid_starts={train_summary['eligible_chunk_start_fraction']:.1%}, "
                f"validation={val_summary['n_pids_after_filter']}/{val_summary['n_pids']} PIDs, "
                f"valid_starts={val_summary['eligible_chunk_start_fraction']:.1%}",
                flush=True,
            )
        effective_fit_kwargs = dict(self.fit_kwargs)
        if uses_history:
            effective_fit_kwargs["elapsed_time_channel"] = True
            effective_fit_kwargs["n_signal_channels"] = len(self.features)
        if self.ml == "InceptionTime_LSTM_classifier":
            requested_window = int(effective_fit_kwargs.get("window_length", 250))
            if requested_window <= 0:
                raise ValueError("InceptionTime+LSTM window_length must be positive")
            window_length = min(requested_window, int(self.chunk_window_size))
            effective_fit_kwargs["window_length"] = window_length
            overlap = int(effective_fit_kwargs.get("window_overlap", window_length // 6))
            if overlap >= window_length:
                overlap = window_length // 6
            effective_fit_kwargs["window_overlap"] = overlap
            print(
                f"[setup] InceptionTime+LSTM window_length={window_length} "
                f"window_overlap={overlap}",
                flush=True,
            )
        n_input_channels = (
            len(self.features)
            + int(effective_elapsed_time_channel)
            + len(self.features) * int(self.missingness_indicator_channels)
        )
        encoder = algorithm.make_model(
            (self.chunk_window_size, n_input_channels), output_shape=1, **effective_fit_kwargs
        )
        device = torch.device(f"cuda:{self.gpu}" if torch.cuda.is_available() else "cpu")
        if self.mode == "survival":
            # Encoders with an ``fc`` classifier use the native DeepHit head;
            # feature-only encoders use the shared wrapper.
            if self.elapsed_time_feature or not hasattr(encoder, "fc"):
                if not hasattr(encoder, "forward_features"):
                    raise TypeError(
                        "Survival encoders without an fc layer must expose forward_features(x)"
                    )
                net = ElapsedTimeDeepHitNet(encoder, len(cuts), self.elapsed_time_feature)
            else:
                net = encoder
                net.fc = torch.nn.Linear(net.feature_dim, len(cuts))
            model = DeepHitSingle(net, make_optimizer(net.parameters()),
                                  device=device, duration_index=cuts, alpha=self.deephit_alpha, sigma=self.deephit_sigma)
        elif self.mode in {"marked", "marked_corn"}:
            net = MarkedDeepHitNet(
                encoder, len(cuts), len(thresholds),
                mark_head_mode="corn" if self.mode == "marked_corn" else "time_conditioned",
                elapsed_time_feature=self.elapsed_time_feature,
            )
            model = MarkedDeepHitSingle(net, make_optimizer(net.parameters()),
                device=device, duration_index=cuts, alpha=self.deephit_alpha, sigma=self.deephit_sigma,
                mark_loss_weight=self.mark_loss_weight,
                mark_loss_mode="corn" if self.mode == "marked_corn" else "bce")
        else:
            n_causes = 2 if self.mode == "competing_threshold" else len(thresholds) + 1
            competing_net_class = (
                CauseSpecificCompetingDeepHitNet
                if self.competing_head == "cause_specific"
                else CompetingDeepHitNet
            )
            if self.competing_head == "cause_specific":
                net = competing_net_class(
                    encoder,
                    n_causes,
                    len(cuts),
                    hidden_dims=self.competing_head_hidden_dims,
                    batch_norm=self.competing_head_batch_norm,
                    dropout=self.competing_head_dropout,
                    elapsed_time_feature=self.elapsed_time_feature,
                )
            else:
                net = competing_net_class(
                    encoder,
                    n_causes,
                    len(cuts),
                    elapsed_time_feature=self.elapsed_time_feature,
                )
            model = PartialLabelDeepHit(net, make_optimizer(net.parameters()),
                device=device, duration_index=cuts, alpha=self.deephit_alpha, sigma=self.deephit_sigma,
                missing_delivery_loss_weight=self.missing_delivery_loss_weight)
        parameter_count = sum(parameter.numel() for parameter in net.parameters())
        print(
            f"[setup] model ready on {device}: parameters={parameter_count:,} time_bins={len(cuts)}",
            flush=True,
        )
        optimizer_config = {
            "optimizer": self.optimizer,
            "lr_schedule": self.lr_schedule,
            "learning_rate": None if self.learning_rate is None else float(self.learning_rate),
            "lr_warmup_fraction": float(self.lr_warmup_fraction),
            "lr_min_fraction": float(self.lr_min_fraction),
            "lr_minimum": (
                None if self.learning_rate is None
                else float(self.learning_rate * self.lr_min_fraction)
            ),
            "beta2_decay": -0.8,
            "eps1": None,
            "eps2": 1e-3,
            "update_clip_threshold": 1.0,
            "weight_decay": float(self.weight_decay),
            "cocob_alpha": self.cocob_alpha if self.optimizer == "cocob" else None,
            "cocob_gradient_clip_norm": self.cocob_gradient_clip_norm if self.optimizer == "cocob" else None,
            "cocob_max_update_norm": self.cocob_max_update_norm if self.optimizer == "cocob" else None,
            "cocob_max_parameter_abs": self.cocob_max_parameter_abs if self.optimizer == "cocob" else None,
            "cocob_max_rejected_steps": self.cocob_max_rejected_steps if self.optimizer == "cocob" else None,
            "cocob_gradient_multiplier": 1.0 if self.optimizer == "cocob" else None,
        }
        if self.eval_only:
            try:
                self.checkpoint_path = self._resolve_evaluation_checkpoint(net, device)
            except FileNotFoundError as exc:
                if not self.skip_missing_checkpoint:
                    raise
                self._record_skipped_evaluation(exc)
                return
            checkpoint = torch.load(self.checkpoint_path, map_location=device, weights_only=True)
            if not isinstance(checkpoint, dict):
                raise TypeError("checkpoint_path must contain a state_dict mapping")
            state_dict = checkpoint.get("state_dict", checkpoint)
            net.load_state_dict(state_dict)
            net.to(device)
            stem = f"{self.result_stem}_{self.evaluation_name}"
            print(
                f"[evaluation] loaded checkpoint {self.checkpoint_path}; generating landmark reports",
                flush=True,
            )
        else:
            early_stopping_dir = os.path.join("cache", "early_stopping")
            os.makedirs(early_stopping_dir, exist_ok=True)
            early_stopping_path = os.path.join(
                early_stopping_dir, f"{self.mode}_deephit_{self.run_id}.pt"
            )
            training_landmark_monitor = (
                TrainingLandmarkMonitor(train_landmark_loader)
                if train_landmark_loader is not None else None
            )
            history_writer = PeriodicHistoryWriter(
                f"{self.result_stem}_history.csv",
                every_epochs=self.history_write_interval,
            )
            lr_scheduler = (
                WarmupCosineSchedule(
                    total_steps=self.epochs * len(train_loader),
                    initial_lr=self.learning_rate,
                    warmup_fraction=self.lr_warmup_fraction,
                    min_fraction=self.lr_min_fraction,
                )
                if self.lr_schedule == "cosine" else None
            )
            callbacks = [
                *([lr_scheduler] if lr_scheduler is not None else []),
                *([training_landmark_monitor] if training_landmark_monitor is not None else []),
                TrainingProgress(
                    self.epochs,
                    len(train_loader),
                    len(val_loader),
                    gpu_index=self.gpu,
                    training_landmark_monitor=training_landmark_monitor,
                    optimizer_name=self.optimizer,
                    progress_bar=self.progress_bar,
                ),
                history_writer,
            ]
            # Always restore the lowest-validation-loss weights before the
            # final checkpoint and landmark evaluation. A patience of zero
            # disables *termination*, not best-checkpoint selection: use a
            # patience larger than the planned epoch count so the callback
            # runs its normal on_fit_end restore without ending fitting early.
            callbacks.append(tt.callbacks.EarlyStopping(
                patience=(
                    self.early_stopping_patience
                    if self.early_stopping_patience else self.epochs + 1
                ),
                file_path=early_stopping_path,
            ))
            if self.optimizer == "cocob":
                print(
                    "[setup] using guarded COCOB with parameterfree's native "
                    "gradient multiplier (lr=1.0).",
                    flush=True,
                )
            elif lr_scheduler is None:
                print(
                    f"[setup] using fixed {self.optimizer} learning-rate scale={self.learning_rate:g}.",
                    flush=True,
                )
            else:
                print(
                    f"[setup] using cosine {self.optimizer} schedule: "
                    f"initial={self.learning_rate:g}, warmup={lr_scheduler.warmup_steps}/"
                    f"{lr_scheduler.total_steps} updates, minimum="
                    f"{self.learning_rate * self.lr_min_fraction:g}.",
                    flush=True,
                )
            if not self.early_stopping_patience:
                print(
                    "[setup] early stopping disabled (early_stopping_patience=0); "
                    "will restore the best validation checkpoint after the final epoch.",
                    flush=True,
                )
            log = model.fit_dataloader(
                train_loader,
                epochs=self.epochs,
                callbacks=callbacks,
                val_dataloader=val_loader,
            )
            print("[evaluation] training complete; generating landmark reports", flush=True)
            stem = self.result_stem
            torch.save(net.state_dict(), self.auto_checkpoint_path)
            with open(str(Path(self.auto_checkpoint_path).with_suffix("")) + "_training_config.json", "w") as handle:
                json.dump(
                    {
                        "training_key": self.training_key,
                        "checkpoint": self.auto_checkpoint_path,
                        "mode": self.mode,
                        "event_label": self.event_label,
                        "features": list(self.features),
                        "durations": [float(value) for value in self.durations],
                        "cuts": (np.asarray(cuts, dtype=float) / rate).tolist(),
                        "chunk_window_size": int(self.chunk_window_size),
                        "batch_size": int(self.batch_size),
                        "missing_data_method": self.missing_data_method,
                        "missingness_indicator_channels": bool(self.missingness_indicator_channels),
                        "elapsed_time_channel": bool(self.elapsed_time_channel),
                        "elapsed_time_feature": bool(self.elapsed_time_feature),
                        "chunk_missingness_max_fraction": self.chunk_missingness_max_fraction,
                        "model": self._model_metadata(),
                        "imputation": self._imputation_metadata(),
                        "random_state": int(self.random_state),
                        "fit_kwargs": self.fit_kwargs,
                        "optimization": optimizer_config,
                        "restore_best_validation_checkpoint": True,
                        "training_sampling_strategy": self.training_sampling_strategy,
                        "training_chunks_per_patient_per_epoch": (
                            self.training_chunks_per_patient_per_epoch
                        ),
                        "validation_landmarks_per_patient": self.validation_landmarks_per_patient,
                        "validation_panel": self._validation_panel_name(),
                        "training_landmark_diagnostic_patients": self.training_landmark_diagnostic_patients,
                        "training_landmark_diagnostic_samples": (
                            train_landmark_sample_count
                        ),
                    },
                    handle,
                    indent=2,
                )
            history_writer.write()
            pd.DataFrame([optimizer_config]).to_csv(stem + "_optimizer.csv", index=False)
        if self.mode == "survival":
            matched = np.flatnonzero(np.isclose(thresholds, float(self.event_label.rsplit(" ", 1)[-1])))
            if matched.size != 1:
                raise ValueError(f"No pH threshold matches event_label: {self.event_label}")
            target_names = [_threshold_name(thresholds[int(matched[0])])]
            target_indices = [int(matched[0])]
            include_delivery = False
            require_label = True
        elif self.mode == "competing_threshold":
            matched = np.flatnonzero(np.isclose(thresholds, float(self.event_label.rsplit(" ", 1)[-1])))
            if matched.size != 1:
                raise ValueError(f"No pH threshold matches event_label: {self.event_label}")
            target_names = [_threshold_name(thresholds[int(matched[0])])]
            target_indices = [int(matched[0])]
            include_delivery = True
            require_label = False
        else:
            target_names = [_threshold_name(value) for value in thresholds]
            target_indices = list(range(len(thresholds)))
            include_delivery = True
            require_label = False
        prediction = _prediction_function(
            net, mode=self.mode, device=device, feature_stats=stats, cuts=cuts,
            sample_rate_hz=rate, target_names=target_names,
            elapsed_time_feature=self.elapsed_time_feature,
            elapsed_time_scale_seconds=self.elapsed_time_scale_seconds,
            elapsed_time_channel=effective_elapsed_time_channel,
            n_physiological_features=len(self.features),
            history_window_count=history_window_count,
        )
        landmark_metrics, landmark_predictions = evaluate_landmark_suite(
            trace_file=self.trace_file,
            train_label_file=self.label_trainfile,
            test_label_file=self.label_testfile,
            features=self.features,
            chunk_window_size=self.chunk_window_size,
            lab_order_delay=self.lab_order_delay,
            missingness_indicator_channels=self.missingness_indicator_channels,
            missing_data_method=self.missing_data_method,
            chunk_missingness_max_fraction=self.chunk_missingness_max_fraction,
            history_window_count=history_window_count,
            prediction=prediction,
            target_names=target_names,
            target_indices=target_indices,
            include_delivery=include_delivery,
            splits_file=self.splits_file,
            require_label=require_label,
            bootstrap_samples=int(self.bootstrap_samples),
            bootstrap_confidence_level=float(self.bootstrap_confidence_level),
            bootstrap_random_state=int(self.random_state),
        )
        print(
            f"[evaluation] wrote {len(landmark_metrics)} metric rows and "
            f"{len(landmark_predictions)} prediction rows",
            flush=True,
        )
        landmark_metrics.to_csv(stem + "_landmark_metrics.csv", index=False)
        landmark_predictions.to_csv(stem + "_landmark_predictions.csv", index=False)
        with open(stem + ".json", "w") as handle:
            json.dump({"mode": self.mode, "event_label": self.event_label,
                       "model": self._model_metadata(),
                       "imputation": self._imputation_metadata(),
                       "random_state": int(self.random_state),
                       "artifact_label": self.artifact_label,
                       "trace_file": self.trace_file, "thresholds": thresholds.tolist(), "cuts": cuts.tolist(),
                       "feature_stats": stats,
                       "batch_size": int(self.batch_size),
                       "chunk_window_size": int(self.chunk_window_size),
                       "epochs": int(self.epochs),
                       "history_write_interval": int(self.history_write_interval),
                       "n_train_pids": len(train_loader.frame) if train_loader is not None else len(train_frame),
                       "n_validation_pids": len(val_loader.frame) if val_loader is not None else len(val_frame),
                       "elapsed_time_feature": bool(self.elapsed_time_feature),
                       "elapsed_time_scale_seconds": float(self.elapsed_time_scale_seconds),
                       "elapsed_time_channel": bool(effective_elapsed_time_channel),
                       "history_seconds": self.history_seconds if uses_history else None,
                       "history_window_count": history_window_count,
                       "validation_landmarks_per_patient": self.validation_landmarks_per_patient,
                       "validation_panel": self._validation_panel_name(),
                       "training_sampling_strategy": self.training_sampling_strategy,
                       "training_chunks_per_patient_per_epoch": (
                           self.training_chunks_per_patient_per_epoch
                       ),
                       "mark_loss_weight": float(self.mark_loss_weight),
                       "missingness_indicator_channels": self.missingness_indicator_channels,
                       "missing_data_method": self.missing_data_method,
                       "chunk_missingness_max_fraction": self.chunk_missingness_max_fraction,
                       "chunk_eligibility_cache_dir": self.chunk_eligibility_cache_dir,
                       "train_chunk_eligibility": train_loader.eligibility_summary if train_loader is not None else None,
                       "validation_chunk_eligibility": val_loader.eligibility_summary if val_loader is not None else None,
                       "optimizer": self.optimizer,
                       "mark_loss_weight": float(self.mark_loss_weight),
                       "training_key": self.training_key,
                       "checkpoint": self.auto_checkpoint_path if not self.eval_only else self.checkpoint_path,
                       "eval_only": self.eval_only,
                       "source_checkpoint": self.checkpoint_path if self.eval_only else None,
                       "evaluation_name": self.evaluation_name if self.eval_only else None,
                       "optimizer_config": {
                           **optimizer_config,
                           "early_stopping_patience": int(self.early_stopping_patience),
                           "restore_best_validation_checkpoint": True,
                       },
                       "optimizer_settings": stem + "_optimizer.csv" if not self.eval_only else None,
                       "bootstrap": {
                           "samples": int(self.bootstrap_samples),
                           "confidence_level": float(self.bootstrap_confidence_level),
                           "random_state": int(self.random_state),
                       },
                       "landmark_metrics": stem + "_landmark_metrics.csv",
                       "landmark_predictions": stem + "_landmark_predictions.csv"}, handle, indent=2)


if __name__ == "__main__":
    fire.Fire(Trainer)
