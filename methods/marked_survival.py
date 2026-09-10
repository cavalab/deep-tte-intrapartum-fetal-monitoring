"""Marked DeepHit utilities for delivery time and cord-pH outcomes.

The delivery process is modelled with PyCox's unchanged DeepHit likelihood and
ranking loss. A second, masked head models the pH mark conditional on the
delivery-time bin.
"""
from __future__ import annotations

import re
from contextlib import nullcontext
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchtuples as tt

from pycox.models import DeepHitSingle
from pycox.models.data import DeepHitDataset, pair_rank_mat
from pycox.models.utils import pad_col
from pycox.preprocessing.label_transforms import LabTransDiscreteTime

import utils


_PH_THRESHOLD_RE = re.compile(r"^pH Cord\s*<\s*([0-9.]+)\s*$")
REQUIRED_EVALUATION_CUT_SECONDS = np.asarray([0.0, 20 * 60, 40 * 60, 60 * 60])


def _normalise_pid(value):
    """Normalize HDF5 byte/string PID representations for joins and lookups."""
    if isinstance(value, bytes):
        return value.decode("utf-8")
    value = str(value)
    if value.startswith("b'") and value.endswith("'"):
        return value[2:-1]
    return value


def discover_ph_threshold_columns(columns):
    """Discover sorted pH-threshold label columns.

    Parameters
    ----------
    columns : Iterable[str]
        Candidate table column names.

    Returns
    -------
    list[tuple[float, str]]
        Numeric threshold/source-column pairs ordered from low to high pH.
    """
    found = []
    for col in columns:
        match = _PH_THRESHOLD_RE.match(str(col))
        if match:
            found.append((float(match.group(1)), col))
    if not found:
        raise ValueError("No columns matching 'pH Cord < <threshold>' were found.")
    thresholds = [x[0] for x in found]
    if len(set(thresholds)) != len(thresholds):
        raise ValueError("pH threshold columns contain duplicate numeric thresholds.")
    return sorted(found)


def cumulative_ph_threshold_targets(frame, threshold_columns=None, raw_ph_column="pH Cord"):
    """Create ordered cumulative pH targets and a row-level observed-mark mask.

    Targets are derived from the authoritative threshold columns when present;
    observed raw pH is retained solely for missingness/auditing and validation.

    Parameters
    ----------
    frame : pandas.DataFrame
        Rows containing raw pH and threshold columns.
    threshold_columns : sequence[str] | None
        Explicit label columns, or ``None`` to discover them from ``frame``.
    raw_ph_column : str, default="pH Cord"
        Raw pH column used for missingness and label validation.

    Returns
    -------
    tuple
        Sorted thresholds, source column names, binary cumulative targets,
        observed-pH mask, and raw pH values.
    """
    pairs = discover_ph_threshold_columns(
        frame.columns if threshold_columns is None else threshold_columns
    )
    thresholds, columns = zip(*pairs)
    values = frame.loc[:, list(columns)].apply(pd.to_numeric, errors="coerce").to_numpy(float)
    raw_ph = pd.to_numeric(frame[raw_ph_column], errors="coerce").to_numpy(float)
    mask = np.isfinite(raw_ph)
    # A pH below a smaller threshold must also be below every larger threshold.
    targets = np.nan_to_num(values, nan=0.0).astype(np.float32)
    targets = (targets > 0).astype(np.float32)
    if mask.any():
        expected = (raw_ph[mask, None] < np.asarray(thresholds)[None, :]).astype(np.float32)
        if not np.array_equal(targets[mask], expected):
            raise ValueError("Threshold labels disagree with raw pH Cord values.")
    return np.asarray(thresholds, dtype=float), list(columns), targets, mask.astype(bool), raw_ph


def _duration_cuts(raw_durations, sample_rate_hz, durations=None, num_durations=100):
    """Build sample-index delivery cuts from raw durations or seconds cutoffs."""
    raw_durations = np.asarray(raw_durations, dtype=float)
    if raw_durations.size == 0 or not np.isfinite(raw_durations).all() or (raw_durations < 0).any():
        raise ValueError("Delivery durations must be finite and non-negative.")
    max_duration = float(raw_durations.max())
    if durations is None:
        # PyCox's equidistant fitting convention, with an explicit zero cut for
        # chunks at delivery.
        n = int(num_durations)
        if n < 2:
            raise ValueError("num_durations must be at least 2.")
        cuts = np.linspace(0.0, max_duration, n)
    else:
        cuts = np.asarray(durations, dtype=float) * float(sample_rate_hz)
        if cuts.ndim != 1 or cuts.size == 0 or not np.isfinite(cuts).all() or (cuts < 0).any():
            raise ValueError("durations must be a finite, non-negative 1D list in seconds.")
        cuts = np.unique(cuts)
        if cuts[0] != 0:
            cuts = np.r_[0.0, cuts]
        if cuts[-1] < max_duration:
            cuts = np.r_[cuts, max_duration]
    # Exact bins make the 0/20/40/60-minute end-landmark predictions
    # reproducible rather than dependent on nearest-grid interpolation.
    required = REQUIRED_EVALUATION_CUT_SECONDS * float(sample_rate_hz)
    cuts = np.unique(np.r_[cuts, required])
    return np.asarray(cuts, dtype=float)


def load_hdf5_chunked_optional_labels(
    trace_file,
    label_file,
    *,
    features,
    label_columns,
    missing="ffill",
    horizon=0,
    lab_order_delay=30,
    chunk_window_size=900,
    overlap=0,
    duration_filter=None,
    elapsed_time_channel=False,
):
    """Chunk every tracing and attach lab labels when they are available.

    This loader deliberately uses a left join. A late, absent, or pH-missing lab creates an
    unobserved mark rather than removing an otherwise valid delivery tracing.
    """
    if chunk_window_size <= 0 or overlap < 0 or overlap >= chunk_window_size:
        raise ValueError("chunk_window_size must be positive and exceed overlap")
    trace_file, label_file = str(trace_file), str(label_file)
    sample_rate_hz = utils.infer_sample_rate_hz(trace_file)
    is_hdf5 = trace_file.endswith((".h5", ".hdf5"))
    if is_hdf5:
        traces = utils.load_hdf5_index(trace_file)
        traces["PID"] = traces["PID"].map(_normalise_pid)
    else:
        raise ValueError("Only HDF5 trace stores are supported.")
    labels = utils.load_label_table(label_file)
    required_columns = ["PID", *label_columns]
    missing_columns = set(required_columns).difference(labels.columns)
    if missing_columns:
        raise ValueError(f"Label file is missing columns: {sorted(missing_columns)}")

    # Timeliness controls whether the mark is observed, never whether the
    # tracing contributes to delivery-time learning.
    if "labs" in label_file and lab_order_delay > 0:
        if "last_time" not in labels.columns:
            if "last_time" not in traces.columns:
                raise ValueError(
                    "Lab-delay filtering requires 'last_time' in either the label or trace file."
                )
            labels = labels.merge(
                traces.loc[:, ["PID", "last_time"]], on="PID", how="left"
            )
        labels = utils.filter_labs_by_time_delay(
            labels.copy(), lab_order_delay=lab_order_delay, horizon=horizon
        )
    labels = labels.loc[:, required_columns].copy()
    if labels["PID"].duplicated().any():
        raise ValueError(
            "Optional-label loading requires at most one eligible lab row per PID."
        )
    labels["PID"] = labels["PID"].map(_normalise_pid)
    labels_by_pid = labels.set_index("PID")

    stride = chunk_window_size - overlap
    chunks, rows, pids = [], [], []
    h5_context = nullcontext(None)
    if is_hdf5:
        import h5py
        h5_context = h5py.File(trace_file, "r")
    with h5_context as h5f:
        for _, trace in traces.iterrows():
            pid = _normalise_pid(trace["PID"])
            arrays = []
            hdf_trace = utils._read_hdf5_trace(h5f, pid, features=features) if h5f else None
            for feature_index, feature in enumerate(features):
                values = (
                    np.asarray(hdf_trace[:, feature_index], dtype=float).ravel()
                    if hdf_trace is not None
                    else np.asarray(trace[feature], dtype=float).ravel()
                )
                if missing == "ffill":
                    values = pd.Series(values).ffill().bfill().to_numpy()
                elif missing == "zeros":
                    values = np.nan_to_num(values, nan=0.0)
                elif missing == "drop":
                    values = pd.Series(values).dropna().to_numpy()
                arrays.append(values)
            if elapsed_time_channel:
                total_elapsed_length = max((len(values) for values in arrays), default=0)
                arrays.append(np.arange(total_elapsed_length, dtype=float) / sample_rate_hz)
            total_len = max((len(values) for values in arrays), default=0)
            if duration_filter is not None:
                minimum, maximum = duration_filter
                seconds = total_len / float(sample_rate_hz)
                if (minimum is not None and seconds < minimum) or (
                    maximum is not None and seconds >= maximum
                ):
                    continue
            if total_len < chunk_window_size:
                continue
            starts = list(range(0, total_len - chunk_window_size + 1, stride))
            final_start = total_len - chunk_window_size
            if starts[-1] != final_start:
                starts.append(final_start)
            if pid in labels_by_pid.index:
                label_values = labels_by_pid.loc[pid].to_dict()
            else:
                label_values = {column: np.nan for column in label_columns}
            for start in starts:
                sample = np.full((chunk_window_size, len(arrays)), np.nan, dtype=float)
                for feature_index, values in enumerate(arrays):
                    end = min(start + chunk_window_size, len(values))
                    sample[: end - start, feature_index] = values[start:end]
                chunks.append(sample)
                pids.append(pid)
                rows.append(
                    {
                        "duration": min(start + chunk_window_size, total_len),
                        "end_of_trace": min(start + chunk_window_size, total_len) == total_len,
                        **label_values,
                    }
                )
    if not chunks:
        raise ValueError("No trace chunks were available after optional-label loading.")
    return np.stack(chunks), pd.DataFrame(rows), np.asarray(pids)


@dataclass
class MarkedSurvivalData:
    X: np.ndarray
    duration: np.ndarray
    duration_idx: np.ndarray
    delivery_event: np.ndarray
    ph: np.ndarray
    ph_threshold_targets: np.ndarray
    mark_mask: np.ndarray
    PIDs: np.ndarray
    labtrans: LabTransDiscreteTime
    thresholds: np.ndarray
    threshold_columns: list
    scalers: list | None = None

    def target(self):
        """Return the delivery/mark target tuple used by marked DeepHit."""
        return (self.duration_idx, self.delivery_event, self.ph_threshold_targets, self.mark_mask)


def load_marked_survival_data(
    trace_file, label_file, features=("toco", "fecg"), missing="ffill", random_state=42,
    horizon=0, lab_order_delay=30, chunk_window_size=900, overlap=0,
    durations=None, num_durations=100, duration_filter=None, scale=False,
    elapsed_time_channel=False,
):
    """Load marked-survival chunks without dropping samples with missing pH.

    Parameters
    ----------
    trace_file, label_file : str
        Tracing and lab data sources.
    features : sequence[str]
        Signal channels included in each chunk.
    missing, horizon, lab_order_delay, chunk_window_size, overlap :
        Chunking and lab-timeliness configuration.
    durations, num_durations : sequence[float] | None, int
        Delivery-time cuts in seconds or the fallback number of cuts.
    duration_filter, scale, elapsed_time_channel : optional
        Trace eligibility and feature-transformation controls.

    Returns
    -------
    MarkedSurvivalData
        Features plus aligned delivery, pH mark, PID, and discretization data.
    """
    trace_file = str(trace_file)
    label_file = str(label_file)
    labels_df = utils.load_label_table(label_file)
    pairs = discover_ph_threshold_columns(labels_df.columns)
    threshold_columns = [c for _, c in pairs]
    if "pH Cord" not in labels_df.columns:
        raise ValueError("Marked survival requires the raw 'pH Cord' column.")
    # The optional-label path retains trace-only PIDs.  Existing loaders retain
    # their inner-join semantics for tasks requiring every label.
    X, y, pids = load_hdf5_chunked_optional_labels(
        trace_file,
        label_file,
        features=list(features),
        label_columns=["pH Cord", *threshold_columns],
        missing=missing,
        horizon=horizon,
        lab_order_delay=lab_order_delay,
        chunk_window_size=chunk_window_size,
        overlap=overlap,
        duration_filter=duration_filter,
        elapsed_time_channel=elapsed_time_channel,
    )
    sample_rate_hz = utils.infer_sample_rate_hz(trace_file)
    observed_end = pd.to_numeric(y["duration"], errors="coerce").to_numpy(float)
    if not np.isfinite(observed_end).all():
        raise ValueError("Chunk end durations must be finite.")
    trace_end = pd.Series(observed_end).groupby(pd.Series(pids)).transform("max").to_numpy(float)
    duration = trace_end - observed_end
    if (duration < 0).any():
        raise ValueError("Calculated time-to-delivery contains negative durations.")
    cuts = _duration_cuts(duration, sample_rate_hz, durations, num_durations)
    labtrans = LabTransDiscreteTime(cuts)
    delivery_event = np.ones(len(duration), dtype=np.int64)
    duration_idx, delivery_event = labtrans.transform(duration, delivery_event)
    thresholds, threshold_columns, targets, mark_mask, raw_ph = cumulative_ph_threshold_targets(y)

    scalers = None
    if scale:
        from sklearn.preprocessing import MinMaxScaler
        scalers = []
        for i in range(X.shape[2]):
            scaler = MinMaxScaler(feature_range=(-1, 1))
            X[:, :, i] = scaler.fit_transform(X[:, :, i])
            scalers.append(scaler)
    return MarkedSurvivalData(X, duration.astype(float), np.asarray(duration_idx, dtype=np.int64),
                              np.asarray(delivery_event, dtype=np.int64), raw_ph, targets,
                              mark_mask, np.asarray(pids), labtrans, thresholds,
                              threshold_columns, scalers)


# Alias with the existing loader's naming pattern.
load_data_marked_survival = load_marked_survival_data


class MarkedDeepHitNet(nn.Module):
    """An encoder with delivery and conditional pH-mark heads.

    ``mark_head_mode='corn'`` has the same time-indexed output shape as the
    ordinary ``'time_conditioned'`` head, but each output is a CORN
    conditional ordinal transition rather than an independent pH-threshold
    probability.
    """
    def __init__(
        self, encoder, n_time_bins, n_thresholds, mark_head_mode="time_conditioned",
        elapsed_time_feature=False,
    ):
        """Build shared encoder, delivery head, and conditional pH mark head.

        When ``elapsed_time_feature`` is enabled, ``forward`` also requires a
        normalized elapsed-time scalar for each tracing chunk. It is appended
        after the physiological encoder, rather than treated as a signal.
        """
        super().__init__()
        if not hasattr(encoder, "forward_features"):
            raise TypeError("encoder must expose forward_features(x)")
        if mark_head_mode not in {"time_conditioned", "time_independent", "corn"}:
            raise ValueError(
                "mark_head_mode must be 'time_conditioned', 'time_independent', or 'corn'"
            )
        self.encoder = encoder
        self.n_time_bins = int(n_time_bins)
        self.n_thresholds = int(n_thresholds)
        self.mark_head_mode = mark_head_mode
        self.elapsed_time_feature = bool(elapsed_time_feature)
        feature_dim = getattr(encoder, "feature_dim", None)
        if feature_dim is None and hasattr(encoder, "fc"):
            feature_dim = encoder.fc.in_features
        if feature_dim is None:
            raise TypeError("encoder must expose feature_dim or an fc layer")
        head_input_dim = feature_dim + int(self.elapsed_time_feature)
        self.delivery_head = nn.Linear(head_input_dim, self.n_time_bins)
        out = self.n_thresholds * (
            self.n_time_bins if mark_head_mode in {"time_conditioned", "corn"} else 1
        )
        self.mark_head = nn.Linear(head_input_dim, out)

    def forward(self, x, elapsed_time=None):
        """Return delivery logits and independent or time-conditioned mark logits."""
        features = self.encoder.forward_features(x)
        if self.elapsed_time_feature:
            if elapsed_time is None:
                raise ValueError("elapsed_time is required when elapsed_time_feature is enabled")
            elapsed_time = elapsed_time.reshape(features.shape[0], 1).to(features.dtype)
            features = torch.cat([features, elapsed_time], dim=1)
        delivery = self.delivery_head(features)
        marks = self.mark_head(features)
        if self.mark_head_mode in {"time_conditioned", "corn"}:
            marks = marks.reshape(-1, self.n_time_bins, self.n_thresholds)
        return delivery, marks


class MarkedDeepHitDataset(DeepHitDataset):
    """DeepHitDataset plus pH targets/mask; rank matrices remain PyCox's own."""
    def __getitem__(self, index):
        """Fetch an item and add PyCox's batch-specific ranking matrix."""
        input_, target = tt.data.DatasetTuple.__getitem__(self, index)
        duration_idx, event, marks, mark_mask = target.to_numpy()
        # This is the exact function called by PyCox's DeepHitDataset.
        rank_mat = pair_rank_mat(duration_idx, event)
        target = tt.tuplefy(duration_idx, event, rank_mat, marks, mark_mask).to_tensor()
        return tt.tuplefy(input_, target)


def selected_mark_logits(mark_logits, duration_idx):
    """Select delivery-bin mark logits for each batch row.

    Parameters
    ----------
    mark_logits : torch.Tensor
        Independent ``[B, K]`` or time-conditioned ``[B, T, K]`` logits.
    duration_idx : torch.Tensor
        Observed delivery-bin indices.

    Returns
    -------
    torch.Tensor
        Selected mark logits shaped ``[B, K]``.
    """
    if mark_logits.ndim == 2:
        return mark_logits
    if mark_logits.ndim != 3:
        raise ValueError("mark logits must have shape [B,K] or [B,T,K]")
    idx = duration_idx.long().clamp(0, mark_logits.shape[1] - 1)
    return mark_logits[torch.arange(len(idx), device=idx.device), idx]


def masked_mark_bce(mark_logits, duration_idx, targets, mark_mask):
    """Compute mark BCE only where cord pH is observed.

    Parameters
    ----------
    mark_logits, duration_idx, targets, mark_mask : torch.Tensor
        Mark outputs, delivery bins, cumulative pH labels, and row-level pH
        observation indicators.

    Returns
    -------
    torch.Tensor
        Scalar BCE or a differentiable zero when no marks are observed.
    """
    logits = selected_mark_logits(mark_logits, duration_idx)
    mask = mark_mask.bool().reshape(-1)
    if not torch.any(mask):
        return logits.sum() * 0.0
    return F.binary_cross_entropy_with_logits(logits[mask], targets.float()[mask])


def masked_corn_loss(mark_logits, duration_idx, targets, mark_mask):
    """Compute the observed-pH CORN ordinal loss at the delivery-time bin.

    The input targets are cumulative pH indicators in ascending threshold
    order (``pH < 7.05``, ..., ``pH < 7.20``).  Their sum is an ordinal
    severity category: zero for pH at or above the largest threshold and
    ``K`` for pH below the smallest.  CORN learns the sequential transitions
    ``P(category >= j | category >= j - 1, X, delivery bin)``.  The loss is
    masked entirely for deliveries with unknown pH.
    """
    logits = selected_mark_logits(mark_logits, duration_idx)
    observed = mark_mask.bool().reshape(-1)
    if not torch.any(observed):
        return logits.sum() * 0.0
    logits = logits[observed]
    category = targets[observed].float().sum(dim=1).round().long()
    transition = torch.arange(1, logits.shape[1] + 1, device=logits.device)
    eligible = category[:, None] >= (transition - 1)
    transition_targets = (category[:, None] >= transition).to(logits.dtype)
    return F.binary_cross_entropy_with_logits(
        logits[eligible], transition_targets[eligible]
    )


def corn_cumulative_mark_probabilities(mark_logits):
    """Convert CORN transition logits to ordered cumulative pH risks.

    Returns ``P(pH < threshold | delivery bin, X)`` in the repository's
    ascending threshold order.  Products of the sequential CORN transitions
    guarantee nondecreasing risks across pH thresholds.
    """
    if mark_logits.shape[-1] < 1:
        raise ValueError("CORN mark logits must contain at least one transition")
    if torch.is_tensor(mark_logits):
        severe_tail = torch.sigmoid(mark_logits).cumprod(dim=-1)
        return severe_tail.flip(dims=(-1,))
    severe_tail = (1.0 / (1.0 + np.exp(-mark_logits))).cumprod(axis=-1)
    return np.flip(severe_tail, axis=-1).copy()


class MarkedDeepHitSingle(DeepHitSingle):
    """DeepHitSingle with a masked independent or CORN ordinal mark loss."""
    def __init__(self, *args, mark_loss_weight=1.0, mark_loss_mode="bce", **kwargs):
        """Initialize DeepHit with an additional masked pH-mark loss weight."""
        self.mark_loss_weight = float(mark_loss_weight)
        if mark_loss_mode not in {"bce", "corn"}:
            raise ValueError("mark_loss_mode must be 'bce' or 'corn'")
        self.mark_loss_mode = mark_loss_mode
        super().__init__(*args, **kwargs)

    def make_dataloader(self, data, batch_size, shuffle, num_workers=0):
        """Create a loader using the marked dataset and PyCox rank construction."""
        return super(DeepHitSingle, self).make_dataloader(
            data, batch_size, shuffle, num_workers, make_dataset=MarkedDeepHitDataset
        )

    def compute_metrics(self, data, metrics=None):
        """Compute DeepHit delivery loss plus observed-row pH mark BCE."""
        input_, target = data
        input_, target = self._to_device(input_), self._to_device(target)
        delivery_logits, mark_logits = self.net(*input_)
        duration_idx, event, rank_mat, marks, mark_mask = target
        delivery_loss = self.loss(delivery_logits, duration_idx, event, rank_mat)
        mark_loss = (
            masked_corn_loss(mark_logits, duration_idx, marks, mark_mask)
            if self.mark_loss_mode == "corn"
            else masked_mark_bce(mark_logits, duration_idx, marks, mark_mask)
        )
        total = delivery_loss + self.mark_loss_weight * mark_loss
        # Training/early stopping should continue to use the familiar ``loss``.
        return {"loss": total, "delivery_loss": delivery_loss.detach(), "mark_loss": mark_loss.detach()}

    def predict_pmf(self, input, batch_size=8224, numpy=None, eval_=True, to_cpu=False, num_workers=0):
        """Predict delivery PMF for input chunks using the delivery head."""
        outputs = self.predict(input, batch_size, False, eval_, False, to_cpu, num_workers)
        delivery_logits = outputs[0] if isinstance(outputs, (tuple, list)) else outputs[0]
        pmf = pad_col(delivery_logits).softmax(1)[:, :-1]
        return tt.utils.array_or_tensor(pmf, numpy, input)

    def predict_mark_logits(self, input, batch_size=8224, numpy=True):
        """Predict raw pH mark logits for input chunks."""
        outputs = self.predict(input, batch_size, numpy=numpy)
        return outputs[1]


def delivery_pmf(delivery_logits):
    """Convert DeepHit delivery logits to a time-bin PMF.

    Parameters
    ----------
    delivery_logits : numpy.ndarray | torch.Tensor
        Logits shaped ``[B, T]``.

    Returns
    -------
    numpy.ndarray | torch.Tensor
        Delivery PMF shaped ``[B, T]`` without the survival-tail cell.
    """
    if isinstance(delivery_logits, np.ndarray):
        return torch.softmax(torch.from_numpy(np.c_[delivery_logits, np.zeros((len(delivery_logits), 1))]), 1).numpy()[:, :-1]
    return torch.softmax(F.pad(delivery_logits, (0, 1)), 1)[:, :-1]


def conditional_mark_probabilities(mark_logits):
    """Transform pH mark logits to conditional probabilities.

    Parameters
    ----------
    mark_logits : numpy.ndarray | torch.Tensor
        Independent or time-conditioned mark logits.

    Returns
    -------
    numpy.ndarray | torch.Tensor
        Sigmoid probabilities with the input shape.
    """
    return torch.sigmoid(mark_logits) if torch.is_tensor(mark_logits) else 1.0 / (1.0 + np.exp(-mark_logits))


def joint_bin_risks(pmf, conditional_marks):
    """Compute joint pH-threshold and delivery-bin risks.

    Parameters
    ----------
    pmf : array-like
        Delivery PMF shaped ``[B, T]``.
    conditional_marks : array-like
        Mark probabilities shaped ``[B, K]`` or ``[B, T, K]``.

    Returns
    -------
    array-like
        Joint risks shaped ``[B, T, K]``.
    """
    if torch.is_tensor(pmf):
        marks = conditional_marks if conditional_marks.ndim == 3 else conditional_marks[:, None, :].expand(-1, pmf.shape[1], -1)
        return pmf[:, :, None] * marks
    marks = conditional_marks if conditional_marks.ndim == 3 else np.repeat(conditional_marks[:, None, :], pmf.shape[1], axis=1)
    return pmf[:, :, None] * marks


def joint_cumulative_risks(pmf, conditional_marks, horizons=None):
    """Accumulate joint risks over time bins.

    Parameters
    ----------
    pmf, conditional_marks : array-like
        Inputs accepted by :func:`joint_bin_risks`.
    horizons : sequence[int] | None
        Optional time-bin indices to retain.

    Returns
    -------
    array-like
        Cumulative joint risks shaped ``[B, T, K]``.
    """
    risks = joint_bin_risks(pmf, conditional_marks).cumsum(axis=1) if isinstance(pmf, np.ndarray) else joint_bin_risks(pmf, conditional_marks).cumsum(1)
    return risks if horizons is None else risks[:, horizons]
