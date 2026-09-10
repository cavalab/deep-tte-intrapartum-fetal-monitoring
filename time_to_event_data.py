"""Streaming sampled-chunk batches for variable-length survival traces."""
import hashlib
import json
import os
from math import ceil
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torchtuples as tt
from pycox.models.data import pair_rank_mat
from tqdm.auto import tqdm

import utils
from methods.marked_survival import (
    _normalise_pid,
    cumulative_ph_threshold_targets,
    discover_ph_threshold_columns,
)


def _normalise_missingness_fraction(value):
    """Return an optional missing-data fraction from a fraction or percentage."""
    if value is None or (isinstance(value, str) and value.strip().lower() in {"", "none", "null"}):
        return None
    fraction = float(value)
    if fraction > 1:
        fraction /= 100.0
    if not 0 <= fraction <= 1:
        raise ValueError("chunk_missingness_max_fraction must be between 0 and 1 (or 0 and 100)")
    return fraction


def _true_ranges(mask):
    """Encode a Boolean vector as inclusive ``[start, end]`` true ranges."""
    mask = np.asarray(mask, dtype=bool)
    if not mask.any():
        return np.empty((0, 2), dtype=np.int32)
    transitions = np.diff(np.r_[False, mask, False].astype(np.int8))
    starts = np.flatnonzero(transitions == 1)
    ends = np.flatnonzero(transitions == -1) - 1
    return np.column_stack([starts, ends]).astype(np.int32, copy=False)


def _eligibility_cache_file(trace_file, frame, *, features, chunk_window_size,
                            max_missing_fraction, cache_dir):
    """Return a content-addressed sidecar path for one eligibility index."""
    trace_path = Path(trace_file).resolve()
    stat = trace_path.stat()
    identity = {
        "version": 1,
        "trace_file": str(trace_path),
        "trace_size": stat.st_size,
        "trace_mtime_ns": stat.st_mtime_ns,
        "pids": sorted(frame.PID.astype(str).tolist()),
        "features": list(features),
        "chunk_window_size": int(chunk_window_size),
        "max_missing_fraction": float(max_missing_fraction),
    }
    digest = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()[:20]
    return Path(cache_dir) / f"chunk_eligibility_{digest}.npz", identity


def _decode_eligibility_cache(path, identity):
    """Read a validated compressed eligibility sidecar, if one exists."""
    if not path.exists():
        return None
    try:
        with np.load(path, allow_pickle=False) as data:
            cached_identity = json.loads(str(data["identity"][0]))
            if cached_identity != identity:
                return None
            pids = data["pids"].astype(str)
            offsets = data["offsets"].astype(np.int64)
            ranges = data["ranges"].astype(np.int32)
    except (OSError, ValueError, KeyError, json.JSONDecodeError):
        return None
    if len(offsets) != len(pids) + 1 or offsets[-1] != len(ranges):
        return None
    return {
        pid: ranges[offsets[index]:offsets[index + 1]]
        for index, pid in enumerate(pids)
    }


def _write_eligibility_cache(path, identity, ranges_by_pid):
    """Atomically persist compressed PID-to-valid-start-range mappings."""
    path.parent.mkdir(parents=True, exist_ok=True)
    pids = np.asarray(sorted(ranges_by_pid), dtype=str)
    range_blocks = [ranges_by_pid[pid] for pid in pids]
    offsets = np.r_[0, np.cumsum([len(block) for block in range_blocks], dtype=np.int64)]
    ranges = (
        np.concatenate(range_blocks, axis=0)
        if offsets[-1]
        else np.empty((0, 2), dtype=np.int32)
    )
    temporary = path.with_name(f".{path.stem}.{os.getpid()}.tmp.npz")
    np.savez_compressed(
        temporary,
        identity=np.asarray([json.dumps(identity, sort_keys=True)]),
        pids=pids,
        offsets=offsets,
        ranges=ranges,
    )
    os.replace(temporary, path)


def precompute_eligible_chunk_ranges(
    trace_file, frame, *, features, chunk_window_size, max_missing_fraction,
    cache_dir="cache/chunk_eligibility", progress_bar=False,
):
    """Build or load a compact index of chunk starts passing a missingness rule.

    Missingness is calculated from raw values before forward filling.  Every
    possible fixed-width start is evaluated in linear time per trace with a
    prefix sum; valid starts are stored as contiguous inclusive ranges rather
    than one row per nearly identical window.

    Parameters
    ----------
    trace_file : str
        HDF5 variable-length tracing store.
    frame : pandas.DataFrame
        One metadata row per PID to index.
    features : sequence[str]
        Raw signal channels included in the missingness numerator and
        denominator.
    chunk_window_size : int
        Fixed chunk width in samples.
    max_missing_fraction : float
        Maximum fraction of missing values across all requested channels.
    cache_dir : str or pathlib.Path
        Directory for a versioned compressed sidecar cache.
    progress_bar : bool, default=False
        Show progress while scanning the HDF5 traces.

    Returns
    -------
    tuple[dict[str, numpy.ndarray], dict]
        PID-to-``[start, end]`` valid-start ranges and coverage counts.
    """
    import h5py

    fraction = _normalise_missingness_fraction(max_missing_fraction)
    if fraction is None:
        raise ValueError("max_missing_fraction is required when precomputing eligibility")
    if frame.PID.astype(str).duplicated().any():
        raise ValueError("Chunk eligibility requires one metadata row per PID")
    cache_file, identity = _eligibility_cache_file(
        trace_file, frame, features=features, chunk_window_size=chunk_window_size,
        max_missing_fraction=fraction, cache_dir=cache_dir,
    )
    cached = _decode_eligibility_cache(cache_file, identity)
    cache_hit = cached is not None
    if cached is None:
        # A sweep can start many jobs with the same data configuration at once.
        # Lock the sidecar so only one process performs the HDF5 scan.
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        lock_file = cache_file.with_suffix(cache_file.suffix + ".lock")
        with open(lock_file, "a+") as lock:
            import fcntl
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
            cached = _decode_eligibility_cache(cache_file, identity)
            if cached is not None:
                ranges_by_pid = cached
                cache_hit = True
            else:
                ranges_by_pid = {}
                with h5py.File(trace_file, "r") as h5f:
                    scan = frame.itertuples(index=False)
                    if progress_bar:
                        scan = tqdm(
                            scan,
                            total=len(frame),
                            desc="[h5] indexing traces",
                            unit="trace",
                            dynamic_ncols=True,
                        )
                    for row in scan:
                        trace = utils._read_hdf5_trace(h5f, row.PID, features=features)
                        n_starts = len(trace) - int(chunk_window_size) + 1
                        if n_starts <= 0:
                            ranges = np.empty((0, 2), dtype=np.int32)
                        else:
                            missing_per_time = (~np.isfinite(trace)).sum(axis=1, dtype=np.int32)
                            cumulative = np.r_[0, np.cumsum(missing_per_time, dtype=np.int64)]
                            missing_per_window = cumulative[chunk_window_size:] - cumulative[:-chunk_window_size]
                            valid = missing_per_window <= fraction * chunk_window_size * len(features)
                            ranges = _true_ranges(valid)
                        ranges_by_pid[str(row.PID)] = ranges
                        if progress_bar:
                            scan.set_postfix(
                                eligible_chunks=sum(
                                    int(end) - int(start) + 1 for start, end in ranges
                                ),
                                refresh=False,
                            )
                _write_eligibility_cache(cache_file, identity, ranges_by_pid)
            fcntl.flock(lock.fileno(), fcntl.LOCK_UN)
    else:
        ranges_by_pid = cached
    candidate_starts = np.asarray(
        [max(0, int(length) - int(chunk_window_size) + 1) for length in frame.trace_length], dtype=np.int64
    )
    valid_starts = np.asarray([
        int((ranges_by_pid.get(str(pid), np.empty((0, 2), dtype=np.int32))[:, 1]
             - ranges_by_pid.get(str(pid), np.empty((0, 2), dtype=np.int32))[:, 0] + 1).sum())
        for pid in frame.PID
    ], dtype=np.int64)
    summary = {
        "cache_file": str(cache_file),
        "cache_hit": bool(cache_hit),
        "max_missing_fraction": fraction,
        "n_pids": int(len(frame)),
        "n_pids_with_eligible_chunks": int((valid_starts > 0).sum()),
        "n_candidate_chunk_starts": int(candidate_starts.sum()),
        "n_eligible_chunk_starts": int(valid_starts.sum()),
    }
    summary["eligible_chunk_start_fraction"] = (
        float(summary["n_eligible_chunk_starts"] / summary["n_candidate_chunk_starts"])
        if summary["n_candidate_chunk_starts"] else float("nan")
    )
    return ranges_by_pid, summary


class RandomChunkLoader:
    """Yield sampled training chunks or fixed multi-landmark validation chunks.

    No chunk array is precomputed. The trace file is
    opened only while iterating an epoch and each PID trace is read once for its
    selected chunk.
    """

    def __init__(
        self, trace_file, frame, *, features, chunk_window_size, batch_size,
        labtrans, target_kind, random_state=42, training=True, feature_stats=None,
        elapsed_time_feature=False, sample_rate_hz=None, elapsed_time_scale_seconds=43200,
        elapsed_time_channel=False, missingness_indicator_channels=False,
        missing_data_method="ffill",
        chunk_missingness_max_fraction=None, chunk_eligibility_cache_dir="cache/chunk_eligibility",
        history_window_count=1, validation_landmarks_per_patient=4,
        training_sampling_strategy="uniform", training_chunks_per_patient_per_epoch="auto",
        progress_bar=False,
    ):
        """Configure a lazy random-chunk batch iterable.

        Parameters
        ----------
        trace_file : str
            Variable-length tracing store.
        frame : pandas.DataFrame
            One eligible row per PID, including ``trace_length`` and targets.
        features : sequence[str]
            Channels read from each tracing.
        chunk_window_size, batch_size : int
            Chunk width in samples and number of PIDs per yielded batch.
        labtrans : LabTransDiscreteTime
            Maps remaining sample durations to DeepHit time-bin indices.
        target_kind : {"single", "marked", "competing"}
            Target tuple format yielded for the corresponding model mode.
        random_state : int, default=42
            Seed for PID order and random chunk starts.
        training : bool, default=True
            Randomly sample starts when true; use final chunks when false.
        feature_stats : list[dict] | None
            Optional training-set feature scaling statistics.
        elapsed_time_feature : bool, default=False
            Yield normalized elapsed chunk-end time as a second model input.
        elapsed_time_channel : bool, default=False
            Append elapsed seconds since tracing start as a synthetic waveform
            channel before scaling. This preserves the raw observation pattern:
            each sample in a chunk receives its own absolute elapsed time.
        sample_rate_hz : float | None
            Trace sampling rate used to convert chunk-end samples to seconds.
            Required when ``elapsed_time_feature`` is true.
        elapsed_time_scale_seconds : float, default=43200
            Fixed normalization scale for elapsed time. This must not depend on
            the eventual trace length, which would expose future information.
        missingness_indicator_channels : bool, default=False
            Append one binary observed-value channel per raw signal. Each mask
            is calculated before forward-fill imputation and is not scaled.
        missing_data_method : {"ffill", "zeros"}, default="ffill"
            Missing-data transformation applied after selecting the exact
            historical chunk.
        chunk_missingness_max_fraction : float | None, default=None
            When set, retain only chunk starts whose raw, pre-imputation total
            missing-value fraction across ``features`` is no greater than this
            value. Values above one are interpreted as percentages. One valid
            start is sampled uniformly per PID in each training epoch.
        chunk_eligibility_cache_dir : str, default="cache/chunk_eligibility"
            Directory holding compressed reusable valid-start sidecars.
        history_window_count : int, default=1
            Number of non-overlapping historical windows ending at the sampled
            endpoint. Earlier unavailable windows are left-padded with NaN.
        validation_landmarks_per_patient : int, default=4
            With ``training=False``, evaluate this many deterministic,
            evenly-spaced eligible endpoints per patient. This fixed panel
            approximates the random training endpoint distribution while
            remaining stable for early stopping. When
            ``training_sampling_strategy="terminal_balanced"``, the panel
            additionally includes deterministic endpoints at 0, 20, 40, and
            60 minutes before delivery, so the elapsed-time and terminal
            portions have equal weight in validation loss. With
            ``training_sampling_strategy="quartiles_plus_terminal"``, it uses
            four elapsed-time quartile anchors plus one fixed terminal sample
            per PID drawn reproducibly from the final 20 minutes.
        training_sampling_strategy : {"uniform", "stratified_elapsed", "stratified_elapsed_plus_terminal", "terminal_balanced", "quartiles_plus_terminal"}, default="uniform"
            Training-start sampling policy. ``"uniform"`` samples uniformly
            from all eligible starts, as in the original implementation.
            ``"stratified_elapsed"`` divides the observable interval from
            tracing start to its final usable start into the same
            start-to-end landmark strata used for validation. Each patient
            contributes one start per epoch and rotates through strata across
            epochs. This balances elapsed-time coverage without using time
            remaining until delivery to choose a chunk.
            ``"terminal_balanced"`` alternates those elapsed-time strata with
            four remaining-time strata centered on 0, 20, 40, and 60 minutes
            before delivery. It is intended for a retrospective terminal-
            landmark objective; delivery time is used only to select training
            chunks, never as a model input.
            ``"quartiles_plus_terminal"`` contributes five chunks per PID per
            epoch: one random eligible start from each elapsed-time quartile,
            plus one random eligible start ending within 20 minutes of
            delivery. Its validation panel uses the four elapsed anchors and
            one deterministic pseudo-random terminal start per PID from the
            same 20-minute interval.
            ``"stratified_elapsed_plus_terminal"`` uses the configurable
            elapsed-time strata plus the same final-20-minute terminal
            stratum.
        training_chunks_per_patient_per_epoch : {"auto", "all_strata"} or int, default="auto"
            Independent per-PID epoch budget. An integer samples that many
            distinct strata, rotating their assignment across epochs.
            ``"all_strata"`` samples every stratum once per epoch. Integers
            greater than the number of strata are rejected. ``"auto"``
            preserves the historical policy: all five quartile-plus-terminal
            strata, or one chunk for every other strategy. Uniform sampling
            has no finite strata and accepts any positive integer.
        progress_bar : bool, default=False
            Show HDF5, chunk-enumeration, and training-epoch progress bars.
        """
        self.trace_file = str(trace_file)
        self.frame = frame.reset_index(drop=True).copy()
        self.features = list(features)
        self.chunk_window_size = int(chunk_window_size)
        self.batch_size = int(batch_size)
        self.labtrans = labtrans
        self.target_kind = target_kind
        self.training = bool(training)
        self.feature_stats = feature_stats
        self.elapsed_time_feature = bool(elapsed_time_feature)
        self.elapsed_time_channel = bool(elapsed_time_channel)
        self.missingness_indicator_channels = bool(missingness_indicator_channels)
        self.missing_data_method = str(missing_data_method).strip().lower()
        if self.missing_data_method not in {"ffill", "zeros"}:
            raise ValueError("missing_data_method must be 'ffill' or 'zeros'")
        self.sample_rate_hz = None if sample_rate_hz is None else float(sample_rate_hz)
        self.elapsed_time_scale_seconds = float(elapsed_time_scale_seconds)
        self.chunk_missingness_max_fraction = _normalise_missingness_fraction(
            chunk_missingness_max_fraction
        )
        self.chunk_eligibility_cache_dir = str(chunk_eligibility_cache_dir)
        self.history_window_count = int(history_window_count)
        self.validation_landmarks_per_patient = int(validation_landmarks_per_patient)
        self.training_sampling_strategy = str(training_sampling_strategy).strip().lower()
        self.progress_bar = bool(progress_bar)
        if self.history_window_count < 1:
            raise ValueError("history_window_count must be positive")
        if self.validation_landmarks_per_patient < 1:
            raise ValueError("validation_landmarks_per_patient must be positive")
        if self.training_sampling_strategy not in {
            "uniform", "stratified_elapsed", "stratified_elapsed_plus_terminal",
            "terminal_balanced", "quartiles_plus_terminal"
        }:
            raise ValueError(
                "training_sampling_strategy must be 'uniform', 'stratified_elapsed', "
                "'stratified_elapsed_plus_terminal', 'terminal_balanced', or "
                "'quartiles_plus_terminal'"
            )
        if self.training_sampling_strategy in {
            "stratified_elapsed_plus_terminal", "terminal_balanced", "quartiles_plus_terminal"
        } and (
            self.sample_rate_hz is None or self.sample_rate_hz <= 0
        ):
            raise ValueError(
                "sample_rate_hz must be positive when training_sampling_strategy="
                "'stratified_elapsed_plus_terminal', 'terminal_balanced', or "
                "'quartiles_plus_terminal'"
            )
        self.training_chunks_per_patient_per_epoch = self._resolve_training_chunk_budget(
            training_chunks_per_patient_per_epoch
        )
        if (self.elapsed_time_feature or self.elapsed_time_channel) and (
            self.sample_rate_hz is None or self.sample_rate_hz <= 0
        ):
            raise ValueError(
                "sample_rate_hz must be positive when an elapsed-time feature is enabled"
            )
        if self.elapsed_time_scale_seconds <= 0:
            raise ValueError("elapsed_time_scale_seconds must be positive")
        self.random_state = int(random_state)
        self.rng = np.random.default_rng(self.random_state)
        self._training_epoch_index = 0
        # Torchtuples probes ``dataloader.dataset`` only to infer optional input
        # metadata before fitting.  The streaming loader has no indexable sample
        # dataset (indexing would defeat random-per-epoch chunk selection), so a
        # ``None`` probe deliberately tells it to skip that optional metadata.
        self.dataset = self
        if self.target_kind not in {"single", "marked", "competing"}:
            raise ValueError("target_kind must be single, marked, or competing")
        self.eligibility_summary = None
        self._eligible_ranges = None
        if self.chunk_missingness_max_fraction is not None:
            ranges_by_pid, summary = precompute_eligible_chunk_ranges(
                self.trace_file,
                self.frame,
                features=self.features,
                chunk_window_size=self.chunk_window_size,
                max_missing_fraction=self.chunk_missingness_max_fraction,
                cache_dir=self.chunk_eligibility_cache_dir,
                progress_bar=self.progress_bar,
            )
            ranges = [ranges_by_pid.get(str(pid), np.empty((0, 2), dtype=np.int32))
                      for pid in self.frame.PID]
            keep = np.asarray([len(item) > 0 for item in ranges], dtype=bool)
            self.frame = self.frame.loc[keep].reset_index(drop=True)
            self._eligible_ranges = [item for item, retain in zip(ranges, keep) if retain]
            self.eligibility_summary = {
                **summary,
                "n_pids_excluded_no_eligible_chunks": int((~keep).sum()),
                "n_pids_after_filter": int(keep.sum()),
            }
            if self.frame.empty:
                raise ValueError(
                    "No PIDs have a chunk passing chunk_missingness_max_fraction="
                    f"{self.chunk_missingness_max_fraction}"
                )
        self._validation_samples = None if self.training else self._build_validation_samples()

    def __len__(self):
        """Return the number of fixed training or validation batches in one epoch."""
        sample_count = (
            self.training_chunks_per_patient_per_epoch * len(self.frame)
            if self.training else len(self._validation_samples)
        )
        return ceil(sample_count / self.batch_size)

    def _sampling_strata_count(self):
        """Return the finite number of sampling strata, or ``None`` for uniform."""
        if self.training_sampling_strategy == "uniform":
            return None
        if self.training_sampling_strategy == "stratified_elapsed":
            return self.validation_landmarks_per_patient
        if self.training_sampling_strategy == "stratified_elapsed_plus_terminal":
            return self.validation_landmarks_per_patient + 1
        if self.training_sampling_strategy == "terminal_balanced":
            return self.validation_landmarks_per_patient + 4
        if self.training_sampling_strategy == "quartiles_plus_terminal":
            return 5
        raise RuntimeError(f"Unknown training sampling strategy: {self.training_sampling_strategy}")

    def _resolve_training_chunk_budget(self, value):
        """Validate and resolve a per-PID epoch chunk budget."""
        text = str(value).strip().lower()
        strata = self._sampling_strata_count()
        if text == "auto":
            return 5 if self.training_sampling_strategy == "quartiles_plus_terminal" else 1
        if text == "all_strata":
            if strata is None:
                raise ValueError(
                    "training_chunks_per_patient_per_epoch='all_strata' is not supported "
                    "for training_sampling_strategy='uniform'"
                )
            return strata
        try:
            budget = int(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "training_chunks_per_patient_per_epoch must be 'auto', 'all_strata', "
                "or a positive integer"
            ) from exc
        if budget < 1:
            raise ValueError("training_chunks_per_patient_per_epoch must be positive")
        if strata is not None and budget > strata:
            raise ValueError(
                "training_chunks_per_patient_per_epoch cannot exceed the number of "
                f"sampling strata ({strata}) for {self.training_sampling_strategy!r}"
            )
        return budget

    def __getitem__(self, index):
        """Return ``None`` for Torchtuples' metadata probe.

        The iterable deliberately has no indexable sample dataset because an
        indexed chunk would no longer be randomly selected per epoch.
        """
        return None

    def _trace_chunk(self, h5f, row, start):
        """Read, impute, scale, and slice one PID trace.

        Parameters
        ----------
        h5f : h5py.File
            Open tracing store handle for the current epoch.
        row : pandas.Series
            Metadata row identifying the PID and trace length.
        start : int
            Chunk start in samples.

        Returns
        -------
        numpy.ndarray
            Chunk shaped ``[chunk_window_size, n_features]`` or, when
            requested, physical channels followed by the elapsed-time channel
            and then binary observed-value masks.
        """
        trace = utils._read_hdf5_trace(h5f, row.PID, features=self.features)
        observed = np.isfinite(trace).astype(np.float32, copy=False)
        trace = utils._fill_missing_trace(trace, self.missing_data_method)
        if self.elapsed_time_channel:
            elapsed_channel = (
                np.arange(len(trace), dtype=np.float32) / self.sample_rate_hz
            ).reshape(-1, 1)
            trace = np.concatenate([trace, elapsed_channel], axis=1)
        if self.feature_stats is not None:
            trace = utils._apply_feature_stats(trace, self.feature_stats)
        if self.missingness_indicator_channels:
            trace = np.concatenate([trace, observed], axis=1)
        return trace[start:start + self.chunk_window_size]

    def _sample_uniform_eligible_start(self, position, max_start):
        """Choose one uniformly weighted valid start for a loader row."""
        if self._eligible_ranges is None:
            return int(self.rng.integers(max_start + 1)) if self.training else max_start
        ranges = self._eligible_ranges[position]
        lengths = ranges[:, 1].astype(np.int64) - ranges[:, 0].astype(np.int64) + 1
        if self.training:
            draw = int(self.rng.integers(int(lengths.sum())))
            interval = int(np.searchsorted(np.cumsum(lengths), draw, side="right"))
            previous = int(lengths[:interval].sum())
            return int(ranges[interval, 0] + draw - previous)
        # A deterministic validation chunk remains the latest usable chunk,
        # not necessarily the literal trace end when that window is too sparse.
        return int(ranges[-1, 1])

    def _landmark_targets(self, max_start):
        """Return the start-position anchors shared by validation and training strata."""
        if self.validation_landmarks_per_patient == 1:
            return np.asarray([int(max_start)], dtype=np.int64)
        return np.rint(np.linspace(
            0, max_start, self.validation_landmarks_per_patient
        )).astype(np.int64)

    def _stratified_elapsed_start(self, position, max_start, epoch_index):
        """Sample an eligible start from a rotating elapsed-time landmark stratum.

        Strata are bounded by midpoints between the deterministic validation
        landmark targets. Thus their locations are defined from elapsed time
        since tracing start only; ``max_start - start`` is never consulted to
        decide which stratum a patient receives.
        """
        targets = self._landmark_targets(max_start)
        if len(targets) == 1:
            return self._sample_uniform_eligible_start(position, max_start)

        # Rotate strata deterministically across patients and epochs. This
        # makes all strata equally represented in each epoch (to within one
        # PID) and makes each patient visit every stratum over K epochs.
        stratum = (int(position) + int(epoch_index)) % len(targets)
        return self._sample_elapsed_stratum(
            position, max_start, stratum=stratum, strata=len(targets), targets=targets,
        )

    def _sample_elapsed_stratum(self, position, max_start, *, stratum, strata=4, targets=None):
        """Sample one eligible start from a specified elapsed-time stratum."""
        targets = self._landmark_targets(max_start) if targets is None else targets
        if len(targets) != strata:
            targets = np.rint(np.linspace(0, max_start, strata)).astype(np.int64)
        boundaries = np.empty(len(targets) + 1, dtype=np.int64)
        boundaries[0] = 0
        boundaries[-1] = int(max_start) + 1
        boundaries[1:-1] = np.rint((targets[:-1] + targets[1:]) / 2.0).astype(np.int64)
        lower = int(boundaries[stratum])
        upper = int(boundaries[stratum + 1] - 1)
        if upper < lower:
            # Very short traces can yield repeated rounded landmarks. Falling
            # back to the target remains deterministic and eligibility-safe.
            return self._nearest_eligible_start(position, targets[stratum])

        if self._eligible_ranges is None:
            return int(self.rng.integers(lower, upper + 1))

        ranges = self._eligible_ranges[position]
        clipped_lower = np.maximum(ranges[:, 0], lower)
        clipped_upper = np.minimum(ranges[:, 1], upper)
        lengths = np.maximum(clipped_upper - clipped_lower + 1, 0).astype(np.int64)
        total = int(lengths.sum())
        if total == 0:
            # A missingness filter can leave a landmark stratum empty. Project
            # to the nearest eligible start rather than silently changing the
            # requested elapsed-time stratum for this patient.
            return self._nearest_eligible_start(position, targets[stratum])
        draw = int(self.rng.integers(total))
        interval = int(np.searchsorted(np.cumsum(lengths), draw, side="right"))
        previous = int(lengths[:interval].sum())
        return int(clipped_lower[interval] + draw - previous)

    def _sample_terminal_20_minute_start(self, position, max_start, *, rng):
        """Sample an eligible start whose chunk end is within 20 minutes of delivery."""
        window_samples = int(round(20.0 * 60.0 * self.sample_rate_hz))
        lower = max(0, int(max_start) - window_samples)
        upper = int(max_start)
        if self._eligible_ranges is None:
            return int(rng.integers(lower, upper + 1))
        ranges = self._eligible_ranges[position]
        clipped_lower = np.maximum(ranges[:, 0], lower)
        clipped_upper = np.minimum(ranges[:, 1], upper)
        lengths = np.maximum(clipped_upper - clipped_lower + 1, 0).astype(np.int64)
        total = int(lengths.sum())
        if total == 0:
            return self._nearest_eligible_start(position, upper)
        draw = int(rng.integers(total))
        interval = int(np.searchsorted(np.cumsum(lengths), draw, side="right"))
        previous = int(lengths[:interval].sum())
        return int(clipped_lower[interval] + draw - previous)

    def _deterministic_terminal_20_minute_start(self, position, max_start):
        """Choose one reproducible terminal start for a validation PID."""
        pid = str(self.frame.iloc[position].PID)
        digest = hashlib.sha256(f"{self.random_state}:{pid}".encode("utf-8")).digest()
        seed = int.from_bytes(digest[:8], byteorder="little", signed=False)
        return self._sample_terminal_20_minute_start(
            position, max_start, rng=np.random.default_rng(seed),
        )

    def _sample_terminal_balanced_start(self, position, max_start, epoch_index):
        """Sample a 50:50 mix of elapsed and terminal landmark strata.

        Four of the eight rotating strata are the usual elapsed-time strata.
        The other four cover 0--10, 10--30, 30--50, and 50--70 minutes before
        delivery, respectively.  The latter are centered on the 0/20/40/60
        minute end landmarks while retaining within-stratum randomness.
        """
        elapsed_strata = self.validation_landmarks_per_patient
        if elapsed_strata < 1:
            raise RuntimeError("validation_landmarks_per_patient must be positive")
        slot = (int(position) + int(epoch_index)) % (elapsed_strata + 4)
        if slot < elapsed_strata:
            # Offset the epoch so this call selects the requested elapsed
            # stratum under the existing deterministic rotation.
            return self._stratified_elapsed_start(
                position, max_start, epoch_index=epoch_index + slot - ((position + epoch_index) % elapsed_strata)
            )

        terminal_stratum = slot - elapsed_strata
        minute = 60.0 * self.sample_rate_hz
        lower_remaining = (0.0, 10.0, 30.0, 50.0)[terminal_stratum] * minute
        upper_remaining = (10.0, 30.0, 50.0, 70.0)[terminal_stratum] * minute
        # Remaining duration is ``max_start - start``.  Convert its inclusive
        # interval into start coordinates and clip it to the observable trace.
        lower = max(0, int(np.ceil(max_start - upper_remaining)))
        upper = min(max_start, int(np.floor(max_start - lower_remaining)))
        target = int(np.clip(
            round(max_start - (0.0, 20.0, 40.0, 60.0)[terminal_stratum] * minute),
            0, max_start,
        ))
        if upper < lower:
            return self._nearest_eligible_start(position, target)
        if self._eligible_ranges is None:
            return int(self.rng.integers(lower, upper + 1))

        ranges = self._eligible_ranges[position]
        clipped_lower = np.maximum(ranges[:, 0], lower)
        clipped_upper = np.minimum(ranges[:, 1], upper)
        lengths = np.maximum(clipped_upper - clipped_lower + 1, 0).astype(np.int64)
        total = int(lengths.sum())
        if total == 0:
            return self._nearest_eligible_start(position, target)
        draw = int(self.rng.integers(total))
        interval = int(np.searchsorted(np.cumsum(lengths), draw, side="right"))
        previous = int(lengths[:interval].sum())
        return int(clipped_lower[interval] + draw - previous)

    def _sample_start_from_stratum(self, position, max_start, stratum):
        """Sample one start from a named strategy's zero-based stratum."""
        if self.training_sampling_strategy == "stratified_elapsed":
            return self._sample_elapsed_stratum(
                position, max_start, stratum=stratum,
                strata=self.validation_landmarks_per_patient,
            )
        if self.training_sampling_strategy == "stratified_elapsed_plus_terminal":
            if stratum < self.validation_landmarks_per_patient:
                return self._sample_elapsed_stratum(
                    position, max_start, stratum=stratum,
                    strata=self.validation_landmarks_per_patient,
                )
            return self._sample_terminal_20_minute_start(position, max_start, rng=self.rng)
        if self.training_sampling_strategy == "terminal_balanced":
            if stratum < self.validation_landmarks_per_patient:
                return self._sample_elapsed_stratum(
                    position, max_start, stratum=stratum,
                    strata=self.validation_landmarks_per_patient,
                )
            return self._sample_terminal_balanced_start(
                position, max_start,
                epoch_index=stratum - int(position),
            )
        if self.training_sampling_strategy == "quartiles_plus_terminal":
            if stratum < 4:
                return self._sample_elapsed_stratum(
                    position, max_start, stratum=stratum, strata=4,
                )
            return self._sample_terminal_20_minute_start(position, max_start, rng=self.rng)
        raise RuntimeError(f"No finite strata for {self.training_sampling_strategy!r}")

    def _sample_start(self, position, max_start, epoch_index=0, draw_index=0):
        """Choose a training start under the configured policy and epoch budget."""
        strata = self._sampling_strata_count()
        if self.training and strata is not None:
            stratum = (int(position) + int(epoch_index) + int(draw_index)) % strata
            return self._sample_start_from_stratum(position, max_start, stratum)
        return self._sample_uniform_eligible_start(position, max_start)

    def _nearest_eligible_start(self, position, target):
        """Project a desired validation start to the closest eligible sample."""
        target = int(target)
        if self._eligible_ranges is None:
            return target
        ranges = self._eligible_ranges[position]
        candidates = np.clip(target, ranges[:, 0], ranges[:, 1])
        return int(candidates[np.argmin(np.abs(candidates - target))])

    def _build_validation_samples(self):
        """Create a reproducible, equally weighted endpoint panel per PID."""

        samples = []
        for position, row in self.frame.iterrows():
            max_start = int(row.trace_length) - self.chunk_window_size
            targets = (
                np.rint(np.linspace(0, max_start, 4)).astype(np.int64)
                if self.training_sampling_strategy == "quartiles_plus_terminal"
                else self._landmark_targets(max_start)
            )
            if self.training_sampling_strategy == "terminal_balanced":
                # Match terminal-balanced fitting with four deterministic
                # terminal centers. Preserve the elapsed endpoint at delivery
                # as a separate panel member: it represents a distinct
                # elapsed-time stratum during training, even though it shares
                # the zero-minute target with the terminal panel.
                terminal_minutes = np.asarray([0.0, 20.0, 40.0, 60.0])
                terminal_targets = np.rint(
                    max_start - terminal_minutes * 60.0 * self.sample_rate_hz
                ).astype(np.int64)
                targets = np.concatenate([
                    targets,
                    np.clip(terminal_targets, 0, max_start),
                ])
            for target in targets:
                samples.append((int(position), self._nearest_eligible_start(position, target)))
            if self.training_sampling_strategy in {
                "quartiles_plus_terminal", "stratified_elapsed_plus_terminal"
            }:
                samples.append((
                    int(position),
                    self._deterministic_terminal_20_minute_start(position, max_start),
                ))
        return samples

    def _trace_history(self, h5f, row, endpoint):
        """Return a left-padded, non-overlapping history ending at ``endpoint``.

        Every real window has exactly ``chunk_window_size`` samples.  Padding is
        NaN rather than zero so a history-aware encoder can distinguish absent
        early history from a zero-valued physiological signal.  When a raw
        missingness threshold is configured, sparse earlier windows are also
        left as unavailable padding.  The selected endpoint already satisfies
        that threshold; applying it to every history window prevents a native
        masking encoder from receiving a nearly empty context window.
        """
        width = self.chunk_window_size
        available = min(self.history_window_count, int(endpoint) // width)
        shape = (self.history_window_count, width, len(self.features)
                 + int(self.elapsed_time_channel)
                 + len(self.features) * int(self.missingness_indicator_channels))
        history = np.full(shape, np.nan, dtype=np.float32)
        first_start = int(endpoint) - available * width
        destination = self.history_window_count - available
        raw_trace = utils._read_hdf5_trace(h5f, row.PID, features=self.features)
        observed = np.isfinite(raw_trace)
        trace = utils._fill_missing_trace(raw_trace, self.missing_data_method)
        if self.elapsed_time_channel:
            elapsed_channel = (
                np.arange(len(trace), dtype=np.float32) / self.sample_rate_hz
            ).reshape(-1, 1)
            trace = np.concatenate([trace, elapsed_channel], axis=1)
        if self.feature_stats is not None:
            trace = utils._apply_feature_stats(trace, self.feature_stats)
        if self.missingness_indicator_channels:
            trace = np.concatenate([trace, observed.astype(np.float32, copy=False)], axis=1)
        for index in range(available):
            start = first_start + index * width
            end = start + width
            raw_window = observed[start:end]
            if self.chunk_missingness_max_fraction is not None:
                missing_count = int((~raw_window).sum())
                if missing_count > self.chunk_missingness_max_fraction * raw_window.size:
                    continue
            history[destination + index] = trace[start:end]
        return history

    def __iter__(self):
        """Yield fresh feature/target batches for one epoch.

        Yields
        ------
        torchtuples.TupleTree
            Features shaped ``[B, C, T]`` (or ``[B, W, C, T]`` for history)
            and a mode-specific DeepHit target tuple. The tracing store is
            opened only for the duration of this iteration.
        """
        import h5py

        if self.training:
            samples = [
                (int(position), draw_index)
                for position in range(len(self.frame))
                for draw_index in range(self.training_chunks_per_patient_per_epoch)
            ]
            epoch_index = self._training_epoch_index
            self._training_epoch_index += 1
        else:
            samples = self._validation_samples
            epoch_index = 0
        if self.training:
            self.rng.shuffle(samples)
        with h5py.File(self.trace_file, "r") as h5f:
            batch_iterator = range(0, len(samples), self.batch_size)
            if self.training and self.progress_bar:
                batch_iterator = tqdm(
                    batch_iterator,
                    total=len(self),
                    desc="[train] chunks",
                    unit="batch",
                    dynamic_ncols=True,
                )
            for batch_start in batch_iterator:
                batch_samples = samples[batch_start:batch_start + self.batch_size]
                if self.training and self.progress_bar:
                    batch_iterator.set_postfix(
                        effective_chunks=min(batch_start + len(batch_samples), len(samples)),
                        total_chunks=len(samples),
                        refresh=False,
                    )
                positions = [sample[0] for sample in batch_samples]
                rows = self.frame.iloc[positions]
                chunks = []
                durations, elapsed_times = [], []
                for (position, draw_index), (_, row) in zip(batch_samples, rows.iterrows()):
                    max_start = int(row.trace_length) - self.chunk_window_size
                    start = (
                        self._sample_start(
                            position, max_start, epoch_index=epoch_index, draw_index=draw_index,
                        )
                        if self.training else int(draw_index)
                    )
                    endpoint = start + self.chunk_window_size
                    if self.history_window_count > 1:
                        chunks.append(self._trace_history(h5f, row, endpoint))
                    else:
                        chunks.append(self._trace_chunk(h5f, row, start))
                    durations.append(max_start - start)
                    if self.elapsed_time_feature:
                        elapsed_times.append((start + self.chunk_window_size) / self.sample_rate_hz)
                stacked = np.stack(chunks)
                if self.history_window_count > 1:
                    X = np.ascontiguousarray(stacked.transpose(0, 1, 3, 2), dtype=np.float32)
                else:
                    X = np.ascontiguousarray(stacked.transpose(0, 2, 1), dtype=np.float32)
                duration_idx, delivery_event = self.labtrans.transform(
                    np.asarray(durations, dtype=float), np.ones(len(rows), dtype=np.int64)
                )
                # Pandas/HDF-backed arrays can be read-only views.  Torch warns
                # when converting those views, so own the small label arrays
                # explicitly before torchtuples turns them into tensors.
                duration_idx = np.array(duration_idx, dtype=np.int64, copy=True)
                if self.target_kind == "single":
                    event = np.array(rows.event, dtype=np.int64, copy=True)
                    rank = np.array(pair_rank_mat(duration_idx, event), copy=True)
                    target = tt.tuplefy(duration_idx, event, rank).to_tensor()
                elif self.target_kind == "marked":
                    marks = np.stack(rows.mark_targets.to_numpy()).astype(np.float32)
                    mask = np.array(rows.mark_observed, dtype=bool, copy=True)
                    rank = np.array(pair_rank_mat(duration_idx, delivery_event), copy=True)
                    target = tt.tuplefy(duration_idx, delivery_event, rank, marks, mask).to_tensor()
                else:
                    event_type = np.array(rows.event_type, dtype=np.int64, copy=True)
                    observed = np.array(rows.ph_observed, dtype=bool, copy=True)
                    target = tt.tuplefy(duration_idx, event_type, observed).to_tensor()
                trace_input = torch.from_numpy(X)
                if self.elapsed_time_feature:
                    elapsed = np.asarray(elapsed_times, dtype=np.float32).reshape(-1, 1)
                    elapsed /= self.elapsed_time_scale_seconds
                    input_ = tt.tuplefy(trace_input, torch.from_numpy(elapsed))
                else:
                    input_ = trace_input
                yield tt.tuplefy(input_, target)


def trace_metadata(trace_file):
    """Load normalized tracing metadata for lab joins.

    Parameters
    ----------
    trace_file : str
        Variable-length tracing store.

    Returns
    -------
    pandas.DataFrame
        Trace index with normalized ``PID``, trace length, and timing columns.
    """
    frame = utils.load_hdf5_index(str(trace_file))
    frame["PID"] = frame.PID.map(_normalise_pid)
    return frame


def lab_metadata(trace_file, label_file, *, chunk_window_size, lab_order_delay=30, horizon=0):
    """Join eligible traces to optional pH labels for streaming training.

    Parameters
    ----------
    trace_file : str
        Variable-length tracing store.
    label_file : str
        Lab table containing raw pH and threshold labels.
    chunk_window_size : int
        Minimum required trace length in samples.
    lab_order_delay : int, default=30
        Positive delay window in minutes; zero disables timing filtering.
    horizon : int, default=0
        Additional allowed lab-delay minutes.

    Returns
    -------
    tuple[pandas.DataFrame, numpy.ndarray]
        Eligible trace rows with optional mark targets and sorted pH thresholds.
        Traces without an eligible pH label remain present for delivery learning.
    """
    traces = trace_metadata(trace_file)
    labels = utils.load_label_table(label_file)
    labels["PID"] = labels.PID.map(_normalise_pid)
    pairs = discover_ph_threshold_columns(labels.columns)
    columns = ["PID", "pH Cord", *[column for _, column in pairs]]
    # CTU labels contain cord-gas outcomes but no lab-order timestamp.
    # In that case there is no timeliness filter to apply.
    if (
        "labs" in str(label_file)
        and lab_order_delay > 0
        and "LabOrderDtime" in labels.columns
    ):
        if "last_time" not in labels.columns:
            labels = labels.merge(traces[["PID", "last_time"]], on="PID", how="left")
        labels = utils.filter_labs_by_time_delay(labels, lab_order_delay, horizon)
    labels = labels[columns]
    if labels.PID.duplicated().any():
        raise ValueError("Streaming training requires at most one eligible lab row per PID.")
    frame = traces.merge(labels, on="PID", how="left")
    frame = frame[frame.trace_length >= int(chunk_window_size)].reset_index(drop=True)
    thresholds, _, targets, observed, _ = cumulative_ph_threshold_targets(frame)
    frame["mark_targets"] = list(targets)
    frame["mark_observed"] = observed
    # Competing-risk loss uses the same row-level pH observation indicator,
    # under a name that distinguishes it from marked-head supervision.
    frame["ph_observed"] = observed
    category = np.searchsorted(thresholds, frame["pH Cord"].to_numpy(float), side="right")
    frame["event_type"] = 0
    frame.loc[observed, "event_type"] = category[observed] + 1
    return frame, thresholds
