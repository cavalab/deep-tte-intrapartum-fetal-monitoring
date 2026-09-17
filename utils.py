"""Small shared helpers for the HDF5 time-to-event workflow."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
import re
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm.auto import tqdm


def infer_sample_rate_hz(trace_file: str | Path) -> float:
    """Read the encoded sampling rate (for example, ``025`` = 0.25 Hz)."""
    match = re.search(r"sample_rate-(\d+)", str(trace_file))
    return int(match.group(1)) / 100 if match else 0.25


def _fill_missing_trace(trace: np.ndarray, method: str) -> np.ndarray:
    """Impute one ``(time, channel)`` trace without changing its shape."""
    trace = np.asarray(trace, dtype=np.float32).copy()
    method = str(method).lower()
    if method == "none":
        return trace
    if method == "ffill":
        for channel in range(trace.shape[1]):
            trace[:, channel] = pd.Series(trace[:, channel]).ffill().bfill()
        return trace
    if method == "zeros":
        trace[np.isnan(trace)] = 0
        return trace
    raise ValueError(f"Unsupported missing-data method: {method}")


def _apply_feature_stats(trace: np.ndarray, feature_stats: list[dict] | None) -> np.ndarray:
    """Scale finite values in a trace to [-1, 1] using training statistics."""
    if feature_stats is None:
        return trace
    scaled = np.asarray(trace, dtype=np.float32).copy()
    for channel, stats in enumerate(feature_stats):
        lower, upper = float(stats["min"]), float(stats["max"])
        if not np.isfinite(lower) or not np.isfinite(upper):
            continue
        if upper <= lower:
            scaled[:, channel] = 0
            continue
        valid = np.isfinite(scaled[:, channel])
        scaled[valid, channel] = 2 * (scaled[valid, channel] - lower) / (upper - lower) - 1
    return scaled


def _decode_hdf5_strings(values: np.ndarray) -> np.ndarray:
    """Decode HDF5 byte strings while preserving already-string values."""
    if values.dtype.kind not in {"S", "O"}:
        return values
    return np.asarray(
        [value.decode("utf-8") if isinstance(value, bytes) else str(value) for value in values],
        dtype=object,
    )


def _read_hdf5_trace(h5f, pid: str, features: tuple[str, ...] | list[str] = ("toco", "fecg")) -> np.ndarray:
    """Load requested signal channels for one patient from an open HDF5 file."""
    group = h5f["tracings"][str(pid)]
    return np.column_stack([group[feature][()].astype(np.float32) for feature in features])


def load_hdf5_index(trace_file: str | Path) -> pd.DataFrame:
    """Load the minimal PID, final-time, and length index from a tracing store."""
    import h5py

    with h5py.File(trace_file, "r") as h5f:
        index = h5f["index"]
        return pd.DataFrame(
            {
                "PID": _decode_hdf5_strings(index["pid"][()]),
                "last_time": pd.to_datetime(index["last_time_ns"][()], unit="ns"),
                "trace_length": index["length"][()],
            }
        )


def load_label_table(label_file: str | Path) -> pd.DataFrame:
    """Load a train/test label split from its parquet sidecar."""
    return pd.read_parquet(label_file)


def filter_labs_by_time_delay(frame: pd.DataFrame, lab_order_delay: float = 30, horizon: float = 0) -> pd.DataFrame:
    """Keep labels ordered after the trace and within the allowed delay window."""
    if "LabOrderDtime" not in frame:
        return frame
    delay = pd.to_datetime(frame["LabOrderDtime"]) - pd.to_datetime(frame["last_time"])
    maximum = pd.Timedelta(minutes=float(lab_order_delay) + float(horizon))
    return frame.loc[(delay > pd.Timedelta(0)) & (delay < maximum)].copy()


def compute_hdf5_feature_stats(
    trace_file: str | Path,
    pids,
    features: tuple[str, ...] | list[str] = ("toco", "fecg"),
    missing_data_method: str = "ffill",
    progress_bar: bool = False,
    cache_dir: str | Path | None = "cache/feature_statistics",
) -> list[dict[str, float]]:
    """Compute reusable min/max statistics on the training patients only."""
    import h5py

    trace_path = Path(trace_file)
    identity = {
        "trace_file": str(trace_path.resolve()),
        "size": trace_path.stat().st_size,
        "mtime_ns": trace_path.stat().st_mtime_ns,
        "pids": [str(pid) for pid in pids],
        "features": list(features),
        "missing_data_method": missing_data_method,
    }
    cache_path = None
    if cache_dir is not None:
        digest = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()[:16]
        cache_path = Path(cache_dir) / f"hdf5_feature_stats_{digest}.json"
        try:
            cached = json.loads(cache_path.read_text())
            if cached["identity"] == identity:
                return cached["feature_stats"]
        except (FileNotFoundError, KeyError, json.JSONDecodeError):
            pass

    minima = np.full(len(features), np.inf, dtype=np.float32)
    maxima = np.full(len(features), -np.inf, dtype=np.float32)
    iterator = tqdm(pids, desc="Computing feature statistics", disable=not progress_bar)
    with h5py.File(trace_path, "r") as h5f:
        for pid in iterator:
            trace = _fill_missing_trace(_read_hdf5_trace(h5f, pid, features), missing_data_method)
            for channel in range(len(features)):
                values = trace[np.isfinite(trace[:, channel]), channel]
                if values.size:
                    minima[channel] = min(minima[channel], values.min())
                    maxima[channel] = max(maxima[channel], values.max())

    stats = [
        {"min": float(lower if np.isfinite(lower) else 0), "max": float(upper if np.isfinite(upper) else 1)}
        for lower, upper in zip(minima, maxima)
    ]
    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        # Multiple Slurm jobs can compute this shared cache entry concurrently.
        # Give each writer a unique file in the target directory, then atomically
        # publish it so one job cannot remove another job's temporary file.
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=cache_path.parent,
            prefix=f".{cache_path.stem}.", suffix=".tmp", delete=False,
        ) as handle:
            json.dump({"identity": identity, "feature_stats": stats}, handle, indent=2)
            temporary = Path(handle.name)
        os.replace(temporary, cache_path)
    return stats


def smooth(values, window_len: int = 11, window: str = "hanning", series: bool = False):
    """Smooth a one-dimensional signal with a reflected moving window."""
    index = values.index if isinstance(values, pd.Series) else None
    values = np.asarray(values)
    if values.ndim != 1 or values.size < window_len:
        raise ValueError("smooth requires a one-dimensional array at least as long as its window")
    if window_len < 3:
        result = values
    else:
        windows = {"flat": np.ones, "hanning": np.hanning, "hamming": np.hamming,
                   "bartlett": np.bartlett, "blackman": np.blackman}
        if window not in windows:
            raise ValueError(f"Unsupported smoothing window: {window}")
        reflected = np.r_[values[window_len - 1:0:-1], values, values[-2:-window_len - 1:-1]]
        weights = windows[window](window_len)
        result = np.convolve(weights / weights.sum(), reflected, mode="valid")
        result = result[round(window_len / 2 - 1):-round(window_len / 2)]
    return pd.Series(result, index=index) if series else result
