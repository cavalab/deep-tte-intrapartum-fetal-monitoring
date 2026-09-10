"""Disk cache for compact baseline features at fixed TTE landmarks.

Caching raw one-hour waveform panels would be several gigabytes per landmark.
This module instead caches only PID-aligned outcome metadata and the extracted
FRI or XGBoost feature matrix.  It lets a restarted baseline run skip both the
full HDF5 scan and feature extraction while retaining the standard evaluator.
"""
import hashlib
import json
import os
from pathlib import Path

import numpy as np

from ctg_feature_extraction import extract_ctg_feature_matrix
from ctg_fri import ctg_only_fri_risk
from time_to_event_evaluation import LandmarkData, load_landmark_data


def _file_identity(path):
    path = Path(path).resolve()
    stat = path.stat()
    return {"path": str(path), "size": stat.st_size, "mtime_ns": stat.st_mtime_ns}


def _cache_identity(*, trace_file, label_file, landmark_seconds, from_end,
                    chunk_window_size, lab_order_delay, feature_kind,
                    require_acceleration, pids):
    pids = sorted(str(pid) for pid in pids) if pids is not None else None
    return {
        "version": 1,
        "trace": _file_identity(trace_file),
        "labels": _file_identity(label_file),
        "landmark_seconds": float(landmark_seconds),
        "from_end": bool(from_end),
        "chunk_window_size": int(chunk_window_size),
        "lab_order_delay": float(lab_order_delay),
        "feature_kind": str(feature_kind),
        "require_acceleration": bool(require_acceleration),
        "pids_digest": (
            hashlib.sha256("\n".join(pids).encode()).hexdigest() if pids is not None else None
        ),
        "n_pids_requested": len(pids) if pids is not None else None,
        "features": ["fecg", "toco"],
        "missing": "ffill",
    }


def _cache_path(cache_dir, identity):
    digest = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()[:20]
    location = "end" if identity["from_end"] else "start"
    return Path(cache_dir) / (
        f"baseline_{identity['feature_kind']}_{location}_"
        f"{int(identity['landmark_seconds'])}_{digest}.npz"
    )


def _decode(path, identity):
    if not path.exists():
        return None
    try:
        with np.load(path, allow_pickle=False) as values:
            cached_identity = json.loads(str(values["identity"][0]))
            if cached_identity != identity:
                return None
            return LandmarkData(
                X=values["X"], PIDs=values["PIDs"].astype(str),
                remaining_seconds=values["remaining_seconds"], elapsed_seconds=values["elapsed_seconds"],
                ph_targets=values["ph_targets"], ph_observed=values["ph_observed"].astype(bool),
                thresholds=values["thresholds"],
            )
    except (OSError, ValueError, KeyError, json.JSONDecodeError):
        return None


def _write(path, identity, panel):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.stem}.{os.getpid()}.tmp.npz")
    np.savez(
        temporary, identity=np.asarray([json.dumps(identity, sort_keys=True)]),
        X=np.asarray(panel.X, dtype=np.float32), PIDs=np.asarray(panel.PIDs, dtype=str),
        remaining_seconds=np.asarray(panel.remaining_seconds, dtype=np.float64),
        elapsed_seconds=np.asarray(panel.elapsed_seconds, dtype=np.float64),
        ph_targets=np.asarray(panel.ph_targets, dtype=np.float32),
        ph_observed=np.asarray(panel.ph_observed, dtype=bool),
        thresholds=np.asarray(panel.thresholds, dtype=np.float64),
    )
    os.replace(temporary, path)


def load_baseline_landmark_panel(
    trace_file, label_file, *, landmark_seconds, from_end, chunk_window_size,
    lab_order_delay, sample_rate_hz, feature_kind, cache_dir="cache/baseline_landmarks",
    require_acceleration=True, pids=None,
):
    """Load one cached baseline feature panel or build it from HDF5.

    ``feature_kind`` is ``"fri"`` for a single adverse-FRI covariate or
    ``"xgboost"`` for the FIGO-rule tabular feature matrix. Cache identity
    includes source file modification metadata and all extraction settings.
    """
    if feature_kind not in {"fri", "xgboost"}:
        raise ValueError("feature_kind must be 'fri' or 'xgboost'")
    identity = _cache_identity(
        trace_file=trace_file, label_file=label_file,
        landmark_seconds=landmark_seconds, from_end=from_end,
        chunk_window_size=chunk_window_size, lab_order_delay=lab_order_delay,
        feature_kind=feature_kind, require_acceleration=require_acceleration, pids=pids,
    )
    path = _cache_path(cache_dir, identity)
    panel = _decode(path, identity)
    if panel is not None:
        print(f"[baseline-cache] hit {path} ({len(panel.PIDs):,} rows)", flush=True)
        return panel

    print(f"[baseline-cache] miss {path}; extracting landmark panel", flush=True)
    raw = load_landmark_data(
        trace_file, label_file, features=["fecg", "toco"],
        landmark_seconds=landmark_seconds, from_end=from_end,
        chunk_window_size=int(chunk_window_size), lab_order_delay=float(lab_order_delay),
        missing="ffill", pids=pids,
    )
    if feature_kind == "fri":
        X = ctg_only_fri_risk(
            raw.X, sample_rate_hz=sample_rate_hz, fhr_index=0, toco_index=1,
            require_acceleration=require_acceleration,
        )[:, None]
    else:
        X = extract_ctg_feature_matrix(raw.X, sample_rate_hz=sample_rate_hz)
    panel = LandmarkData(
        X=X, PIDs=raw.PIDs, remaining_seconds=raw.remaining_seconds,
        elapsed_seconds=raw.elapsed_seconds, ph_targets=raw.ph_targets,
        ph_observed=raw.ph_observed, thresholds=raw.thresholds,
    )
    # Release the several-GB waveform panel before persisting compact features.
    del raw
    _write(path, identity, panel)
    print(f"[baseline-cache] wrote {path} ({len(panel.PIDs):,} rows)", flush=True)
    return panel
