"""Portable landmark-baseline bundles and evaluation-only replay helpers."""
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

import utils
from baseline_landmark_cache import load_baseline_landmark_panel
from methods.marked_survival import discover_ph_threshold_columns
from time_to_event_evaluation import (
    END_LANDMARK_SECONDS, EVALUATION_HORIZON_SECONDS, MAX_TRACING_SECONDS,
    START_LANDMARK_SECONDS, evaluate_time_to_event,
)


BUNDLE_VERSION = 1


def evaluation_time_grid():
    """Exact end offsets, 1/2/4-hour horizons, and full-trace endpoint."""
    return np.unique(np.concatenate((
        END_LANDMARK_SECONDS,
        EVALUATION_HORIZON_SECONDS,
        [MAX_TRACING_SECONDS],
    )))


def landmark_key(landmark_seconds, *, from_end):
    prefix = "end" if from_end else "start"
    return f"{prefix}_{int(landmark_seconds)}"


def write_bundle(path, bundle):
    """Persist a complete fitted landmark-model collection as a pickle."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        pickle.dump(bundle, handle, protocol=pickle.HIGHEST_PROTOCOL)


def load_bundle(path, *, baseline):
    """Load and validate one portable baseline bundle."""
    with Path(path).open("rb") as handle:
        bundle = pickle.load(handle)
    if not isinstance(bundle, dict) or bundle.get("bundle_version") != BUNDLE_VERSION:
        raise ValueError("Unsupported or malformed baseline model bundle.")
    if bundle.get("baseline") != baseline:
        raise ValueError(
            f"Bundle baseline is {bundle.get('baseline')!r}, expected {baseline!r}."
        )
    if not isinstance(bundle.get("landmark_models"), dict):
        raise ValueError("Baseline bundle has no landmark_models mapping.")
    return bundle


def evaluate_bundle(
    *,
    bundle, trace_file, test_label_file, savedir, baseline, risk_curve,
    file_stem, lab_order_delay=0, chunk_window_size=None,
    bootstrap_samples=100, bootstrap_confidence_level=0.95,
    random_state=42, landmark_cache_dir="cache/baseline_landmarks",
):
    """Evaluate fitted landmark-specific models on a new labelled cohort.

    The bundle is never refit: its stored Cox/XGBoost models and calibration
    state are applied to features extracted from ``trace_file`` only.
    """
    rate = utils.infer_sample_rate_hz(trace_file)
    if not np.isclose(rate, float(bundle["sample_rate_hz"])):
        raise ValueError(
            f"Bundle requires {bundle['sample_rate_hz']:g} Hz traces; got {rate:g} Hz."
        )
    configured_window = int(bundle["chunk_window_size"])
    if chunk_window_size is not None and int(chunk_window_size) != configured_window:
        raise ValueError(
            f"Bundle requires chunk_window_size={configured_window}; got {chunk_window_size}."
        )
    target_names = list(bundle["target_names"])
    observed_names = [f"pH < {threshold:g}" for threshold, _ in discover_ph_threshold_columns(
        utils.load_label_table(test_label_file).columns
    )]
    if observed_names != target_names:
        raise ValueError(
            f"Label thresholds {observed_names} do not match bundle targets {target_names}."
        )

    output = Path(savedir)
    output.mkdir(parents=True, exist_ok=True)
    records, prediction_rows = [], []
    feature_kind = bundle["feature_kind"]
    require_acceleration = bool(bundle.get("require_acceleration", True))
    grid = evaluation_time_grid()

    def score(panel, *, landmark, from_end):
        key = landmark_key(landmark, from_end=from_end)
        fits = bundle["landmark_models"].get(key)
        if fits is None:
            raise ValueError(f"Bundle is missing fitted models for {key}.")

        def prediction(X):
            return {"time_grid": grid, **{
                name: risk_curve(fit, X, grid) for name, fit in fits.items()
            }}

        result = evaluate_time_to_event(
            train_data=panel, eval_data=panel, prediction=prediction,
            # The supplied labels are the held-out external cohort. Keep the
            # standard Test label so selection-table filtering is uniform
            # across baseline, classifier, and DeepHit artifacts.
            split_name="Test", target_names=target_names,
            include_delivery=True,
            start_landmark_seconds=None if from_end else landmark,
            end_landmark_seconds=landmark if from_end else None,
            bootstrap_samples=int(bootstrap_samples),
            bootstrap_confidence_level=float(bootstrap_confidence_level),
            bootstrap_random_state=int(random_state),
        )
        records.extend(result[0])
        prediction_rows.extend(result[1])

    for landmark in START_LANDMARK_SECONDS:
        panel = load_baseline_landmark_panel(
            trace_file, test_label_file, landmark_seconds=landmark, from_end=False,
            chunk_window_size=configured_window, lab_order_delay=float(lab_order_delay),
            sample_rate_hz=rate, feature_kind=feature_kind,
            cache_dir=landmark_cache_dir, require_acceleration=require_acceleration,
        )
        if len(panel.PIDs):
            score(panel, landmark=landmark, from_end=False)
    for landmark in END_LANDMARK_SECONDS:
        panel = load_baseline_landmark_panel(
            trace_file, test_label_file, landmark_seconds=landmark, from_end=True,
            chunk_window_size=configured_window, lab_order_delay=float(lab_order_delay),
            sample_rate_hz=rate, feature_kind=feature_kind,
            cache_dir=landmark_cache_dir, require_acceleration=require_acceleration,
        )
        if len(panel.PIDs):
            score(panel, landmark=landmark, from_end=True)

    metrics_path = output / f"{file_stem}_external_landmark_metrics.csv"
    predictions_path = output / f"{file_stem}_external_landmark_predictions.csv"
    pd.DataFrame(records).to_csv(metrics_path, index=False)
    pd.DataFrame(prediction_rows).to_csv(predictions_path, index=False)
    manifest_path = output / f"{file_stem}_external_manifest.json"
    manifest_path.write_text(json.dumps({
        "baseline": baseline,
        "evaluation": "external_validation_no_refit",
        "model_file": str(bundle.get("model_file", "")),
        "trace_file": str(trace_file),
        "test_label_file": str(test_label_file),
        "sample_rate_hz": float(rate),
        "chunk_window_size": configured_window,
        "lab_order_delay": float(lab_order_delay),
        "landmark_cache_dir": str(landmark_cache_dir),
        "landmark_metrics": str(metrics_path),
        "landmark_predictions": str(predictions_path),
    }, indent=2) + "\n")
    return str(metrics_path)
