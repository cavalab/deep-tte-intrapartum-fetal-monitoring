"""Train landmark-specific XGBoost Cox baselines on rule-based CTG features."""
import json
from pathlib import Path
import fire
import numpy as np
import pandas as pd
import utils
from baseline_model_bundle import (
    BUNDLE_VERSION, evaluate_bundle, evaluation_time_grid, landmark_key,
    load_bundle, write_bundle,
)
from baseline_landmark_cache import load_baseline_landmark_panel
from ctg_feature_extraction import feature_names
from methods.marked_survival import discover_ph_threshold_columns
from time_to_event_evaluation import END_LANDMARK_SECONDS, START_LANDMARK_SECONDS, evaluate_time_to_event


def _xgb():
    try:
        from xgboost import XGBRegressor
    except ImportError as error:
        raise ImportError("Install the optional xgboost dependency to run this baseline.") from error
    return XGBRegressor


def _fit_cox(X, duration, event, *, random_state, n_estimators, max_depth, learning_rate):
    event = np.asarray(event, bool); duration = np.maximum(np.asarray(duration, float), 1.)
    if event.sum() == 0: return None
    model = _xgb()(objective="survival:cox", n_estimators=int(n_estimators), max_depth=int(max_depth),
        learning_rate=float(learning_rate), subsample=.8, colsample_bytree=.8, n_jobs=-1,
        random_state=int(random_state), eval_metric="cox-nloglik")
    # XGBoost's Cox objective encodes right-censored observations as negative labels.
    model.fit(X, np.where(event, duration, -duration))
    risk = np.maximum(model.predict(X), 1e-12)
    event_times = np.unique(duration[event])
    jumps = np.asarray([(duration[event] == time).sum() / risk[duration >= time].sum() for time in event_times])
    return model, event_times, np.cumsum(jumps)


def _risk_curve(fit, X, grid):
    if fit is None: return np.zeros((len(X), len(grid)))
    model, event_times, cumulative_hazard = fit
    # A zero-minute end landmark is represented by the first positive-time
    # Cox step (durations are floored at one second during fitting), preserving
    # the fitted ranking rather than returning an all-zero score.
    model_grid = np.maximum(np.asarray(grid, dtype=float), 1.0)
    baseline = np.interp(model_grid, event_times, cumulative_hazard,
                         left=0., right=cumulative_hazard[-1])
    return 1. - np.exp(-np.maximum(model.predict(X), 1e-12)[:, None] * baseline[None, :])


def run(trace_file, label_trainfile, label_testfile=None, savedir="results_xgboost_ctg",
        splits_file="data/train_val_test_splits_extended.csv", chunk_window_size=3600,
        lab_order_delay=30, bootstrap_samples=100, bootstrap_confidence_level=.95,
        random_state=42, landmark_cache_dir="cache/baseline_landmarks",
        evaluate_validation=False,
        n_estimators=300, max_depth=3, learning_rate=.05):
    """Fit separate landmark-specific delivery and cause-specific Cox models."""
    rate = utils.infer_sample_rate_hz(trace_file)
    if not np.isclose(rate, 1.): raise ValueError("This feature baseline is defined for a 1 Hz trace store.")
    pairs = discover_ph_threshold_columns(utils.load_label_table(label_trainfile).columns)
    thresholds = [pair[0] for pair in pairs]; names = [f"pH < {x:g}" for x in thresholds]
    folds = pd.read_csv(splits_file, dtype={"PID": str}).set_index("PID").fold
    split_pids = {fold: folds.index[folds == fold].to_numpy() for fold in folds.unique()}
    output = Path(savedir); output.mkdir(parents=True, exist_ok=True)
    records, prediction_rows, landmark_models = [], [], {}
    grid = evaluation_time_grid()
    def fit_models(train):
        Xtrain = train.X
        fits = {"delivery": _fit_cox(Xtrain, train.remaining_seconds, np.ones(len(train.PIDs), bool), random_state=random_state, n_estimators=n_estimators, max_depth=max_depth, learning_rate=learning_rate)}
        for index, name in enumerate(names):
            observed = train.ph_observed
            fits[name] = _fit_cox(Xtrain[observed], train.remaining_seconds[observed], train.ph_targets[observed, index] > 0, random_state=random_state + index + 1, n_estimators=n_estimators, max_depth=max_depth, learning_rate=learning_rate)
        return fits
    def score(fits, evaluation, start=None, end=None, split="Validation"):
        def prediction(chunks):
            # Keep exact 1/2/4-hour forecast cuts in sync with the shared
            # evaluator, while retaining the endpoint for concordance.
            return {"time_grid": grid, **{key: _risk_curve(fit, chunks, grid) for key, fit in fits.items()}}
        result = evaluate_time_to_event(train_data=evaluation, eval_data=evaluation, prediction=prediction,
            split_name=split, target_names=names, include_delivery=True,
            start_landmark_seconds=start, end_landmark_seconds=end,
            bootstrap_samples=int(bootstrap_samples),
            bootstrap_confidence_level=float(bootstrap_confidence_level),
            bootstrap_random_state=int(random_state))
        records.extend(result[0]); prediction_rows.extend(result[1])
    for landmark in START_LANDMARK_SECONDS:
        train = load_baseline_landmark_panel(trace_file, label_trainfile,
            landmark_seconds=landmark, from_end=False, chunk_window_size=int(chunk_window_size),
            lab_order_delay=lab_order_delay, sample_rate_hz=rate, feature_kind="xgboost",
            cache_dir=landmark_cache_dir, pids=split_pids["Train"])
        if not len(train.PIDs):
            continue
        fits = fit_models(train)
        landmark_models[landmark_key(landmark, from_end=False)] = fits
        if bool(evaluate_validation):
            val = load_baseline_landmark_panel(trace_file, label_trainfile,
                landmark_seconds=landmark, from_end=False, chunk_window_size=int(chunk_window_size),
                lab_order_delay=lab_order_delay, sample_rate_hz=rate, feature_kind="xgboost",
                cache_dir=landmark_cache_dir, pids=split_pids["Validation"])
            if len(val.PIDs): score(fits, val, start=landmark)
        if label_testfile:
            test = load_baseline_landmark_panel(trace_file, label_testfile,
                landmark_seconds=landmark, from_end=False, chunk_window_size=int(chunk_window_size),
                lab_order_delay=lab_order_delay, sample_rate_hz=rate, feature_kind="xgboost",
                cache_dir=landmark_cache_dir, pids=split_pids["Test"])
            if len(test.PIDs): score(fits, test, start=landmark, split="Test")
    # End landmarks use independently fitted models from same-offset training chunks.
    for landmark in END_LANDMARK_SECONDS:
        train = load_baseline_landmark_panel(trace_file, label_trainfile,
            landmark_seconds=landmark, from_end=True, chunk_window_size=int(chunk_window_size),
            lab_order_delay=lab_order_delay, sample_rate_hz=rate, feature_kind="xgboost",
            cache_dir=landmark_cache_dir, pids=split_pids["Train"])
        if not len(train.PIDs):
            continue
        fits = fit_models(train)
        landmark_models[landmark_key(landmark, from_end=True)] = fits
        if bool(evaluate_validation):
            val = load_baseline_landmark_panel(trace_file, label_trainfile,
                landmark_seconds=landmark, from_end=True, chunk_window_size=int(chunk_window_size),
                lab_order_delay=lab_order_delay, sample_rate_hz=rate, feature_kind="xgboost",
                cache_dir=landmark_cache_dir, pids=split_pids["Validation"])
            if len(val.PIDs): score(fits, val, end=landmark)
        if label_testfile:
            test = load_baseline_landmark_panel(trace_file, label_testfile,
                landmark_seconds=landmark, from_end=True, chunk_window_size=int(chunk_window_size),
                lab_order_delay=lab_order_delay, sample_rate_hz=rate, feature_kind="xgboost",
                cache_dir=landmark_cache_dir, pids=split_pids["Test"])
            if len(test.PIDs): score(fits, test, end=landmark, split="Test")
    pd.DataFrame(records).to_csv(output / "xgboost_ctg_landmark_metrics.csv", index=False)
    pd.DataFrame(prediction_rows).to_csv(output / "xgboost_ctg_landmark_predictions.csv", index=False)
    model_path = output / "xgboost_ctg_model.pkl"
    write_bundle(model_path, {
        "bundle_version": BUNDLE_VERSION,
        "baseline": "xgboost_ctg_cox",
        "model_file": str(model_path),
        "sample_rate_hz": float(rate),
        "chunk_window_size": int(chunk_window_size),
        "feature_kind": "xgboost",
        "require_acceleration": True,
        "target_names": names,
        "feature_names": feature_names(),
        "landmark_models": landmark_models,
        "hyperparameters": {
            "n_estimators": int(n_estimators), "max_depth": int(max_depth),
            "learning_rate": float(learning_rate), "random_state": int(random_state),
        },
    })
    with open(output / "xgboost_ctg_manifest.json", "w") as handle:
        json.dump({"model": "landmark-specific XGBoost survival:cox", "features": feature_names(),
            "assumptions": ["1 Hz traces", "tracing start treated as labor onset", "FIGO-rule event approximation", "no HF spectral feature above Nyquist"],
            "pH_model": "cause-specific Cox; non-acidemic observed deliveries censored",
            "sample_rate_hz": rate, "landmark_cache_dir": str(landmark_cache_dir),
            "model_file": str(model_path)}, handle, indent=2)


def evaluate(
    model_file, trace_file, test_label_file, savedir="results_xgboost_ctg_external",
    lab_order_delay=0, chunk_window_size=None, bootstrap_samples=100,
    bootstrap_confidence_level=.95, random_state=42,
    landmark_cache_dir="cache/baseline_landmarks_ctu",
):
    """Evaluate a saved FIGO-feature XGBoost bundle on a new cohort without refitting."""
    bundle = load_bundle(model_file, baseline="xgboost_ctg_cox")
    return evaluate_bundle(
        bundle=bundle, trace_file=trace_file, test_label_file=test_label_file,
        savedir=savedir, baseline="xgboost_ctg_cox", risk_curve=_risk_curve,
        file_stem="xgboost_ctg", lab_order_delay=lab_order_delay,
        chunk_window_size=chunk_window_size, bootstrap_samples=bootstrap_samples,
        bootstrap_confidence_level=bootstrap_confidence_level,
        random_state=random_state, landmark_cache_dir=landmark_cache_dir,
    )


if __name__ == "__main__": fire.Fire({"run": run, "evaluate": evaluate})
