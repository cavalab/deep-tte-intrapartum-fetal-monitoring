"""Evaluate CTG-only FRI through landmark-specific one-variable Cox models."""
import json
from pathlib import Path

import fire
import numpy as np
import pandas as pd
from sksurv.linear_model import CoxPHSurvivalAnalysis
from sksurv.util import Surv

import utils
from baseline_model_bundle import (
    BUNDLE_VERSION, evaluate_bundle, evaluation_time_grid, landmark_key,
    load_bundle, write_bundle,
)
from baseline_landmark_cache import load_baseline_landmark_panel
from methods.marked_survival import discover_ph_threshold_columns
from time_to_event_evaluation import (
    END_LANDMARK_SECONDS, START_LANDMARK_SECONDS, evaluate_time_to_event,
)


def _fit_cox(fri_risk, duration, event):
    """Fit a one-variable Cox model, returning ``None`` if it is unidentified."""
    fri_risk = np.asarray(fri_risk, dtype=float)
    duration = np.maximum(np.asarray(duration, dtype=float), 1.0)
    event = np.asarray(event, dtype=bool)
    mean = fri_risk.mean(axis=0)
    scale = fri_risk.std(axis=0)
    if event.sum() == 0 or np.any(scale == 0):
        return None
    standardized = (fri_risk - mean) / scale
    try:
        # The five-point FRI is discrete and can nearly separate small causes.
        # A modest ridge penalty avoids unstable/infinite Cox estimates.
        model = CoxPHSurvivalAnalysis(alpha=1.0, ties="breslow").fit(
            standardized, Surv.from_arrays(event, duration)
        )
        return model, mean, scale
    except (ArithmeticError, ValueError, np.linalg.LinAlgError):
        return None


def _risk_curve(fit, fri_risk, grid):
    """Predict delivery/cause-specific cumulative risks at exact horizons."""
    if fit is None:
        return np.zeros((len(fri_risk), len(grid)), dtype=float)
    model, mean, scale = fit
    survival = model.predict_survival_function((fri_risk - mean) / scale)
    # scikit-survival StepFunctions reject times beyond their final observed
    # event. A Cox survival curve is conventionally constant after that final
    # event, so clip only for StepFunction evaluation and retain the requested
    # evaluator grid shape.
    values = []
    for curve in survival:
        supported_grid = np.clip(grid, curve.domain[0], curve.domain[1])
        values.append(1.0 - curve(supported_grid))
    return np.clip(np.asarray(values, dtype=float), 0.0, 1.0)


def run(
    trace_file,
    label_trainfile,
    label_testfile=None,
    savedir="results_ctg_fri_cox",
    splits_file="data/train_val_test_splits_extended.csv",
    chunk_window_size=3600,
    lab_order_delay=30,
    bootstrap_samples=100,
    bootstrap_confidence_level=0.95,
    random_state=42,
    require_acceleration=True,
    landmark_cache_dir="cache/baseline_landmarks",
    evaluate_validation=False,
):
    """Fit one-variable FRI Cox models at all standard evaluation landmarks.

    A delivery Cox model uses all deliveries. Each pH-threshold model is a
    cause-specific Cox model fit only where pH is observed; non-acidemic
    deliveries are censored at delivery. Thus pH outputs are not competing-risk
    CIFs, but the Cox/Breslow curves are valid within that stated model.
    """
    rate = utils.infer_sample_rate_hz(trace_file)
    if not np.isclose(rate, 1.0):
        raise ValueError(
            f"CTG-only FRI baseline is specified for 1 Hz traces; got {rate:g} Hz."
        )
    pairs = discover_ph_threshold_columns(utils.load_label_table(label_trainfile).columns)
    if not pairs:
        raise ValueError("No pH threshold label columns found in label_trainfile.")
    target_names = [f"pH < {threshold:g}" for threshold, _ in pairs]
    folds = pd.read_csv(splits_file, dtype={"PID": str}).set_index("PID")["fold"]
    split_pids = {fold: folds.index[folds == fold].to_numpy() for fold in folds.unique()}
    output = Path(savedir)
    output.mkdir(parents=True, exist_ok=True)
    records, prediction_rows = [], []
    landmark_models = {}
    grid = evaluation_time_grid()

    def fit_models(train):
        train_fri = train.X
        fits = {"delivery": _fit_cox(
            train_fri, train.remaining_seconds, np.ones(len(train.PIDs), dtype=bool)
        )}
        for index, name in enumerate(target_names):
            observed = train.ph_observed
            fits[name] = _fit_cox(
                train_fri[observed], train.remaining_seconds[observed],
                train.ph_targets[observed, index] > 0,
            )
        return fits

    def score(fits, evaluation, *, start=None, end=None, split="Validation"):

        def prediction(chunks):
            # Include every exact forecast cut requested by the shared
            # evaluator (currently 1/2/4 h), the end-landmark offsets, and
            # the retained-trace endpoint required for full-curve concordance.
            return {"time_grid": grid, **{
                name: _risk_curve(fit, chunks, grid) for name, fit in fits.items()
            }}

        result = evaluate_time_to_event(
            train_data=evaluation, eval_data=evaluation, prediction=prediction,
            split_name=split, target_names=target_names, include_delivery=True,
            start_landmark_seconds=start, end_landmark_seconds=end,
            bootstrap_samples=int(bootstrap_samples),
            bootstrap_confidence_level=float(bootstrap_confidence_level),
            bootstrap_random_state=int(random_state),
        )
        records.extend(result[0])
        prediction_rows.extend(result[1])

    def panel(landmark, *, from_end=False):
        kwargs = {"landmark_seconds": landmark, "from_end": from_end}
        train = load_baseline_landmark_panel(
            trace_file, label_trainfile, chunk_window_size=int(chunk_window_size),
            lab_order_delay=float(lab_order_delay), sample_rate_hz=rate,
            feature_kind="fri", cache_dir=landmark_cache_dir,
            require_acceleration=bool(require_acceleration), pids=split_pids["Train"], **kwargs,
        )
        if not len(train.PIDs):
            return
        fits = fit_models(train)
        landmark_models[landmark_key(landmark, from_end=from_end)] = fits
        if bool(evaluate_validation):
            validation = load_baseline_landmark_panel(
                trace_file, label_trainfile, chunk_window_size=int(chunk_window_size),
                lab_order_delay=float(lab_order_delay), sample_rate_hz=rate,
                feature_kind="fri", cache_dir=landmark_cache_dir,
                require_acceleration=bool(require_acceleration), pids=split_pids["Validation"], **kwargs,
            )
            if len(train.PIDs) and len(validation.PIDs):
                score(fits, validation, start=None if from_end else landmark,
                      end=landmark if from_end else None)
        if label_testfile:
            test = load_baseline_landmark_panel(
                trace_file, label_testfile, chunk_window_size=int(chunk_window_size),
                lab_order_delay=float(lab_order_delay), sample_rate_hz=rate,
                feature_kind="fri", cache_dir=landmark_cache_dir,
                require_acceleration=bool(require_acceleration), pids=split_pids["Test"], **kwargs,
            )
            if len(train.PIDs) and len(test.PIDs):
                score(fits, test, split="Test", start=None if from_end else landmark,
                      end=landmark if from_end else None)

    for landmark in START_LANDMARK_SECONDS:
        panel(landmark)
    for landmark in END_LANDMARK_SECONDS:
        panel(landmark, from_end=True)

    pd.DataFrame(records).to_csv(output / "ctg_fri_cox_landmark_metrics.csv", index=False)
    pd.DataFrame(prediction_rows).to_csv(output / "ctg_fri_cox_landmark_predictions.csv", index=False)
    model_path = output / "ctg_fri_cox_model.pkl"
    write_bundle(model_path, {
        "bundle_version": BUNDLE_VERSION,
        "baseline": "ctg_fri_cox",
        "model_file": str(model_path),
        "sample_rate_hz": float(rate),
        "chunk_window_size": int(chunk_window_size),
        "feature_kind": "fri",
        "require_acceleration": bool(require_acceleration),
        "target_names": target_names,
        "landmark_models": landmark_models,
    })
    with open(output / "ctg_fri_cox_manifest.json", "w") as handle:
        json.dump({
            "baseline": "CTG-only FRI one-variable Cox",
            "components": ["FHR baseline", "FHR variability", "accelerations",
                           "decelerations", "uterine activity"],
            "clinical_fri_components": "not scored; unavailable by design",
            "cox_covariate": "100 - CTG-only FRI (higher = more abnormal)",
            "cox_covariate_standardization": "training-landmark mean/std",
            "cox_ridge_alpha": 1.0,
            "pH_model": "cause-specific Cox; non-acidemic observed deliveries censored",
            "sample_rate_hz": rate, "analysis_window_seconds": 1800,
            "landmark_cache_dir": str(landmark_cache_dir),
            "require_acceleration": bool(require_acceleration),
            "model_file": str(model_path),
            "figo_reference": "FIGO 2015 CTG interpretation guideline",
        }, handle, indent=2)


def evaluate(
    model_file, trace_file, test_label_file, savedir="results_ctg_fri_cox_external",
    lab_order_delay=0, chunk_window_size=None, bootstrap_samples=100,
    bootstrap_confidence_level=0.95, random_state=42,
    landmark_cache_dir="cache/baseline_landmarks_ctu",
):
    """Evaluate a saved CTG-only FRI bundle on a new cohort without refitting."""
    bundle = load_bundle(model_file, baseline="ctg_fri_cox")
    return evaluate_bundle(
        bundle=bundle, trace_file=trace_file, test_label_file=test_label_file,
        savedir=savedir, baseline="ctg_fri_cox", risk_curve=_risk_curve,
        file_stem="ctg_fri_cox", lab_order_delay=lab_order_delay,
        chunk_window_size=chunk_window_size, bootstrap_samples=bootstrap_samples,
        bootstrap_confidence_level=bootstrap_confidence_level,
        random_state=random_state, landmark_cache_dir=landmark_cache_dir,
    )


if __name__ == "__main__":
    fire.Fire({"run": run, "evaluate": evaluate})
