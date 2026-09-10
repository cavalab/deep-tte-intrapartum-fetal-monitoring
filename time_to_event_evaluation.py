"""Exact landmark evaluation for chunked time-to-event models."""
from dataclasses import dataclass
from contextlib import nullcontext
import time
import warnings
import zlib

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.exceptions import UndefinedMetricWarning

# scikit-survival releases that still call ``np.trapz`` are otherwise
# compatible with NumPy 2, which removed that legacy alias in favor of
# ``np.trapezoid``.  Restore the alias before importing/using its metrics so
# dynamic AUC evaluation remains portable across the cluster environments.
if not hasattr(np, "trapz"):
    np.trapz = np.trapezoid

from sksurv.util import Surv
from pycox.evaluation import EvalSurv

import utils
from methods.marked_survival import (
    _normalise_pid,
    cumulative_ph_threshold_targets,
    discover_ph_threshold_columns,
)


# A deliberately sparse landmark/horizon grid keeps complete held-out
# evaluation feasible within a training job's wall-clock allocation. The
# tracing store retains at most the final 48 hours, so a forecast must end no
# later than that observed-tracing limit.
MAX_TRACING_SECONDS = 48 * 60 * 60
START_LANDMARK_SECONDS = np.asarray([1, 6, 12, 20], dtype=float) * 60 * 60
EVALUATION_HORIZON_SECONDS = np.asarray([1, 2, 4], dtype=float) * 60 * 60
END_LANDMARK_SECONDS = np.asarray([0, 20 * 60, 40 * 60, 60 * 60], dtype=float)


def _valid_forecast_horizons(landmark_seconds):
    """Return ``(score, outcome)`` horizons that remain within the trace cap."""
    landmark = float(landmark_seconds)
    regular = EVALUATION_HORIZON_SECONDS[
        landmark + EVALUATION_HORIZON_SECONDS <= MAX_TRACING_SECONDS
    ]
    return [(float(horizon), float(horizon)) for horizon in regular]


@dataclass
class LandmarkData:
    X: np.ndarray
    PIDs: np.ndarray
    remaining_seconds: np.ndarray
    elapsed_seconds: np.ndarray
    ph_targets: np.ndarray
    ph_observed: np.ndarray
    thresholds: np.ndarray


def _eligible_labels(traces, label_file, label_columns, lab_order_delay, horizon):
    """Return indexed lab labels eligible under the configured timeliness rule."""
    labels = utils.load_label_table(label_file)
    labels["PID"] = labels["PID"].map(_normalise_pid)
    required = ["PID", *label_columns]
    absent = set(required).difference(labels.columns)
    if absent:
        raise ValueError(f"Label file is missing columns: {sorted(absent)}")
    # CTU labels have no lab-order timestamp, so skip timeliness filtering
    # when that column is absent.
    if (
        "labs" in str(label_file)
        and lab_order_delay > 0
        and "LabOrderDtime" in labels.columns
    ):
        if "last_time" not in labels.columns:
            if "last_time" not in traces.columns:
                raise ValueError("Lab-delay filtering requires 'last_time' in labels or tracings.")
            labels = labels.merge(traces[["PID", "last_time"]], on="PID", how="left")
        labels = utils.filter_labs_by_time_delay(
            labels.copy(), lab_order_delay=lab_order_delay, horizon=horizon
        )
    labels = labels[required]
    if labels.PID.duplicated().any():
        raise ValueError("Landmark evaluation requires at most one eligible lab row per PID.")
    return labels.set_index("PID")


def load_landmark_data(
    trace_file,
    label_file,
    *,
    features,
    landmark_seconds,
    from_end=False,
    missing="ffill",
    lab_order_delay=30,
    horizon=0,
    chunk_window_size=900,
    require_label=False,
    missingness_indicator_channels=False,
    chunk_missingness_max_fraction=None,
    history_window_count=1,
    progress_label=None,
    pids=None,
):
    """Create exact-endpoint chunks for a start or end landmark.

    ``from_end=False`` uses a common elapsed-time endpoint.  ``from_end=True``
    uses an endpoint exactly ``landmark_seconds`` before the trace end.

    Parameters
    ----------
    trace_file, label_file : str
        Tracing store and lab table to join.
    features : sequence[str]
        Signal channels included in each chunk.
    landmark_seconds : float
        Elapsed endpoint (or offset before trace end) in seconds.
    from_end : bool, default=False
        Interpret ``landmark_seconds`` as an endpoint offset before trace end.
    missing : {"ffill", "zeros", "drop"}, default="ffill"
        Signal missingness policy applied before chunking.
    lab_order_delay, horizon : int
        Lab timeliness settings in minutes.
    chunk_window_size : int
        Fixed chunk width in samples.
    history_window_count : int, default=1
        Number of non-overlapping windows ending at each landmark. Earlier
        unavailable windows are left-padded with NaN.
    require_label : bool, default=False
        Exclude traces without an eligible pH label when true.
    pids : sequence[str] | None, default=None
        Optional PID subset to scan before waveform extraction.
    missingness_indicator_channels : bool, default=False
        Append one unscaled binary observed-value channel per signal, measured
        before applying ``missing`` imputation.
    chunk_missingness_max_fraction : float | None, default=None
        Keep an exact landmark only when its raw, pre-imputation input window
        has no more than this total missing-value fraction across ``features``.
        Values above one are treated as percentages.

    Returns
    -------
    LandmarkData
        Chunks, PIDs, remaining/elapsed seconds, and pH targets aligned by row.
    """
    trace_file = str(trace_file)
    history_window_count = int(history_window_count)
    if history_window_count < 1:
        raise ValueError("history_window_count must be positive")
    if chunk_missingness_max_fraction is not None:
        chunk_missingness_max_fraction = float(chunk_missingness_max_fraction)
        if chunk_missingness_max_fraction > 1:
            chunk_missingness_max_fraction /= 100.0
        if not 0 <= chunk_missingness_max_fraction <= 1:
            raise ValueError("chunk_missingness_max_fraction must be between 0 and 1 (or 0 and 100)")
    is_hdf5 = trace_file.endswith((".h5", ".hdf5"))
    if is_hdf5:
        traces = utils.load_hdf5_index(trace_file)
        traces["PID"] = traces["PID"].map(_normalise_pid)
    else:
        raise ValueError("Only HDF5 trace stores are supported.")
    if pids is not None:
        requested_pids = {_normalise_pid(pid) for pid in pids}
        traces = traces.loc[traces.PID.isin(requested_pids)].copy()
    labels_frame = utils.load_label_table(label_file)
    pairs = discover_ph_threshold_columns(labels_frame.columns)
    threshold_columns = [column for _, column in pairs]
    if "pH Cord" not in labels_frame.columns:
        raise ValueError("Landmark evaluation requires raw 'pH Cord'.")
    labels = _eligible_labels(
        traces, label_file, ["pH Cord", *threshold_columns], lab_order_delay, horizon
    )
    rate = utils.infer_sample_rate_hz(trace_file)
    endpoint_target = int(round(float(landmark_seconds) * rate))
    chunks, rows, pids, remaining, elapsed = [], [], [], [], []
    landmark_label = progress_label or (
        f"{float(landmark_seconds) / 3600:g}h {'before end' if from_end else 'since start'}"
    )
    started = time.monotonic()
    print(
        f"[evaluation] {landmark_label}: extracting exact chunks from {len(traces):,} traces",
        flush=True,
    )
    h5_context = nullcontext(None)
    if is_hdf5:
        import h5py
        h5_context = h5py.File(trace_file, "r")
    with h5_context as h5f:
      for trace_index, (_, trace) in enumerate(traces.iterrows(), start=1):
        if trace_index % 5000 == 0:
            print(
                f"[evaluation] {landmark_label}: scanned {trace_index:,}/{len(traces):,} "
                f"traces; retained {len(chunks):,} chunks; elapsed {time.monotonic() - started:.0f}s",
                flush=True,
            )
        pid = _normalise_pid(trace.PID)
        if require_label and pid not in labels.index:
            continue
        arrays, observed_arrays = [], []
        hdf_trace = utils._read_hdf5_trace(h5f, pid, features=features) if h5f else None
        for feature_index, feature in enumerate(features):
            values = (
                np.asarray(hdf_trace[:, feature_index], dtype=float).ravel()
                if hdf_trace is not None
                else np.asarray(trace[feature], dtype=float).ravel()
            )
            observed = np.isfinite(values).astype(np.float32, copy=False)
            if missing == "ffill":
                values = pd.Series(values).ffill().bfill().to_numpy()
            elif missing == "zeros":
                values = np.nan_to_num(values, nan=0.0)
            elif missing == "drop":
                values = pd.Series(values).dropna().to_numpy()
                observed = np.ones(len(values), dtype=np.float32)
            else:
                raise ValueError("missing must be ffill, zeros, or drop")
            arrays.append(values)
            observed_arrays.append(observed)
        if missingness_indicator_channels:
            arrays.extend(observed_arrays)
        total = max((len(values) for values in arrays), default=0)
        endpoint = total - endpoint_target if from_end else endpoint_target
        if endpoint < chunk_window_size or endpoint > total:
            continue
        if pid in labels.index:
            row = labels.loc[pid].to_dict()
        else:
            row = {column: np.nan for column in ["pH Cord", *threshold_columns]}
        sample = np.full(
            (history_window_count, chunk_window_size, len(arrays)), np.nan, dtype=float
        )
        start = endpoint - chunk_window_size
        if chunk_missingness_max_fraction is not None:
            raw_window = np.column_stack(observed_arrays)[start:endpoint]
            missing_fraction = 1.0 - float(raw_window.mean())
            if missing_fraction > chunk_missingness_max_fraction:
                continue
        available = min(history_window_count, endpoint // chunk_window_size)
        first_start = endpoint - available * chunk_window_size
        destination = history_window_count - available
        for window in range(available):
            window_start = first_start + window * chunk_window_size
            window_end = window_start + chunk_window_size
            for column, values in enumerate(arrays):
                sample[destination + window, :, column] = values[window_start:window_end]
        chunks.append(sample)
        rows.append(row)
        pids.append(pid)
        remaining.append((total - endpoint) / rate)
        elapsed.append(endpoint / rate)
    if not chunks:
        return LandmarkData(
            np.empty((0, history_window_count, chunk_window_size,
                      len(features) * (1 + int(missingness_indicator_channels)))),
            np.asarray([]), np.asarray([]),
            np.asarray([]), np.empty((0, len(pairs))), np.asarray([], dtype=bool),
            np.asarray([threshold for threshold, _ in pairs]),
        )
    chunks = np.stack(chunks)
    if history_window_count == 1:
        chunks = chunks[:, 0]
    frame = pd.DataFrame(rows)
    thresholds, _, targets, observed, _ = cumulative_ph_threshold_targets(frame)
    return LandmarkData(
        chunks, np.asarray(pids), np.asarray(remaining), np.asarray(elapsed),
        targets, observed, thresholds,
    )


def _value_at(curves, time_grid_seconds, horizon_seconds):
    """Select curve values at one exact requested horizon in seconds."""
    grid = np.asarray(time_grid_seconds, dtype=float)
    matches = np.flatnonzero(np.isclose(grid, horizon_seconds))
    if matches.size != 1:
        raise ValueError(f"Model output lacks exact {horizon_seconds:g}-second evaluation cut.")
    return np.asarray(curves)[:, int(matches[0])]


def _predict_landmark_data(prediction, data):
    """Run a prediction callback, passing elapsed time only when requested.

    Prediction callbacks receive elapsed endpoint seconds only when they
    declare the ``uses_elapsed_time_feature`` attribute.
    """
    if getattr(prediction, "uses_elapsed_time_feature", False):
        return prediction(data.X, data.elapsed_seconds)
    return prediction(data.X)


def _safe_metric(function):
    """Evaluate a metric, converting invalid survival-metric inputs to NaN/error.

    scikit-survival may emit a numerical ``RuntimeWarning`` rather than raising
    when a landmark has no cases at the requested horizon. Treat that outcome
    as an unavailable metric. Scikit-learn's single-class AUROC warning is
    expected in rare-event bootstrap resamples and is suppressed; its non-finite
    result is still recorded as unavailable below.

    Parameters
    ----------
    function : Callable[[], float]
        Metric computation to evaluate.

    Returns
    -------
    tuple[float, str | None]
        Finite metric and ``None`` on success, or ``NaN`` and the captured
        exception/warning text when the metric is not estimable.
    """
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UndefinedMetricWarning)
            warnings.simplefilter("error", RuntimeWarning)
            value = float(function())
        if not np.isfinite(value):
            return np.nan, "metric returned a non-finite value"
        return value, None
    except (ValueError, RuntimeWarning, ZeroDivisionError, FloatingPointError) as error:
        return np.nan, str(error)


def _average_precision(truth, score):
    """Average precision, requiring both endpoint classes to be represented."""
    truth = np.asarray(truth, dtype=int)
    if np.unique(truth).size < 2:
        raise ValueError("AUPRC requires at least one case and one non-case.")
    return float(average_precision_score(truth, score))


def _ap_lift(truth, score):
    """Average-precision lift over the endpoint's observed prevalence."""
    truth = np.asarray(truth, dtype=int)
    prevalence = float(np.mean(truth))
    if prevalence <= 0:
        raise ValueError("AP lift requires at least one case.")
    return _average_precision(truth, score) / prevalence


def _bootstrap_confidence_interval(
    metric, n, *, samples, confidence_level, random_state, key, progress_label=None
):
    """Percentile CI from patient-level resampling of an evaluation cohort."""
    if samples <= 0 or n == 0:
        return np.nan, np.nan, 0
    if not 0 < confidence_level < 1:
        raise ValueError("bootstrap_confidence_level must be in (0, 1).")
    seed = np.random.SeedSequence([
        int(random_state),
        zlib.crc32(str(key).encode("utf-8")),
    ])
    rng = np.random.default_rng(seed)
    values = []
    for sample_index in range(int(samples)):
        indices = rng.integers(0, n, size=n)
        value, _ = _safe_metric(lambda: metric(indices))
        if np.isfinite(value):
            values.append(value)
        completed = sample_index + 1
        if progress_label and (completed == samples or completed % 25 == 0):
            print(
                f"[evaluation] {progress_label}: bootstrap {completed}/{samples} "
                f"({len(values)} valid)",
                flush=True,
            )
    if not values:
        return np.nan, np.nan, 0
    alpha = (1 - confidence_level) / 2
    return (
        float(np.quantile(values, alpha)),
        float(np.quantile(values, 1 - alpha)),
        len(values),
    )


def _add_bootstrap_ci(record, metric, n, *, samples, confidence_level, random_state):
    """Attach patient-bootstrap percentile confidence-interval fields to a record."""
    landmark = record.get("landmark_seconds_since_start")
    endpoint = record.get("minutes_before_delivery")
    location = (
        f"start={float(landmark) / 3600:g}h" if landmark is not None
        else f"end={float(endpoint):g}min" if endpoint is not None
        else "aggregate"
    )
    progress_label = (
        f"{record['split']} {location} {record['target']} {record['metric']}"
    )
    lower, upper, successful = _bootstrap_confidence_interval(
        metric,
        n,
        samples=samples,
        confidence_level=confidence_level,
        random_state=random_state,
        key=(record["split"], record["target"], record["metric"],
             record.get("landmark_seconds_since_start"), record.get("minutes_before_delivery")),
        progress_label=progress_label,
    )
    record.update({
        "ci_lower": lower,
        "ci_upper": upper,
        "bootstrap_samples": int(samples),
        "bootstrap_successful": successful,
        "bootstrap_confidence_level": float(confidence_level),
    })
    return record


def _auprc_record(
    auroc_record, truth, score, *, metric, samples, confidence_level, random_state
):
    """Build an AUPRC companion record, including prevalence and AP lift."""
    truth = np.asarray(truth, dtype=int)
    auroc_record["event_prevalence"] = float(np.mean(truth)) if len(truth) else np.nan
    value, error = _safe_metric(lambda: _average_precision(truth, score))
    ap_lift, ap_lift_error = _safe_metric(lambda: _ap_lift(truth, score))
    record = dict(auroc_record)
    record.update({
        "metric": metric,
        "value": value,
        "error": error,
        "ap_lift": ap_lift,
        "ap_lift_error": ap_lift_error,
    })
    record = _add_bootstrap_ci(
        record,
        lambda indices: _average_precision(truth[indices], score[indices]),
        len(truth), samples=samples, confidence_level=confidence_level,
        random_state=random_state,
    )
    lift_lower, lift_upper, lift_successful = _bootstrap_confidence_interval(
        lambda indices: _ap_lift(truth[indices], score[indices]),
        len(truth), samples=samples, confidence_level=confidence_level,
        random_state=random_state,
        key=(record["split"], record["target"], metric, "ap_lift",
             record.get("landmark_seconds_since_start"), record.get("minutes_before_delivery")),
    )
    record.update({
        "ap_lift_ci_lower": lift_lower,
        "ap_lift_ci_upper": lift_upper,
        "ap_lift_bootstrap_successful": lift_successful,
    })
    return record


def _macro_auprc_record(
    auroc_record, *, metric, average_precision, ap_lift, n, samples,
    confidence_level, random_state,
):
    """Build a macro-threshold AUPRC record and confidence intervals."""
    value, error = _safe_metric(lambda: average_precision(np.arange(n)))
    lift, lift_error = _safe_metric(lambda: ap_lift(np.arange(n)))
    record = dict(auroc_record)
    record.update({
        "metric": metric,
        "value": value,
        "error": error,
        "ap_lift": lift,
        "ap_lift_error": lift_error,
    })
    record = _add_bootstrap_ci(
        record, average_precision, n, samples=samples,
        confidence_level=confidence_level, random_state=random_state,
    )
    lift_lower, lift_upper, lift_successful = _bootstrap_confidence_interval(
        ap_lift, n, samples=samples, confidence_level=confidence_level,
        random_state=random_state,
        key=(record["split"], record["target"], metric, "ap_lift",
             record.get("landmark_seconds_since_start"), record.get("minutes_before_delivery")),
    )
    record.update({
        "ap_lift_ci_lower": lift_lower,
        "ap_lift_ci_upper": lift_upper,
        "ap_lift_bootstrap_successful": lift_successful,
    })
    return record


def _mean_valid_metric(functions):
    """Mean finite component metrics, raising when none are estimable."""
    values = []
    for function in functions:
        value, _ = _safe_metric(function)
        if np.isfinite(value):
            values.append(value)
    if not values:
        raise ValueError("No threshold-specific metric could be estimated.")
    return float(np.mean(values))


def _antolini_concordance(cumulative_risk, time_grid, durations, events):
    """Compute adjusted Antolini concordance from landmark risk curves.

    ``EvalSurv`` expects survival curves arranged as time rows by patient
    columns. The model exposes cumulative risk, so its complement supplies the
    corresponding event-free curve. The metric compares predictions at every
    comparable observed event time instead of using only the final grid value.
    """
    cumulative_risk = np.asarray(cumulative_risk, dtype=float)
    time_grid = np.asarray(time_grid, dtype=float)
    durations = np.asarray(durations, dtype=float)
    events = np.asarray(events, dtype=bool)
    if cumulative_risk.ndim != 2:
        raise ValueError("cumulative_risk must have shape [patients, time bins].")
    if cumulative_risk.shape != (len(durations), len(time_grid)):
        raise ValueError("Risk curves, durations, and time grid have incompatible shapes.")
    survival = pd.DataFrame(
        np.clip(1.0 - cumulative_risk, 0.0, 1.0).T,
        index=time_grid,
    )
    evaluator = EvalSurv(survival, durations, events, steps="post")
    return float(evaluator.concordance_td(method="adj_antolini"))


def _truncate_survival_at_horizon(survival, horizon_seconds):
    """Administratively censor a survival array after one prediction horizon.

    Parameters
    ----------
    survival : numpy.ndarray
        Structured scikit-survival array with event and time fields.
    horizon_seconds : float
        Landmark prediction horizon in seconds.

    Returns
    -------
    numpy.ndarray
        Structured survival array in which events after the horizon are censored
        infinitesimally *after* the horizon.  The small offset leaves the
        target unchanged while satisfying scikit-survival's strict requirement
        that an evaluation time be below the maximum observed follow-up time.
        This prevents a horizon-specific dynamic AUC from requiring IPCW
        support for delivery events many hours later.
    """
    time = np.asarray(survival["time"], dtype=float)
    horizon = float(horizon_seconds)
    event = np.asarray(survival["event"], dtype=bool) & (time <= horizon)
    truncated_time = np.where(time > horizon, np.nextafter(horizon, np.inf), time)
    return Surv.from_arrays(event, truncated_time)


def evaluate_time_to_event(
    *,
    train_data,
    eval_data,
    prediction,
    split_name,
    target_names,
    target_indices=None,
    include_delivery,
    start_landmark_seconds,
    end_landmark_seconds,
    bootstrap_samples=100,
    bootstrap_confidence_level=0.95,
    bootstrap_random_state=42,
):
    """Evaluate one exact landmark pair with model-provided cumulative curves.

    ``prediction`` returns ``{target_name: curve[N, T]}``, with cumulative risk
    curves on the shared ``time_grid_seconds`` supplied as ``"time_grid"``.
    Delivery targets use all rows; pH targets use only observed pH rows.

    Parameters
    ----------
    train_data : LandmarkData
        Retained for caller compatibility; horizon metrics no longer need a
        training-reference cohort.
    eval_data : LandmarkData
        Evaluated cohort at the landmark.
    prediction : callable
        Maps chunk arrays to ``time_grid`` and named cumulative-risk curves.
    split_name : str
        Name written to output rows, usually ``Validation`` or ``Test``.
    target_names : sequence[str]
        Named pH threshold curves to score.
    target_indices : sequence[int] | None
        Corresponding pH target columns for one-threshold models.
    include_delivery : bool
        Include all-delivery-time metrics in addition to pH metrics.
    start_landmark_seconds, end_landmark_seconds : float | None
        Select start-landmark C-index/horizon-AUROC or end-landmark AUROC reports.
    bootstrap_samples, bootstrap_confidence_level, bootstrap_random_state
        Patient-level percentile-bootstrap settings.

    Returns
    -------
    tuple[list[dict], list[dict]]
        Metric records, including confidence intervals, and per-PID raw score
        records for ROC/calibration/postprocessing.
    """
    location = (
        f"start landmark {float(start_landmark_seconds) / 3600:g}h"
        if start_landmark_seconds is not None
        else f"end landmark {float(end_landmark_seconds) / 60:g}min"
    )
    started = time.monotonic()
    print(
        f"[evaluation] {split_name} {location}: predicting {len(eval_data.PIDs):,} evaluation chunks",
        flush=True,
    )
    pred_eval = _predict_landmark_data(prediction, eval_data)
    print(
        f"[evaluation] {split_name} {location}: prediction complete "
        f"(evaluation n={len(eval_data.PIDs):,}); elapsed {time.monotonic() - started:.0f}s",
        flush=True,
    )
    grid = np.asarray(pred_eval.pop("time_grid"), dtype=float)
    records, prediction_rows = [], []
    forecast_horizons = (
        _valid_forecast_horizons(start_landmark_seconds)
        if start_landmark_seconds is not None else np.asarray([], dtype=float)
    )
    targets = list(target_names)
    if include_delivery:
        targets.insert(0, "delivery")
    macro_start_entries, macro_end_entries = [], []
    for target_index, target_name in enumerate(targets):
        print(
            f"[evaluation] {split_name} {location}: scoring target {target_name}",
            flush=True,
        )
        if target_name == "delivery":
            eval_mask = np.ones(len(eval_data.PIDs), dtype=bool)
            eval_event = np.ones(eval_mask.sum(), dtype=bool)
        else:
            ph_target_index = target_index - int(include_delivery)
            if target_indices is not None:
                ph_target_index = target_indices[ph_target_index]
            eval_mask = eval_data.ph_observed
            eval_event = eval_data.ph_targets[eval_mask, ph_target_index] > 0
        if not eval_mask.any():
            continue
        if start_landmark_seconds is not None:
            target_forecast_horizons = forecast_horizons
            eval_surv = Surv.from_arrays(eval_event, eval_data.remaining_seconds[eval_mask])
            risk_curve = np.asarray(pred_eval[target_name])[eval_mask]
            full_grid_risk = risk_curve[:, -1]
            antolini_cindex, error = _safe_metric(
                lambda: _antolini_concordance(
                    risk_curve,
                    grid,
                    eval_surv["time"],
                    eval_surv["event"],
                )
            )
            record = {
                "split": split_name, "target": target_name,
                "metric": "antolini_cindex",
                "landmark_seconds_since_start": start_landmark_seconds,
                "prediction_horizon_seconds": float(grid[-1]),
                "value": antolini_cindex,
                "n": int(eval_mask.sum()), "error": error,
            }
            records.append(_add_bootstrap_ci(
                record,
                lambda indices: _antolini_concordance(
                    risk_curve[indices],
                    grid,
                    eval_surv["time"][indices],
                    eval_surv["event"][indices],
                ),
                len(eval_surv), samples=bootstrap_samples,
                confidence_level=bootstrap_confidence_level, random_state=bootstrap_random_state,
            ))
            horizon_entries = {}
            for score_horizon, outcome_horizon in target_forecast_horizons:
                score = _value_at(pred_eval[target_name], grid, score_horizon)[eval_mask]
                if target_name == "delivery":
                    delivery_by_horizon = (eval_surv["time"] <= outcome_horizon).astype(int)
                    delivery_auc, error = _safe_metric(
                        lambda: roc_auc_score(delivery_by_horizon, score)
                    )
                    record = {
                        "split": split_name, "target": target_name,
                        "metric": "auroc_delivery_by_horizon",
                        "landmark_seconds_since_start": start_landmark_seconds,
                        "prediction_horizon_seconds": float(score_horizon),
                        "outcome_followup_seconds": float(outcome_horizon), "value": delivery_auc,
                        "n": int(len(delivery_by_horizon)),
                        "n_deliveries_within_horizon": int(delivery_by_horizon.sum()),
                        "n_deliveries_after_horizon": int(len(delivery_by_horizon) - delivery_by_horizon.sum()),
                        "error": error,
                    }
                    auprc_record = _auprc_record(
                        record, delivery_by_horizon, score,
                        metric="auprc_delivery_by_horizon",
                        samples=bootstrap_samples,
                        confidence_level=bootstrap_confidence_level,
                        random_state=bootstrap_random_state,
                    )
                    records.append(_add_bootstrap_ci(
                        record,
                        lambda indices, truth=delivery_by_horizon, score=score:
                        roc_auc_score(truth[indices], score[indices]),
                        len(delivery_by_horizon), samples=bootstrap_samples,
                        confidence_level=bootstrap_confidence_level, random_state=bootstrap_random_state,
                    ))
                    records.append(auprc_record)
                else:
                    horizon_surv = _truncate_survival_at_horizon(eval_surv, outcome_horizon)
                    # This is the primary competing-risk discrimination target:
                    # every observed-pH patient remains in the denominator.
                    # Thus non-acidemic deliveries and acidemic deliveries after
                    # the horizon are both explicit controls.
                    event_by_horizon = horizon_surv["event"].astype(int)
                    competing_auc, error = _safe_metric(
                        lambda: roc_auc_score(event_by_horizon, score)
                    )
                    record = {
                        "split": split_name, "target": target_name,
                        "metric": "auroc_acidemic_delivery_by_horizon",
                        "landmark_seconds_since_start": start_landmark_seconds,
                        "prediction_horizon_seconds": float(score_horizon),
                        "outcome_followup_seconds": float(outcome_horizon), "value": competing_auc,
                        "n": int(len(event_by_horizon)),
                        "n_acidemic_deliveries": int(event_by_horizon.sum()),
                        "n_non_cases": int(len(event_by_horizon) - event_by_horizon.sum()),
                        "error": error,
                    }
                    auprc_record = _auprc_record(
                        record, event_by_horizon, score,
                        metric="auprc_acidemic_delivery_by_horizon",
                        samples=bootstrap_samples,
                        confidence_level=bootstrap_confidence_level,
                        random_state=bootstrap_random_state,
                    )
                    records.append(_add_bootstrap_ci(
                        record,
                        lambda indices, truth=event_by_horizon, score=score:
                        roc_auc_score(truth[indices], score[indices]),
                        len(event_by_horizon), samples=bootstrap_samples,
                        confidence_level=bootstrap_confidence_level, random_state=bootstrap_random_state,
                    ))
                    records.append(auprc_record)
                    delivered = eval_surv["time"] <= outcome_horizon
                    truth = eval_surv["event"][delivered].astype(int)
                    delivery_score = score[delivered]
                    conditional_auc, error = _safe_metric(
                        lambda: roc_auc_score(truth, delivery_score)
                    )
                    record = {
                        "split": split_name, "target": target_name,
                        "metric": "auroc_acidemic_delivery_given_delivery",
                        "landmark_seconds_since_start": start_landmark_seconds,
                        "prediction_horizon_seconds": float(score_horizon),
                        "outcome_followup_seconds": float(outcome_horizon), "value": conditional_auc,
                        "n": int(len(truth)), "n_acidemic_deliveries": int(truth.sum()),
                        "n_non_acidemic_deliveries": int(len(truth) - truth.sum()), "error": error,
                    }
                    auprc_record = _auprc_record(
                        record, truth, delivery_score,
                        metric="auprc_acidemic_delivery_given_delivery",
                        samples=bootstrap_samples,
                        confidence_level=bootstrap_confidence_level,
                        random_state=bootstrap_random_state,
                    )
                    records.append(_add_bootstrap_ci(
                        record,
                        lambda indices, truth=truth, delivery_score=delivery_score:
                        roc_auc_score(truth[indices], delivery_score[indices]),
                        len(truth), samples=bootstrap_samples,
                        confidence_level=bootstrap_confidence_level, random_state=bootstrap_random_state,
                    ))
                    records.append(auprc_record)
                    horizon_entries[float(score_horizon)] = {
                        "score": score,
                        "event_by_horizon": event_by_horizon,
                        "delivery_truth": truth, "delivery_score": delivery_score,
                        "outcome_followup_seconds": float(outcome_horizon),
                    }
            if target_name != "delivery":
                macro_start_entries.append({
                    "eval_surv": eval_surv,
                    "risk_curve": risk_curve,
                    "horizons": horizon_entries,
                })
        if end_landmark_seconds is not None and target_name != "delivery":
            score = _value_at(pred_eval[target_name], grid, end_landmark_seconds)[eval_mask]
            truth = eval_event.astype(int)
            value = np.nan if np.unique(truth).size < 2 else float(roc_auc_score(truth, score))
            record = {
                "split": split_name, "target": target_name, "metric": "auroc_end_landmark",
                "minutes_before_delivery": end_landmark_seconds / 60, "value": value,
                "n": int(eval_mask.sum()), "error": None,
            }
            auprc_record = _auprc_record(
                record, truth, score, metric="auprc_end_landmark",
                samples=bootstrap_samples,
                confidence_level=bootstrap_confidence_level,
                random_state=bootstrap_random_state,
            )
            records.append(_add_bootstrap_ci(
                record,
                lambda indices: roc_auc_score(truth[indices], score[indices]),
                len(truth), samples=bootstrap_samples,
                confidence_level=bootstrap_confidence_level, random_state=bootstrap_random_state,
            ))
            records.append(auprc_record)
            macro_end_entries.append({"truth": truth, "score": score})
            for pid, event, remaining_seconds, end_score in zip(
                eval_data.PIDs[eval_mask], truth, eval_data.remaining_seconds[eval_mask], score,
            ):
                prediction_rows.append({
                    "split": split_name, "target": target_name, "PID": pid,
                    "minutes_before_delivery": end_landmark_seconds / 60,
                    "event": int(event),
                    "remaining_seconds": float(remaining_seconds),
                    "end_landmark_risk": end_score,
                })
        if start_landmark_seconds is not None:
            for score_horizon, outcome_horizon in target_forecast_horizons:
                score = _value_at(pred_eval[target_name], grid, score_horizon)[eval_mask]
                for pid, event, remaining_seconds, risk_through_end, horizon_score in zip(
                    eval_data.PIDs[eval_mask], eval_event, eval_data.remaining_seconds[eval_mask],
                    full_grid_risk, score,
                ):
                    prediction_rows.append({
                        "split": split_name, "target": target_name, "PID": pid,
                        "landmark_seconds_since_start": start_landmark_seconds,
                        "prediction_horizon_seconds": float(score_horizon),
                        "outcome_followup_seconds": float(outcome_horizon), "event": int(event),
                        "remaining_seconds": float(remaining_seconds),
                        "event_within_horizon": int(event and remaining_seconds <= outcome_horizon),
                        "delivery_within_horizon": int(remaining_seconds <= outcome_horizon),
                        "risk_through_model_grid": risk_through_end,
                        "horizon_risk": horizon_score,
                    })
    if len(macro_start_entries) > 1:
        n = len(macro_start_entries[0]["eval_surv"])

        def macro_antolini_cindex(indices):
            """Average threshold-specific Antolini C-indexes for a resample."""
            return _mean_valid_metric([
                lambda entry=entry: _antolini_concordance(
                    entry["risk_curve"][indices],
                    grid,
                    entry["eval_surv"]["time"][indices],
                    entry["eval_surv"]["event"][indices],
                )
                for entry in macro_start_entries
            ])

        for metric_name, metric_function, horizon in [
            ("antolini_cindex", macro_antolini_cindex, float(grid[-1])),
        ]:
            value, error = _safe_metric(lambda: metric_function(np.arange(n)))
            record = {
                "split": split_name,
                "target": "macro_pH",
                "metric": metric_name,
                "landmark_seconds_since_start": start_landmark_seconds,
                "prediction_horizon_seconds": horizon,
                "value": value,
                "n": n,
                "macro_threshold_count": len(macro_start_entries),
                "error": error,
            }
            records.append(_add_bootstrap_ci(
                record,
                metric_function,
                n,
                samples=bootstrap_samples,
                confidence_level=bootstrap_confidence_level,
                random_state=bootstrap_random_state,
            ))
        for score_horizon, outcome_horizon in forecast_horizons:
            score_horizon = float(score_horizon)
            outcome_horizon = float(outcome_horizon)
            entries = [entry["horizons"][score_horizon] for entry in macro_start_entries]
            n_horizon = len(entries[0]["event_by_horizon"])
            def macro_competing_auc(indices, entries=entries):
                """Average competing-risk horizon AUROC across pH thresholds."""
                return _mean_valid_metric([
                    lambda entry=entry: roc_auc_score(
                        entry["event_by_horizon"][indices], entry["score"][indices]
                    )
                    for entry in entries
                ])

            def macro_competing_ap(indices, entries=entries):
                """Average competing-risk horizon AUPRC across pH thresholds."""
                return _mean_valid_metric([
                    lambda entry=entry: _average_precision(
                        entry["event_by_horizon"][indices], entry["score"][indices]
                    )
                    for entry in entries
                ])

            def macro_competing_ap_lift(indices, entries=entries):
                """Average AP lift for competing-risk horizons across thresholds."""
                return _mean_valid_metric([
                    lambda entry=entry: _ap_lift(
                        entry["event_by_horizon"][indices], entry["score"][indices]
                    )
                    for entry in entries
                ])

            value, error = _safe_metric(lambda: macro_competing_auc(np.arange(n_horizon)))
            record = {
                "split": split_name, "target": "macro_pH",
                "metric": "auroc_acidemic_delivery_by_horizon",
                "landmark_seconds_since_start": start_landmark_seconds,
                "prediction_horizon_seconds": score_horizon,
                "outcome_followup_seconds": outcome_horizon, "value": value, "n": n_horizon,
                "macro_threshold_count": len(entries),
                "event_prevalence": float(np.mean([
                    np.mean(entry["event_by_horizon"]) for entry in entries
                ])),
                "error": error,
            }
            auprc_record = _macro_auprc_record(
                record, metric="auprc_acidemic_delivery_by_horizon",
                average_precision=macro_competing_ap,
                ap_lift=macro_competing_ap_lift,
                n=n_horizon, samples=bootstrap_samples,
                confidence_level=bootstrap_confidence_level,
                random_state=bootstrap_random_state,
            )
            records.append(_add_bootstrap_ci(
                record, macro_competing_auc, n_horizon, samples=bootstrap_samples,
                confidence_level=bootstrap_confidence_level, random_state=bootstrap_random_state,
            ))
            records.append(auprc_record)
            delivery_n = len(entries[0]["delivery_truth"])

            def macro_delivery_auc(indices, entries=entries):
                """Average pH AUROCs among deliveries within one shared horizon."""
                return _mean_valid_metric([
                    lambda entry=entry: roc_auc_score(
                        entry["delivery_truth"][indices], entry["delivery_score"][indices]
                    )
                    for entry in entries
                ])

            def macro_delivery_ap(indices, entries=entries):
                """Average conditional-delivery AUPRC across pH thresholds."""
                return _mean_valid_metric([
                    lambda entry=entry: _average_precision(
                        entry["delivery_truth"][indices], entry["delivery_score"][indices]
                    )
                    for entry in entries
                ])

            def macro_delivery_ap_lift(indices, entries=entries):
                """Average conditional-delivery AP lift across pH thresholds."""
                return _mean_valid_metric([
                    lambda entry=entry: _ap_lift(
                        entry["delivery_truth"][indices], entry["delivery_score"][indices]
                    )
                    for entry in entries
                ])

            value, error = _safe_metric(lambda: macro_delivery_auc(np.arange(delivery_n)))
            record = {
                "split": split_name, "target": "macro_pH",
                "metric": "auroc_acidemic_delivery_given_delivery",
                "landmark_seconds_since_start": start_landmark_seconds,
                "prediction_horizon_seconds": score_horizon,
                "outcome_followup_seconds": outcome_horizon, "value": value, "n": int(delivery_n),
                "macro_threshold_count": len(entries),
                "event_prevalence": float(np.mean([
                    np.mean(entry["delivery_truth"]) for entry in entries
                ])),
                "error": error,
            }
            auprc_record = _macro_auprc_record(
                record, metric="auprc_acidemic_delivery_given_delivery",
                average_precision=macro_delivery_ap,
                ap_lift=macro_delivery_ap_lift,
                n=delivery_n, samples=bootstrap_samples,
                confidence_level=bootstrap_confidence_level,
                random_state=bootstrap_random_state,
            )
            records.append(_add_bootstrap_ci(
                record, macro_delivery_auc, delivery_n, samples=bootstrap_samples,
                confidence_level=bootstrap_confidence_level, random_state=bootstrap_random_state,
            ))
            records.append(auprc_record)
    if len(macro_end_entries) > 1:
        n = len(macro_end_entries[0]["truth"])

        def macro_end_auc(indices):
            """Average threshold-specific end-landmark AUROCs for a resample."""
            return _mean_valid_metric([
                lambda entry=entry: roc_auc_score(entry["truth"][indices], entry["score"][indices])
                for entry in macro_end_entries
            ])

        def macro_end_ap(indices):
            """Average end-landmark AUPRC across pH thresholds."""
            return _mean_valid_metric([
                lambda entry=entry: _average_precision(entry["truth"][indices], entry["score"][indices])
                for entry in macro_end_entries
            ])

        def macro_end_ap_lift(indices):
            """Average end-landmark AP lift across pH thresholds."""
            return _mean_valid_metric([
                lambda entry=entry: _ap_lift(entry["truth"][indices], entry["score"][indices])
                for entry in macro_end_entries
            ])

        value, error = _safe_metric(lambda: macro_end_auc(np.arange(n)))
        record = {
            "split": split_name,
            "target": "macro_pH",
            "metric": "auroc_end_landmark",
            "minutes_before_delivery": end_landmark_seconds / 60,
            "value": value,
            "n": n,
            "macro_threshold_count": len(macro_end_entries),
            "event_prevalence": float(np.mean([
                np.mean(entry["truth"]) for entry in macro_end_entries
            ])),
            "error": error,
        }
        auprc_record = _macro_auprc_record(
            record, metric="auprc_end_landmark", average_precision=macro_end_ap,
            ap_lift=macro_end_ap_lift, n=n, samples=bootstrap_samples,
            confidence_level=bootstrap_confidence_level,
            random_state=bootstrap_random_state,
        )
        records.append(_add_bootstrap_ci(
            record,
            macro_end_auc,
            n,
            samples=bootstrap_samples,
            confidence_level=bootstrap_confidence_level,
            random_state=bootstrap_random_state,
        ))
        records.append(auprc_record)
    print(
        f"[evaluation] {split_name} {location}: complete; "
        f"records={len(records):,}, predictions={len(prediction_rows):,}, "
        f"elapsed {time.monotonic() - started:.0f}s",
        flush=True,
    )
    return records, prediction_rows


def _subset(data, mask):
    """Return a LandmarkData row subset while preserving aligned fields."""
    mask = np.asarray(mask, dtype=bool)
    return LandmarkData(
        data.X[mask], data.PIDs[mask], data.remaining_seconds[mask], data.elapsed_seconds[mask],
        data.ph_targets[mask], data.ph_observed[mask], data.thresholds,
    )


def evaluate_landmark_suite(
    *,
    trace_file,
    train_label_file,
    test_label_file,
    features,
    chunk_window_size,
    lab_order_delay,
    prediction,
    target_names,
    target_indices=None,
    include_delivery,
    splits_file=None,
    require_label=False,
    missingness_indicator_channels=False,
    missing_data_method="ffill",
    chunk_missingness_max_fraction=None,
    history_window_count=1,
    bootstrap_samples=100,
    bootstrap_confidence_level=0.95,
    bootstrap_random_state=42,
):
    """Run and collect the complete predefined landmark evaluation suite.

    Parameters
    ----------
    trace_file, train_label_file, test_label_file : str | None
        Trace data, train/validation labels, and optional held-out test labels.
    features, chunk_window_size, lab_order_delay :
        Chunk extraction and label-timeliness configuration.
    prediction : callable
        Model-specific cumulative-risk prediction function.
    target_names, target_indices, include_delivery :
        Outcome curves and target-column mapping to report.
    splits_file : str | None
        PID-to-fold table used to partition train versus validation data.
    require_label : bool
        Whether chunks without observed pH are excluded before loading.
    missingness_indicator_channels : bool
        Append unscaled pre-imputation observation masks to landmark chunks.
    chunk_missingness_max_fraction : float | None
        Apply the training chunk-quality rule to each exact evaluation window.
    bootstrap_samples, bootstrap_confidence_level, bootstrap_random_state
        Patient-level confidence-interval configuration.

    Returns
    -------
    tuple[pandas.DataFrame, pandas.DataFrame]
        Landmark metric table and per-PID raw risk-score table.
    """
    if splits_file is not None:
        folds = pd.read_csv(splits_file).set_index("PID")["fold"]

        def train_select(pids):
            """Select PIDs assigned to the training fold."""
            return np.asarray([folds.get(pid, "Train") == "Train" for pid in pids])

        def val_select(pids):
            """Select PIDs assigned to the validation fold."""
            return np.asarray([folds.get(pid, "Train") == "Validation" for pid in pids])
    else:
        def train_select(pids):
            """Use all PIDs as training references when no split table is supplied."""
            return np.ones(len(pids), dtype=bool)

        val_select = train_select
    suite_started = time.monotonic()
    records, predictions = [], []
    print(
        f"[evaluation] suite start: {len(START_LANDMARK_SECONDS)} start landmarks, "
        f"{len(END_LANDMARK_SECONDS)} end landmarks, {bootstrap_samples} bootstraps",
        flush=True,
    )
    for landmark in START_LANDMARK_SECONDS:
        forecast_horizons = _valid_forecast_horizons(landmark)
        if not len(forecast_horizons):
            print(
                f"[evaluation] skipping start landmark {float(landmark) / 3600:g}h: "
                f"no configured forecast horizon fits within the "
                f"{MAX_TRACING_SECONDS / 3600:g}h trace cap",
                flush=True,
            )
            continue
        landmark_label = f"start landmark {float(landmark) / 3600:g}h"
        base = load_landmark_data(
        trace_file, train_label_file, features=features, landmark_seconds=landmark,
        chunk_window_size=chunk_window_size, lab_order_delay=lab_order_delay,
        missing=missing_data_method,
        require_label=require_label, missingness_indicator_channels=missingness_indicator_channels,
            chunk_missingness_max_fraction=chunk_missingness_max_fraction,
            history_window_count=history_window_count,
            progress_label=landmark_label,
        )
        if not len(base.PIDs):
            print(
                f"[evaluation] {landmark_label}: no tracing reaches this landmark; skipping",
                flush=True,
            )
            continue
        train_data, val_data = _subset(base, train_select(base.PIDs)), _subset(base, val_select(base.PIDs))
        print(
            f"[evaluation] {landmark_label}: cohort ready; train n={len(train_data.PIDs):,}, "
            f"validation n={len(val_data.PIDs):,}",
            flush=True,
        )
        if len(train_data.PIDs) and len(val_data.PIDs):
            result = evaluate_time_to_event(
                train_data=train_data, eval_data=val_data, prediction=prediction,
                split_name="Validation", target_names=target_names, include_delivery=include_delivery,
                target_indices=target_indices,
                start_landmark_seconds=landmark, end_landmark_seconds=None,
                bootstrap_samples=bootstrap_samples,
                bootstrap_confidence_level=bootstrap_confidence_level,
                bootstrap_random_state=bootstrap_random_state,
            )
            records.extend(result[0])
            predictions.extend(result[1])
        if test_label_file:
            test_data = load_landmark_data(
            trace_file, test_label_file, features=features, landmark_seconds=landmark,
            chunk_window_size=chunk_window_size, lab_order_delay=lab_order_delay,
            missing=missing_data_method,
                require_label=require_label, missingness_indicator_channels=missingness_indicator_channels,
                chunk_missingness_max_fraction=chunk_missingness_max_fraction,
                history_window_count=history_window_count,
                progress_label=f"{landmark_label} test",
            )
            if len(train_data.PIDs) and len(test_data.PIDs):
                result = evaluate_time_to_event(
                    train_data=train_data, eval_data=test_data, prediction=prediction,
                    split_name="Test", target_names=target_names, include_delivery=include_delivery,
                    target_indices=target_indices,
                    start_landmark_seconds=landmark, end_landmark_seconds=None,
                    bootstrap_samples=bootstrap_samples,
                    bootstrap_confidence_level=bootstrap_confidence_level,
                    bootstrap_random_state=bootstrap_random_state,
                )
                records.extend(result[0])
                predictions.extend(result[1])
    for landmark in END_LANDMARK_SECONDS:
        landmark_label = f"end landmark {float(landmark) / 60:g}min"
        base = load_landmark_data(
            trace_file, train_label_file, features=features, landmark_seconds=landmark,
            from_end=True, chunk_window_size=chunk_window_size, lab_order_delay=lab_order_delay,
            missing=missing_data_method,
            require_label=require_label, missingness_indicator_channels=missingness_indicator_channels,
            chunk_missingness_max_fraction=chunk_missingness_max_fraction,
            history_window_count=history_window_count,
            progress_label=landmark_label,
        )
        if not len(base.PIDs):
            print(
                f"[evaluation] {landmark_label}: no tracing reaches this landmark; skipping",
                flush=True,
            )
            continue
        for split_name, data in [("Validation", _subset(base, val_select(base.PIDs)))]:
            if len(data.PIDs):
                result = evaluate_time_to_event(
                    train_data=data, eval_data=data, prediction=prediction, split_name=split_name,
                    target_names=target_names, include_delivery=include_delivery,
                    target_indices=target_indices,
                    start_landmark_seconds=None, end_landmark_seconds=landmark,
                    bootstrap_samples=bootstrap_samples,
                    bootstrap_confidence_level=bootstrap_confidence_level,
                    bootstrap_random_state=bootstrap_random_state,
                )
                records.extend(result[0])
                predictions.extend(result[1])
        if test_label_file:
            test_data = load_landmark_data(
            trace_file, test_label_file, features=features, landmark_seconds=landmark,
            from_end=True, chunk_window_size=chunk_window_size, lab_order_delay=lab_order_delay,
            missing=missing_data_method,
                require_label=require_label, missingness_indicator_channels=missingness_indicator_channels,
                chunk_missingness_max_fraction=chunk_missingness_max_fraction,
                history_window_count=history_window_count,
                progress_label=f"{landmark_label} test",
            )
            if len(test_data.PIDs):
                result = evaluate_time_to_event(
                    train_data=test_data, eval_data=test_data, prediction=prediction, split_name="Test",
                    target_names=target_names, include_delivery=include_delivery,
                    target_indices=target_indices,
                    start_landmark_seconds=None, end_landmark_seconds=landmark,
                    bootstrap_samples=bootstrap_samples,
                    bootstrap_confidence_level=bootstrap_confidence_level,
                    bootstrap_random_state=bootstrap_random_state,
                )
                records.extend(result[0])
                predictions.extend(result[1])
    print(
        f"[evaluation] suite complete: records={len(records):,}, predictions={len(predictions):,}, "
        f"elapsed {time.monotonic() - suite_started:.0f}s",
        flush=True,
    )
    return pd.DataFrame(records), pd.DataFrame(predictions)
