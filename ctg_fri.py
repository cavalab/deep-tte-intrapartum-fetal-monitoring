"""A transparent CTG-only Fetal Reserve Index (FRI) baseline.

This is deliberately a *partial* FRI.  The published FRI has eight binary
components: five CTG components plus maternal, obstetric, and fetal clinical
risk factors.  The latter are not assumed observed here, so this module
reports a five-point CTG score (scaled to 0--100) and never silently fills in
the three clinical points.

The signal definitions follow FIGO 2015 where they apply: a baseline is
assessed over the final 10 minutes; accelerations and decelerations require a
15 bpm amplitude and 15 s duration; and uterine tachysystole is more than five
contractions per 10 minutes.  External TOCO is used only for contraction
timing/count, not contraction intensity.  This is a reproducible rule-based
approximation, not a certified clinical decision aid or an exact reproduction
of proprietary FRI software.
"""
from dataclasses import dataclass

import numpy as np


FIGO_BASELINE_RANGE = (110.0, 160.0)
FIGO_VARIABILITY_RANGE = (5.0, 25.0)


@dataclass(frozen=True)
class CTGFRIResult:
    """One CTG-only FRI score and its five transparent binary components."""

    score: float
    adverse_risk: float
    baseline_point: int
    variability_point: int
    acceleration_point: int
    deceleration_point: int
    uterine_activity_point: int
    baseline_bpm: float
    variability_bpm: float
    acceleration_count: int
    deceleration_count: int
    contraction_count: int


def _finite_tail(values, samples):
    """Return the requested tail, retaining NaNs for coverage checks."""
    values = np.asarray(values, dtype=float).ravel()
    return values[-min(len(values), int(samples)):]


def _runs(mask):
    """Return inclusive start/end pairs for true runs in a boolean vector."""
    padded = np.r_[False, np.asarray(mask, dtype=bool), False]
    changes = np.diff(padded.astype(np.int8))
    return zip(np.flatnonzero(changes == 1), np.flatnonzero(changes == -1) - 1)


def _event_count(signal, baseline, *, direction, sample_rate_hz,
                 min_amplitude_bpm=15.0, min_seconds=15.0,
                 max_seconds=10 * 60):
    """Count FIGO amplitude/duration FHR excursions relative to baseline."""
    if not np.isfinite(baseline):
        return 0
    if direction == "up":
        mask = np.isfinite(signal) & (signal - baseline >= min_amplitude_bpm)
    elif direction == "down":
        mask = np.isfinite(signal) & (baseline - signal >= min_amplitude_bpm)
    else:  # pragma: no cover - developer invariant.
        raise ValueError("direction must be 'up' or 'down'")
    count = 0
    for start, end in _runs(mask):
        duration = (end - start + 1) / float(sample_rate_hz)
        if min_seconds <= duration < max_seconds:
            count += 1
    return count


def _variability_bpm(fhr, sample_rate_hz):
    """Estimate FIGO's average one-minute FHR bandwidth robustly.

    FIGO describes variability as the average bandwidth amplitude in one-minute
    segments.  We use each segment's 5th--95th percentile range rather than
    raw extrema so isolated artifacts do not determine the score.
    """
    per_minute = max(1, int(round(60 * sample_rate_hz)))
    ranges = []
    for start in range(0, len(fhr), per_minute):
        segment = fhr[start:start + per_minute]
        segment = segment[np.isfinite(segment)]
        if len(segment) >= per_minute * 0.8:
            ranges.append(np.percentile(segment, 95) - np.percentile(segment, 5))
    return float(np.mean(ranges)) if ranges else np.nan


def _contraction_count(toco, sample_rate_hz):
    """Count broad external-TOCO peaks, using timing but not intensity.

    The channel has no reliably comparable intensity scale across patients.
    Peaks are therefore detected after smoothing, with an adaptive prominence
    threshold and a one-minute refractory period.  This is an approximation
    for FIGO contraction frequency, not a measurement of uterine strength.
    """
    toco = np.asarray(toco, dtype=float).ravel()
    finite = np.isfinite(toco)
    if finite.mean() < 0.8 or finite.sum() < 3:
        return 0
    filled = np.interp(np.arange(len(toco)), np.flatnonzero(finite), toco[finite])
    smooth_width = max(1, int(round(15 * sample_rate_hz)))
    smooth = np.convolve(filled, np.ones(smooth_width) / smooth_width, mode="same")
    spread = np.percentile(smooth, 75) - np.percentile(smooth, 25)
    if not np.isfinite(spread) or spread <= 0:
        return 0
    prominence = 0.5 * spread
    refractory = max(1, int(round(60 * sample_rate_hz)))
    peaks, last_peak = 0, -refractory
    for index in range(1, len(smooth) - 1):
        if index - last_peak < refractory:
            continue
        if smooth[index] >= smooth[index - 1] and smooth[index] > smooth[index + 1]:
            left = smooth[max(0, index - refractory):index + 1].min()
            right = smooth[index:min(len(smooth), index + refractory + 1)].min()
            if smooth[index] - max(left, right) >= prominence:
                peaks += 1
                last_peak = index
    return peaks


def ctg_only_fri(fhr, toco, *, sample_rate_hz=1.0, analysis_seconds=30 * 60,
                 require_acceleration=True):
    """Return a five-component CTG-only FRI from the latest trace segment.

    ``score`` is 100 times the fraction of available CTG points (0, 20, ...,
    100). ``adverse_risk = 100 - score`` is the direction used for AUROC and
    AUPRC: higher values represent a more abnormal CTG.

    ``require_acceleration=True`` preserves the historical binary FRI
    acceleration component.  FIGO notes that absent accelerations alone are of
    uncertain significance; analyses should therefore report this choice and
    may run ``require_acceleration=False`` as a FIGO-aligned sensitivity
    analysis.
    """
    rate = float(sample_rate_hz)
    if not np.isfinite(rate) or rate <= 0:
        raise ValueError("sample_rate_hz must be positive")
    analysis_samples = max(1, int(round(float(analysis_seconds) * rate)))
    fhr = _finite_tail(fhr, analysis_samples)
    toco = _finite_tail(toco, analysis_samples)
    baseline_window = _finite_tail(fhr, 10 * 60 * rate)
    baseline = float(np.nanmedian(baseline_window)) if np.isfinite(baseline_window).any() else np.nan
    variability = _variability_bpm(baseline_window, rate)
    accelerations = _event_count(fhr, baseline, direction="up", sample_rate_hz=rate)
    decelerations = _event_count(fhr, baseline, direction="down", sample_rate_hz=rate)
    contractions = _contraction_count(toco, rate)

    baseline_point = int(np.isfinite(baseline) and FIGO_BASELINE_RANGE[0] <= baseline <= FIGO_BASELINE_RANGE[1])
    variability_point = int(np.isfinite(variability) and FIGO_VARIABILITY_RANGE[0] <= variability <= FIGO_VARIABILITY_RANGE[1])
    acceleration_point = int(accelerations > 0) if require_acceleration else 1
    # Isolated decelerations may be physiological; recurrent (>=3 / 30 min)
    # FIGO-defined decelerations are the rule-based abnormality used here.
    deceleration_point = int(decelerations < 3)
    # FIGO: tachysystole is >5 contractions / 10 min in two successive periods
    # or averaged over 30 min.  A 30-min window permits the latter criterion.
    uterine_activity_point = int(contractions <= 15)
    points = (baseline_point + variability_point + acceleration_point +
              deceleration_point + uterine_activity_point)
    score = 100.0 * points / 5.0
    return CTGFRIResult(
        score=score, adverse_risk=100.0 - score,
        baseline_point=baseline_point, variability_point=variability_point,
        acceleration_point=acceleration_point, deceleration_point=deceleration_point,
        uterine_activity_point=uterine_activity_point,
        baseline_bpm=baseline, variability_bpm=variability,
        acceleration_count=accelerations, deceleration_count=decelerations,
        contraction_count=contractions,
    )


def ctg_only_fri_risk(chunks, *, sample_rate_hz=1.0, fhr_index=0, toco_index=1,
                      require_acceleration=True):
    """Vectorize :func:`ctg_only_fri` over ``[patient, time, feature]`` chunks."""
    chunks = np.asarray(chunks, dtype=float)
    if chunks.ndim != 3:
        raise ValueError("CTG-only FRI requires chunks shaped [patient, time, feature].")
    if max(fhr_index, toco_index) >= chunks.shape[2]:
        raise ValueError("Requested FHR/TOCO index is not present in chunks.")
    return np.asarray([
        ctg_only_fri(
            row[:, fhr_index], row[:, toco_index], sample_rate_hz=sample_rate_hz,
            require_acceleration=require_acceleration,
        ).adverse_risk
        for row in chunks
    ], dtype=float)
