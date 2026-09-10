"""Feasible 1 Hz FIGO-rule CTG features for tabular survival baselines.

These features approximate, rather than reproduce, the PeriCALM-derived
features in Vargas-Calixto et al. (2025).  We assume tracing start is labor
onset and use rule-based FIGO event labels.  At 1 Hz the >0.5 Hz spectral band
is not observable, so no HF feature is emitted.
"""
import numpy as np


def feature_names():
    return [
        "fhr_baseline_bpm", "fhr_baseline_slope_bpm_per_hour", "fhr_variability_bpm",
        "acceleration_count", "acceleration_duration_s", "acceleration_area_bpm_s",
        "deceleration_count", "deceleration_duration_s", "deceleration_area_bpm_s",
        "fhr_baseline_mean", "fhr_acceleration_mean", "fhr_deceleration_mean",
        "contraction_count", "contraction_duration_s", "contraction_rest_s",
        "fhr_event_transition_count", "lf_power", "mf_power", "lf_mf_ratio",
        "approximate_entropy", "sample_entropy", "hurst_rs", "deceleration_reserve",
    ]


def _fill(values):
    values = np.asarray(values, dtype=float).ravel()
    finite = np.isfinite(values)
    if not finite.any():
        return np.zeros_like(values), False
    return np.interp(np.arange(len(values)), np.flatnonzero(finite), values[finite]), True


def _runs(mask):
    change = np.diff(np.r_[False, np.asarray(mask, bool), False].astype(np.int8))
    return list(zip(np.flatnonzero(change == 1), np.flatnonzero(change == -1)))


def _valid_events(mask, rate, min_seconds=15, max_seconds=600):
    valid = np.zeros(len(mask), dtype=bool)
    count = 0
    for start, stop in _runs(mask):
        duration = (stop - start) / rate
        if min_seconds <= duration < max_seconds:
            valid[start:stop] = True
            count += 1
    return valid, count


def _entropy(values, sample=False):
    # A bounded, downsampled implementation keeps this feasible in large panels.
    values = np.asarray(values, float)
    values = values[::max(1, len(values) // 300)]
    if len(values) < 20 or np.std(values) == 0:
        return 0.0
    m, tolerance = 2, 0.2 * np.std(values)
    vectors_m = np.asarray([values[i:i + m] for i in range(len(values) - m)])
    vectors_m1 = np.asarray([values[i:i + m + 1] for i in range(len(values) - m - 1)])
    def probability(vectors, self_matches):
        distance = np.max(np.abs(vectors[:, None] - vectors[None, :]), axis=2)
        matches = (distance <= tolerance).sum(axis=1) - int(self_matches)
        denom = len(vectors) - int(self_matches)
        return np.maximum(matches / max(denom, 1), 1e-12)
    if sample:
        a, b = probability(vectors_m, True).sum(), probability(vectors_m1, True).sum()
        return float(-np.log(max(b, 1e-12) / max(a, 1e-12)))
    return float(np.mean(np.log(probability(vectors_m, False))) -
                 np.mean(np.log(probability(vectors_m1, False))))


def _hurst_rs(values):
    values = np.asarray(values, float)
    values = values[::max(1, len(values) // 512)]
    if len(values) < 32 or np.std(values) == 0:
        return 0.5
    sizes, rs = [], []
    for size in (8, 16, 32, 64, 128):
        if size > len(values):
            continue
        blocks = values[:len(values) // size * size].reshape(-1, size)
        centered = blocks - blocks.mean(axis=1, keepdims=True)
        ranges = np.ptp(np.cumsum(centered, axis=1), axis=1)
        std = blocks.std(axis=1)
        valid = std > 0
        if valid.any(): sizes.append(size); rs.append(np.mean(ranges[valid] / std[valid]))
    return float(np.polyfit(np.log(sizes), np.log(rs), 1)[0]) if len(sizes) >= 2 else 0.5


def _spectral_power(values, rate, low, high):
    values = values - values.mean()
    frequencies = np.fft.rfftfreq(len(values), d=1 / rate)
    power = np.abs(np.fft.rfft(values)) ** 2 / max(len(values), 1)
    return float(power[(frequencies >= low) & (frequencies < high)].sum())


def _contractions(toco, rate):
    smooth_width = max(1, int(15 * rate))
    smooth = np.convolve(toco, np.ones(smooth_width) / smooth_width, mode="same")
    spread = np.percentile(smooth, 75) - np.percentile(smooth, 25)
    if spread <= 0: return np.zeros(len(toco), bool), 0
    threshold = np.median(smooth) + 0.5 * spread
    mask = smooth >= threshold
    # A contraction must occupy at least 30 seconds; external TOCO contributes
    # frequency/timing only, never an intensity feature.
    return _valid_events(mask, rate, min_seconds=30, max_seconds=5 * 60)


def extract_ctg_features(chunk, *, sample_rate_hz=1.0, fhr_index=0, toco_index=1):
    """Extract one fixed, finite feature vector from ``[time, fecg, toco]``."""
    chunk = np.asarray(chunk, float)
    fhr, fhr_present = _fill(chunk[:, fhr_index])
    toco, _ = _fill(chunk[:, toco_index])
    rate = float(sample_rate_hz)
    baseline_samples = min(len(fhr), int(10 * 60 * rate))
    baseline = float(np.median(fhr[-baseline_samples:])) if fhr_present else 0.0
    minute = max(1, int(60 * rate))
    bands = [np.percentile(fhr[i:i + minute], 95) - np.percentile(fhr[i:i + minute], 5)
             for i in range(0, len(fhr) - minute + 1, minute)]
    variability = float(np.mean(bands)) if bands else 0.0
    acceleration, n_acc = _valid_events(fhr - baseline >= 15, rate)
    deceleration, n_dec = _valid_events(baseline - fhr >= 15, rate)
    contractions, n_con = _contractions(toco, rate)
    transition_count = int(np.count_nonzero(np.diff((acceleration | deceleration).astype(np.int8))))
    x = np.arange(len(fhr)) / rate
    slope = float(np.polyfit(x, fhr, 1)[0] * 3600) if len(fhr) > 1 else 0.0
    lf = _spectral_power(fhr, rate, .04, .15) if len(fhr) > 4 else 0.0
    mf = _spectral_power(fhr, rate, .15, .5) if len(fhr) > 4 else 0.0
    anchors = np.flatnonzero(np.diff(fhr, prepend=fhr[0]) <= -1.0)
    reserve = 0.0
    half = int(30 * rate)
    valid = anchors[(anchors >= half) & (anchors + half < len(fhr))]
    if len(valid): reserve = float(np.mean([fhr[a - half:a].mean() - fhr[a:a + half].mean() for a in valid]))
    values = np.asarray([
        baseline, slope, variability,
        n_acc, acceleration.sum() / rate, np.maximum(fhr[acceleration] - baseline, 0).sum() / rate,
        n_dec, deceleration.sum() / rate, np.maximum(baseline - fhr[deceleration], 0).sum() / rate,
        np.mean(fhr[~(acceleration | deceleration)]) if (~(acceleration | deceleration)).any() else baseline,
        np.mean(fhr[acceleration]) if acceleration.any() else baseline,
        np.mean(fhr[deceleration]) if deceleration.any() else baseline,
        n_con, contractions.sum() / rate, (~contractions).sum() / rate,
        transition_count, lf, mf, lf / (mf + 1e-12),
        _entropy(fhr), _entropy(fhr, sample=True), _hurst_rs(fhr), reserve,
    ], dtype=float)
    return np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)


def extract_ctg_feature_matrix(chunks, **kwargs):
    """Extract the named feature vector for every chunk in a landmark panel."""
    chunks = np.asarray(chunks)
    if chunks.ndim != 3: raise ValueError("Expected [patient, time, feature] chunks.")
    return np.vstack([extract_ctg_features(chunk, **kwargs) for chunk in chunks])
