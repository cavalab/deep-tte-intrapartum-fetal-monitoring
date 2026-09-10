"""Create project-compatible CTU-UHB time-to-event HDF5 artifacts.

The public CTU-UHB source is distributed as WFDB records.  This script reads
those records directly and writes the same variable-length HDF5 tracing index
and writes train/test outcome splits as parquet sidecars. Tracing data itself
is always stored in HDF5.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from .paths import CTU_WFDB_DIR, PROCESSED_DATA_DIR
except ImportError:  # Supports ``python preprocessing/CTU/preprocess_ctu.py``.
    from paths import CTU_WFDB_DIR, PROCESSED_DATA_DIR


CTU_PREFIX = "MFM_CTU"
RAW_SAMPLE_RATE = 4.0
PH_THRESHOLDS = (7.05, 7.10, 7.15, 7.20)
BE_THRESHOLDS = (-14.0, -10.0, -6.0)


def _missingness_enabled(missingness):
    if missingness is None:
        return False
    if isinstance(missingness, str) and missingness.strip().lower() in {
        "", "none", "null"
    }:
        return False
    return float(missingness) < 1


def _missingness_tag(missingness):
    return (
        f"missingness-{int(float(missingness) * 100)}"
        if _missingness_enabled(missingness)
        else "missingness-none"
    )


def _trace_stem(
    sample_limit,
    sample_rate,
    horizon,
    max_gap_hours,
    window,
    min_length,
    tail_length,
    missingness,
):
    limit_tag = f"sample_limit-{sample_limit}" if sample_limit > 0 else "sample_limit-all"
    window_tag = "window-full" if window == 0 else f"window-{window:g}"
    parts = [
        f"{CTU_PREFIX}_tracings",
        limit_tag,
        f"sample_rate-{sample_rate:.2f}".replace(".", ""),
        f"horizon-{horizon:g}",
        f"max_gap_hours-{max_gap_hours:g}",
        window_tag,
    ]
    if min_length > 0:
        parts.append(f"min_length-{min_length:g}")
    if tail_length > 0:
        parts.append(f"tail_length-{tail_length:g}")
    parts.append(_missingness_tag(missingness))
    return ".".join(parts)


def _record_stems(data_dir):
    """Return one WFDB record stem per CTU PID, preserving numeric order."""
    root = Path(data_dir)
    headers = sorted(root.rglob("*.hea"))
    if headers:
        return sorted({header.with_suffix("") for header in headers}, key=lambda p: p.name)
    csv_files = sorted(path for path in root.glob("*.csv") if path.name != "labels.csv")
    if csv_files:
        return csv_files
    raise FileNotFoundError(
        f"No CTU WFDB headers or converted CSV files found below {root}. "
        "Run get_data.sh first, or pass --data_dir to the CTU record directory."
    )


def _load_record(record_path):
    """Load one CTU WFDB record or an already converted CSV record."""
    record_path = Path(record_path)
    if record_path.suffix.lower() == ".csv":
        frame = pd.read_csv(record_path)
        if "time" in frame.columns:
            frame["time"] = pd.to_datetime(frame["time"])
            frame = frame.set_index("time")
        else:
            frame.index = pd.date_range(
                "2023-12-20", periods=len(frame), freq="250ms"
            )
        frame = frame.rename(columns={"FHR": "fecg", "UC": "toco"})
        return record_path.stem, frame[["fecg", "toco"]].apply(pd.to_numeric, errors="coerce")

    try:
        import wfdb
    except ImportError as exc:
        raise ImportError(
            "wfdb is required to read CTU-UHB .dat/.hea records. "
            "Install the project dependencies before running this script."
        ) from exc

    # wfdb 4.x returns physical-valued signals by default and removed the
    # older ``physical=`` keyword from ``rdsamp``.
    signal, fields = wfdb.rdsamp(str(record_path))
    names = [str(name).strip().lower() for name in fields.get("sig_name", [])]
    try:
        fecg_index = next(i for i, name in enumerate(names) if name in {"fhr", "fecg"})
        toco_index = next(i for i, name in enumerate(names) if name in {"uc", "toco", "ua"})
    except StopIteration as exc:
        raise ValueError(
            f"Could not identify FHR/UC channels in {record_path}; "
            f"available channels: {fields.get('sig_name', [])}"
        ) from exc

    sample_rate = float(fields.get("fs", RAW_SAMPLE_RATE))
    if not np.isclose(sample_rate, RAW_SAMPLE_RATE):
        raise ValueError(
            f"Expected CTU records at {RAW_SAMPLE_RATE:g} Hz, found {sample_rate:g} Hz "
            f"in {record_path}"
        )
    index = pd.date_range("2023-12-20", periods=len(signal), freq="250ms")
    frame = pd.DataFrame(
        {
            "fecg": signal[:, fecg_index],
            "toco": signal[:, toco_index],
        },
        index=index,
    )
    return record_path.stem, frame.apply(pd.to_numeric, errors="coerce")


def _smooth_observed(series, window_len):
    """Smooth finite segments without filling across missing signal gaps."""
    values = series.to_numpy(dtype=float, copy=True)
    finite = np.isfinite(values)
    padded = np.r_[False, finite, False]
    starts = np.flatnonzero(np.diff(padded.astype(np.int8)) == 1)
    ends = np.flatnonzero(np.diff(padded.astype(np.int8)) == -1)
    for start, end in zip(starts, ends):
        if end - start >= window_len:
            segment = values[start:end]
            reflected = np.r_[segment[window_len - 1:0:-1], segment, segment[-2:-window_len - 1:-1]]
            weights = np.hanning(window_len)
            values[start:end] = np.convolve(weights / weights.sum(), reflected, mode="valid")[
                round(window_len / 2 - 1):-round(window_len / 2)
            ]
    return pd.Series(values, index=series.index, name=series.name)


def process_record(
    record_path,
    *,
    sample_rate,
    horizon,
    window,
    min_length,
    tail_length,
    missingness,
):
    """Return one normalized tracing record or a reason for exclusion."""
    pid, frame = _load_record(record_path)
    frame = frame[~frame.index.duplicated(keep="first")].sort_index()

    # Apply the same physiological cleaning bounds used for the private cohort.
    frame.loc[(frame["fecg"] < 30) | (frame["fecg"] > 220), "fecg"] = np.nan
    frame.loc[frame["toco"] >= 128, "toco"] = np.nan

    if horizon > 0:
        cutoff = frame.index.max() - pd.Timedelta(minutes=horizon)
        frame = frame.loc[frame.index < cutoff]
    if frame.empty:
        return "Short / no sequence"

    if window > 0:
        start = frame.index.max() - pd.Timedelta(minutes=window)
        frame = frame.loc[frame.index >= start]
    elif tail_length > 0:
        start = frame.index.max() - pd.Timedelta(minutes=tail_length)
        frame = frame.loc[frame.index > start]

    duration_minutes = (frame.index.max() - frame.index.min()).total_seconds() / 60
    if window == 0 and duration_minutes < min_length:
        return "Tracing too short"

    frame = frame.resample("250ms").asfreq()
    raw_missingness = float(frame[["fecg", "toco"]].isna().to_numpy().mean())
    if _missingness_enabled(missingness) and raw_missingness > float(missingness):
        return "Too much missingness"

    stride = max(1, int(round(RAW_SAMPLE_RATE / sample_rate)))
    frame = frame.apply(lambda column: _smooth_observed(column, stride + 1))
    frame = frame.resample(pd.Timedelta(seconds=1 / sample_rate), label="right").mean()
    down_missingness = float(frame[["fecg", "toco"]].isna().to_numpy().mean())
    if _missingness_enabled(missingness) and down_missingness > float(missingness):
        return "Too much missingness"
    if frame.empty:
        return "Short / no sequence"

    return {
        "PID": str(pid),
        "fecg": frame["fecg"].to_numpy(dtype=np.float32),
        "toco": frame["toco"].to_numpy(dtype=np.float32),
        "last_time": frame.index.max(),
    }


def _write_hdf5(path, records):
    try:
        import h5py
    except ImportError as exc:
        raise ImportError("h5py is required to write CTU HDF5 tracings.") from exc

    string_dtype = h5py.string_dtype(encoding="utf-8")
    with h5py.File(path, "w") as h5f:
        tracings = h5f.create_group("tracings")
        index = h5f.create_group("index")
        index.create_dataset("pid", data=np.asarray([r["PID"] for r in records], dtype=object), dtype=string_dtype)
        index.create_dataset(
            "last_time_ns",
            data=np.asarray([pd.Timestamp(r["last_time"]).value for r in records], dtype=np.int64),
        )
        index.create_dataset(
            "length", data=np.asarray([len(r["fecg"]) for r in records], dtype=np.int32)
        )
        for record in records:
            group = tracings.create_group(str(record["PID"]))
            group.create_dataset("fecg", data=record["fecg"], compression="gzip")
            group.create_dataset("toco", data=record["toco"], compression="gzip")
            group.attrs["last_time_ns"] = pd.Timestamp(record["last_time"]).value
            group.attrs["length"] = len(record["fecg"])


def _header_labels(record_path):
    """Extract pH and base-excess values from a CTU WFDB header."""
    if Path(record_path).suffix.lower() == ".csv":
        return {}
    import wfdb

    comments = wfdb.rdheader(str(record_path)).comments
    values = {}
    for comment in comments:
        match = re.match(r"\s*(pH|BE)\s+([-+]?\d+(?:\.\d+)?)\s*$", comment, re.I)
        if match:
            values[match.group(1).lower()] = float(match.group(2))
    return values


def _labels_from_records(records, record_paths):
    by_pid = {str(path.stem): path for path in record_paths}
    rows = []
    for record in records:
        values = _header_labels(by_pid[record["PID"]])
        row = {
            "PID": record["PID"],
            "pH Cord": values.get("ph", np.nan),
            "Base Excess Cord": values.get("be", np.nan),
            "last_time": record["last_time"],
        }
        rows.append(row)
    labels = pd.DataFrame(rows)
    for threshold in PH_THRESHOLDS:
        labels[f"pH Cord < {threshold:g}"] = labels["pH Cord"] < threshold
    for threshold in BE_THRESHOLDS:
        labels[f"Base Excess Cord < {threshold:g}"] = labels["Base Excess Cord"] < threshold
    return labels


def _split_labels(labels, split, seed):
    if not 0 < float(split) < 1:
        raise ValueError("split must be between 0 and 1")
    if len(labels) <= 1:
        return labels.reset_index(drop=True), labels.iloc[0:0].copy().reset_index(drop=True)
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(labels))
    n_train = min(max(1, int(round(len(labels) * float(split)))), len(labels) - 1)
    return labels.iloc[order[:n_train]].reset_index(drop=True), labels.iloc[order[n_train:]].reset_index(drop=True)


def _run_single(
    data_dir=str(CTU_WFDB_DIR),
    save_dir=str(PROCESSED_DATA_DIR),
    sample_limit=0,
    sample_rate=0.25,
    horizon=0,
    max_gap_hours=18,
    window=0,
    min_length=30,
    tail_length=2880,
    missingness="none",
    split=0.75,
    seed=42,
):
    """Preprocess CTU at one sample rate and write label artifacts."""
    if sample_rate <= 0 or RAW_SAMPLE_RATE / sample_rate < 1:
        raise ValueError("sample_rate must be positive and no greater than 4 Hz")
    if window > 0 and tail_length > 0:
        raise ValueError("tail_length is only valid when window=0")

    output_dir = Path(save_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    record_paths = _record_stems(data_dir)
    if sample_limit > 0:
        record_paths = record_paths[: int(sample_limit)]

    records = []
    failures = {}
    for record_path in record_paths:
        result = process_record(
            record_path,
            sample_rate=sample_rate,
            horizon=horizon,
            window=window,
            min_length=min_length,
            tail_length=tail_length,
            missingness=missingness,
        )
        if isinstance(result, dict):
            records.append(result)
        else:
            failures[result] = failures.get(result, 0) + 1

    if not records:
        raise RuntimeError(f"No CTU records passed preprocessing. Failures: {failures}")

    trace_stem = _trace_stem(
        sample_limit, sample_rate, horizon, max_gap_hours, window, min_length,
        tail_length, missingness,
    )
    if window != 0:
        raise ValueError("HDF5-only preprocessing requires window=0.")
    trace_path = output_dir / f"{trace_stem}.h5"

    labels = _labels_from_records(records, record_paths)
    labs_train, labs_test = _split_labels(labels, split, seed)
    train_path = output_dir / f"{CTU_PREFIX}_labs_train.parquet"
    test_path = output_dir / f"{CTU_PREFIX}_labs_test.parquet"
    test_all_path = output_dir / f"{CTU_PREFIX}_labs_test_all.parquet"
    labs_train.to_parquet(train_path, index=False)
    labs_test.to_parquet(test_path, index=False)
    labels.to_parquet(test_all_path, index=False)
    _write_hdf5(trace_path, records)

    print(f"Wrote {len(records)} CTU tracings to {trace_path}")
    print(f"Wrote {len(labels)} outcome rows to parquet sidecars")
    print(f"Wrote train/test label splits to {train_path} and {test_path}")
    if failures:
        print(f"Excluded records: {failures}")
    return {
        "trace_file": str(trace_path),
        "label_file": str(test_all_path),
        "label_trainfile": str(train_path),
        "label_testfile": str(test_path),
        "label_test_allfile": str(test_all_path),
        "n_tracings": len(records),
        "n_labelled": int(labels["pH Cord"].notna().sum()),
        "failures": failures,
    }


def _parse_sample_rates(sample_rates):
    """Accept Fire-friendly comma-separated or JSON-like sample-rate input."""
    if sample_rates in (None, "", []):
        return None
    if isinstance(sample_rates, str):
        text = sample_rates.strip().strip("[]")
        values = [part.strip() for part in text.split(",") if part.strip()]
    elif isinstance(sample_rates, (list, tuple, np.ndarray)):
        values = sample_rates
    else:
        values = [sample_rates]
    rates = []
    for value in values:
        rate = float(value)
        if rate not in rates:
            rates.append(rate)
    if not rates:
        raise ValueError("sample_rates must contain at least one rate")
    return rates


def run(
    data_dir=str(CTU_WFDB_DIR),
    save_dir=str(PROCESSED_DATA_DIR),
    sample_limit=0,
    sample_rate=0.25,
    sample_rates=None,
    horizon=0,
    max_gap_hours=18,
    window=0,
    min_length=30,
    tail_length=2880,
    missingness="none",
    split=0.75,
    seed=42,
):
    """Preprocess CTU at one rate or multiple rates in a single invocation.

    ``sample_rate`` is used when ``sample_rates`` is not supplied. Specify
    ``sample_rates`` (for example, ``0.25,1.0``) to generate multiple stores.
    """
    rates = _parse_sample_rates(sample_rates)
    if rates is None:
        rates = [float(sample_rate)]
    results = []
    for rate in rates:
        results.append(_run_single(
            data_dir=data_dir, save_dir=save_dir, sample_limit=sample_limit,
            sample_rate=rate, horizon=horizon, max_gap_hours=max_gap_hours,
            window=window, min_length=min_length, tail_length=tail_length,
            missingness=missingness, split=split, seed=seed,
        ))
    if len(results) == 1:
        return results[0]
    return {"sample_rates": rates, "runs": results}


if __name__ == "__main__":
    import fire

    fire.Fire(run)
