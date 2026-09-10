# CTU-UHB preprocessing

The default workflow keeps source and generated data separate:

```text
data/CTU/raw/        PhysioNet WFDB files (.dat/.hea)
data/CTU/converted/  optional CSV files made by the exploration notebook
data/                CTU HDF5 tracing and label parquet files
```

From the repository root:

```bash
bash preprocessing/CTU/get_data.sh
```

Preprocess the downloaded WFDB records directly:

```bash
python preprocessing/CTU/preprocess_ctu.py
```

To produce more than one sampling-rate variant in one invocation:

```bash
python preprocessing/CTU/preprocess_ctu.py --sample_rates 0.25,1.0
```

This writes, by default:

```text
data/MFM_CTU_tracings.sample_limit-all.sample_rate-025.horizon-0.max_gap_hours-18.window-full.min_length-30.tail_length-2880.missingness-none.h5
data/MFM_CTU_labs_train.parquet
data/MFM_CTU_labs_test.parquet
data/MFM_CTU_labs_test_all.parquet
```

For example, the multi-rate command also writes:

```text
data/MFM_CTU_tracings.sample_limit-all.sample_rate-100.horizon-0.max_gap_hours-18.window-full.min_length-30.tail_length-2880.missingness-none.h5
```

`MFM_CTU_labs_test_all.parquet` contains every retained CTU record and is the
label file to pass as `label_testfile` when evaluating an external checkpoint
with all CTU data treated as the test partition. The ordinary train/test files
remain available for CTU-only split experiments.

The CTU prefix keeps these files separate from the private-cohort artifacts.
This repository intentionally keeps only this maintained HDF5 time-to-event
workflow.

To evaluate a checkpoint on CTU-UHB, use
`postprocessing/postprocess_ctu_checkpoints.py` as described in the project
README.
