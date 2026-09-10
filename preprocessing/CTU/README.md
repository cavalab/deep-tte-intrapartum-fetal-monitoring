# CTU-UHB preprocessing

The default workflow keeps source and generated data separate:

```text
data/CTU/raw/        PhysioNet WFDB files (.dat/.hea)
data/CTU/converted/  optional CSV files made by the exploration notebook
data/                CTU HDF5 tracing stores and label parquet sidecars
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
```

Train/test labels are written as parquet sidecars.

For example, the multi-rate command also writes:

```text
data/MFM_CTU_tracings.sample_limit-all.sample_rate-100.horizon-0.max_gap_hours-18.window-full.min_length-30.tail_length-2880.missingness-none.h5
```

The generated HDF5 file contains every retained CTU record. Pass the HDF5 path
for traces and the generated parquet path for labels when evaluating models.

The CTU prefix keeps these files separate from the private-cohort artifacts.
This repository intentionally keeps only this maintained HDF5 time-to-event
workflow.

To evaluate a checkpoint on CTU-UHB, use
`postprocessing/postprocess_ctu_checkpoints.py` as described in the project
README.
