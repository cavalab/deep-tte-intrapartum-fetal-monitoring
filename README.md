# Deep time-to-event intrapartum fetal monitoring

This repository contains the code used for the paper's deep time-to-event models for intrapartum fetal monitoring.
It trains an InceptionTime encoder with marked and competing-risk DeepHit objectives, evaluates fixed clinical landmarks, and supports external evaluation on the public CTU-UHB cohort.

The private study cohort, trained checkpoints, and final selection artifacts
are deliberately not included. See the README in `data/`, `pretrained/`, and
`final_model_selection/` for the expected private contents.

## Setup

Dependencies are defined in `pyproject.toml`.

```bash
uv sync
```

For notebooks, install the optional notebook dependencies:

```bash
uv sync --extra notebook
```

## Data

Training requires the following private artifacts in `data/` (or paths passed
on the command line):

- an HDF5 tracing store, typically named
  `MFM_tracings.sample_limit-all.sample_rate-025...h5`;
- `MFM_labs_train.parquet` and `MFM_labs_test.parquet`;
- `train_val_test_splits_extended.csv` with `PID` and `fold` columns.

The public CTU-UHB data can be downloaded and converted into compatible files:

```bash
bash preprocessing/CTU/get_data.sh
python preprocessing/CTU/preprocess_ctu.py
```

The preprocessing command writes CTU artifacts under `data/`. It requires the
`wfdb` package; install it in the active environment if needed.

## Train InceptionTime

Run the reproducible default training configuration:

```bash
scripts/run_inceptiontime.sh
```

It writes checkpoints, training configuration sidecars, and evaluation files
to `pretrained/inceptiontime/`. Override private file locations and common
settings with environment variables, for example:

```bash
DATA_DIR=/path/to/data GPU=1 EPOCHS=20 scripts/run_inceptiontime.sh
```

## Run the CTG baselines

The paper also reports two 1 Hz baselines. Both scripts fit on the private
cohort and then apply the saved model bundle to CTU-UHB without refitting:

```bash
scripts/run_ctg_fri_cox_1hz.sh
scripts/run_xgboost_figo_ctg_cox_1hz.sh
```

These expect the 1 Hz private trace store and labels in `data/`. Override
`TRACE_FILE`, `RESULTS_ROOT`, `CTU_TRACE_FILE`, and the other path variables in
the scripts when your private files live elsewhere. The FRI baseline requires
`scikit-survival`; the FIGO-feature baseline additionally uses `xgboost`.

For a full architecture and sampling sweep, preview the generated jobs first:

```bash
uv run --active python submit_jobs.py run_config \
  --config experiments/official_experiment.yml \
  --print_only true
```

Set scheduler-specific `account` and `partition` values in a local copy of the
experiment configuration before creating or submitting batch scripts. To run a
sweep as local processes, pass `--local true`.

## Evaluate a checkpoint on CTU-UHB

Use the checkpoint replay script after CTU preprocessing:

```bash
uv run --active python -m postprocessing.postprocess_ctu_checkpoints \
  --checkpoint-dir pretrained/inceptiontime/checkpoints \
  --trace-file data/MFM_CTU_tracings.sample_limit-all.sample_rate-025.horizon-0.max_gap_hours-18.window-full.min_length-30.tail_length-2880.missingness-none.h5 \
  --train-labels data/MFM_CTU_labs_train.parquet \
  --test-labels data/MFM_CTU_labs_test.parquet \
  --output-dir results/ctu_external
```

The external-validation notebook
`notebooks/external_model_auroc_comparison.ipynb` runs this checkpoint
evaluation for the private final-selection archive and builds the paper table.
For baseline bundles, use the companion command-line evaluator:

```bash
uv run --active python postprocessing/postprocess_ctu_baselines.py \
  --model-file results_ctg_baselines_1hz/ctg_only_fri_cox/ctg_fri_cox_model.pkl \
  --model-file results_ctg_baselines_1hz/xgboost_figo_ctg_cox/xgboost_ctg_model.pkl \
  --trace-file data/MFM_CTU_tracings.sample_limit-all.sample_rate-100.horizon-0.max_gap_hours-18.window-full.min_length-30.tail_length-2880.missingness-none.h5 \
  --test-labels data/MFM_CTU_labs_test_all.parquet \
  --output-dir results_final_model_selection_external_ctu/baselines
```

## Repository layout

- `train_time_to_event_models.py`: training and evaluation command-line entry
  point.
- `methods/`: InceptionTime and survival-model implementations.
- `time_to_event_data.py`: streaming HDF5 chunks and labels.
- `time_to_event_evaluation.py`: landmark and bootstrap evaluation.
- `preprocessing/CTU/`: public CTU-UHB download and conversion.
- `postprocessing/`: checkpoint replay and baseline external-evaluation tools.
- `notebooks/external_model_auroc_comparison.ipynb`: CTU external-validation
  evaluation and AUROC table generation.
- `experiments/`: reproducible sweep configuration.
