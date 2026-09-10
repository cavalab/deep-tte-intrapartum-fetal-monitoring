#!/usr/bin/env bash
# Run the primary CTG-only FRI one-variable Cox baseline at 1 Hz.
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_dir="$(cd "${script_dir}/.." && pwd)"
cd "${repo_dir}"

trace_file="${TRACE_FILE:-data/MFM_tracings.sample_limit-all.sample_rate-100.horizon-0.max_gap_hours-18.window-full.min_length-30.tail_length-2880.missingness-none.h5}"
results_root="${RESULTS_ROOT:-results_ctg_baselines_1hz}"
landmark_cache_dir="${LANDMARK_CACHE_DIR:-cache/baseline_landmarks}"
ctu_trace_file="${CTU_TRACE_FILE:-data/MFM_CTU_tracings.sample_limit-all.sample_rate-100.horizon-0.max_gap_hours-18.window-full.min_length-30.tail_length-2880.missingness-none.h5}"
ctu_test_labels="${CTU_TEST_LABELS:-data/MFM_CTU_labs_test_all.parquet}"
external_results_root="${EXTERNAL_RESULTS_ROOT:-results_final_model_selection_external_ctu}"
ctu_landmark_cache_dir="${CTU_LANDMARK_CACHE_DIR:-cache/baseline_landmarks_ctu}"
model_file="${results_root}/ctg_only_fri_cox/ctg_fri_cox_model.pkl"

uv run --active python run_ctg_fri_baseline.py run \
  --trace_file "${trace_file}" \
  --label_trainfile data/MFM_labs_train.parquet \
  --label_testfile data/MFM_labs_test.parquet \
  --splits_file data/train_val_test_splits_extended.csv \
  --chunk_window_size 3600 \
  --lab_order_delay 30 \
  --bootstrap_samples 100 \
  --bootstrap_confidence_level 0.95 \
  --random_state 7196 \
  --landmark_cache_dir "${landmark_cache_dir}" \
  --evaluate_validation false \
  --require_acceleration true \
  --savedir "${results_root}/ctg_only_fri_cox"

# The saved bundle is applied to CTU without any refitting.
uv run --active python run_ctg_fri_baseline.py evaluate \
  --model_file "${model_file}" \
  --trace_file "${ctu_trace_file}" \
  --test_label_file "${ctu_test_labels}" \
  --lab_order_delay 0 \
  --bootstrap_samples 100 \
  --bootstrap_confidence_level 0.95 \
  --random_state 7196 \
  --landmark_cache_dir "${ctu_landmark_cache_dir}" \
  --savedir "${external_results_root}/ctg_only_fri_cox"
