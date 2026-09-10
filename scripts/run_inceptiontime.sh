#!/usr/bin/env bash
# Train the paper's InceptionTime marked time-to-event model.
#
# The data files are private. Override the variables below when they are stored
# elsewhere, for example: GPU=1 EPOCHS=20 scripts/run_inceptiontime.sh
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_dir="$(cd "${script_dir}/.." && pwd)"
cd "${repo_dir}"

data_dir="${DATA_DIR:-data}"
trace_file="${TRACE_FILE:-${data_dir}/MFM_tracings.sample_limit-all.sample_rate-025.horizon-0.max_gap_hours-18.window-full.min_length-30.tail_length-2880.missingness-none.h5}"
train_labels="${TRAIN_LABELS:-${data_dir}/MFM_labs_train.parquet}"
test_labels="${TEST_LABELS:-${data_dir}/MFM_labs_test.parquet}"
splits_file="${SPLITS_FILE:-${data_dir}/train_val_test_splits_extended.csv}"
output_dir="${OUTPUT_DIR:-pretrained/inceptiontime}"

for required_file in "${trace_file}" "${train_labels}" "${test_labels}" "${splits_file}"; do
    [[ -f "${required_file}" ]] || {
        echo "Missing required file: ${required_file}" >&2
        exit 1
    }
done

mkdir -p "${output_dir}"

uv run --active python train_time_to_event_models.py run \
    --trace_file "${trace_file}" \
    --label_trainfile "${train_labels}" \
    --label_testfile "${test_labels}" \
    --splits_file "${splits_file}" \
    --savedir "${output_dir}" \
    --mode marked \
    --ml Inception_classifier \
    --gpu "${GPU:-0}" \
    --epochs "${EPOCHS:-100}" \
    --batch_size "${BATCH_SIZE:-2048}" \
    --chunk_window_size "${CHUNK_WINDOW_SIZE:-900}" \
    --random_state "${SEED:-7196}" \
    --learning_rate "${LEARNING_RATE:-0.01}" \
    --lr_schedule cosine \
    --lr_warmup_fraction 0.05 \
    --lr_min_fraction 0.01 \
    --early_stopping_patience "${EARLY_STOPPING_PATIENCE:-30}" \
    --fit_kwargs '{"n_filters":32,"depth":3,"bottleneck_channels":32,"kernel_sizes":[39,19,9]}'
