#!/bin/bash

# # Preview commands only
# uv run --active python submit_jobs.py run_config \
#   --config experiments/official_experiment.yml \
#   --print_only true

# # Generate scripts without submitting
# uv run --active python submit_jobs.py run_config \
#   --config experiments/official_experiment.yml \
#   --submit false

# # Generate and submit jobs
uv run --active python submit_jobs.py run_config \
  --config experiments/official_experiment.yml \
  --submit true