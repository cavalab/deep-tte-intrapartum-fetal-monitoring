#!/usr/bin/env bash

set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "${script_dir}/../.." && pwd)"
raw_dir="${repo_root}/data/CTU/raw"

mkdir -p "${raw_dir}"

# Keep PhysioNet's directory structure under a clearly named raw-data folder.
wget -r -N -c -np --directory-prefix="${raw_dir}" \
    https://physionet.org/files/ctu-uhb-ctgdb/1.0.0/
