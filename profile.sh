#!/usr/bin/env bash
set -e

project_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
report_dir="$project_dir/src/kernels/ncu-report"

mkdir -p "$report_dir"

report="$1"
shift

ncu --profile-from-start off --set full -f \
    -o "$report_dir/$report" "$@"