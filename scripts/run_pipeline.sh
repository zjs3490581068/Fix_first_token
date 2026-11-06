#!/usr/bin/env bash
set -uo pipefail
if [[ "${1:-}" == "--help" ]]; then
    printf 'Usage: bash scripts/run_pipeline.sh RUN_DIR [CONFIG] [PYTHON]\n'
    exit 0
fi
repo=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
cd "$repo" || exit 1
run_dir=$1
config=${2:-$repo/configs/main.json}
python_bin=${3:-python}
mkdir -p "$run_dir/logs"
"$python_bin" -u -m first_token run --config "$config" --run-dir "$run_dir" >> "$run_dir/logs/coordinator.log" 2>&1
result=$?
printf '%s\n' "$result" > "$run_dir/exit_code.txt"
exit "$result"
