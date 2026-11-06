#!/usr/bin/env bash
set -euo pipefail
if [[ "${1:-}" == "--help" ]]; then
    printf 'Usage: bash scripts/launch_tmux.sh [RUN_ID] [CONFIG]\nUses Python from the active environment; tmux must be installed.\n'
    exit 0
fi
repo=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
python_bin=$(command -v python)
run_id=${1:-$(date -u +%Y%m%d_%H%M%S)}
config=${2:-$repo/configs/main.json}
if [[ ! "$run_id" =~ ^[A-Za-z0-9_-]+$ ]]; then
    printf 'run_id must contain only letters, numbers, underscores or hyphens\n' >&2
    exit 2
fi
cd "$repo"
config=$(realpath -- "$config")
output_root=$(python -c 'import sys; from first_token.core import config; print(config(sys.argv[1])["output_root"])' "$config")
run_dir="$output_root/$run_id"
socket="$output_root/.runtime/tmux.sock"
session="fix_ft_$run_id"
if tmux -S "$socket" has-session -t "$session" 2>/dev/null; then
    printf 'Session already exists: %s\n' "$session" >&2
    exit 2
fi
mkdir -p "$run_dir/logs" "$(dirname -- "$socket")"
printf -v command 'bash --noprofile --norc %q %q %q %q' "$repo/scripts/run_pipeline.sh" "$run_dir" "$config" "$python_bin"
printf 'command=%s\nworkdir=%s\npython=%s\nrun_dir=%s\n' "$command" "$repo" "$python_bin" "$run_dir"
python -c 'import sys; from first_token.core import config; print("GPUs:", config(sys.argv[1])["gpus"])' "$config"
tmux -S "$socket" -f "$repo/configs/tmux.conf" new-session -d -s "$session" -c "$repo" "$command"
printf 'session=%s\nattach: tmux -S %s attach -t %s\n' "$session" "$socket" "$session"
