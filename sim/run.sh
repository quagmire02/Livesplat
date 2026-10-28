#!/usr/bin/env bash
# Run a sim module with the env the gsplat JIT needs. Usage: sim/run.sh <module> [args...]
#   e.g. sim/run.sh sim.check_render ~/livesplat_data/runs/table1/sim
set -euo pipefail
V=${LIVESPLAT_VENV:-$HOME/venvs/livesplat}
export CUDA_HOME=/usr/local/cuda-12.4 PATH="$V/bin:/usr/local/cuda-12.4/bin:$PATH" HF_HUB_DISABLE_XET=1
cd "$(dirname "$0")/.."
exec "$V/bin/python" -m "$@"
