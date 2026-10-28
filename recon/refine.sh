#!/usr/bin/env bash
# Object-focused refinement in the ThriftSplat venv. Usage: recon/refine.sh <run_dir> [refine_objects.py args...]
set -euo pipefail
THRIFT=${THRIFT_DIR:-$HOME/projects/thriftsplat}
TPY=${THRIFT_PY:-$THRIFT/.venv/bin/python}
export HF_HUB_DISABLE_XET=1 CUDA_HOME=/usr/local/cuda-12.4
export PATH="$(dirname "$TPY"):$CUDA_HOME/bin:$PATH" PYTHONPATH="$THRIFT"
HERE=$(cd "$(dirname "$0")/.." && pwd)
cd "$THRIFT"
exec "$TPY" "$HERE/recon/refine_objects.py" "$@"
