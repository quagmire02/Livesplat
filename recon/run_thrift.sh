#!/usr/bin/env bash
# Stage 0: run ThriftSplat in its own venv, then export cameras for LiveSplat.
# Usage: recon/run_thrift.sh <video> <run_name> [extra thriftsplat args...]
#   e.g. recon/run_thrift.sh ~/livesplat_data/videos/table1.mp4 table1 --stage cloud
set -euo pipefail
THRIFT=${THRIFT_DIR:-$HOME/projects/thriftsplat}
PY=${THRIFT_PY:-$THRIFT/.venv/bin/python}
DATA=${LIVESPLAT_DATA:-$HOME/livesplat_data}
export HF_HUB_DISABLE_XET=1 CUDA_HOME=/usr/local/cuda-12.4
export PATH="$CUDA_HOME/bin:$PATH" PYTHONPATH="$THRIFT${PYTHONPATH:+:$PYTHONPATH}"
VIDEO=$1; NAME=$2; shift 2
OUT="$DATA/runs/$NAME/thrift"
mkdir -p "$(dirname "$OUT")"
cd "$THRIFT"
"$PY" -m thriftsplat "$VIDEO" -o "$OUT" "$@"
