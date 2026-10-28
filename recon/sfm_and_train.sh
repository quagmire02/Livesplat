#!/usr/bin/env bash
# Stage 0b (orbits): global SfM poses, splat training, export for LiveSplat.
# Usage: recon/sfm_and_train.sh <run_name> [iters] [width] [cap]
# Needs stage 1 done: recon/run_thrift.sh <video> <run_name> --stage cloud
set -euo pipefail
NAME=$1; ITERS=${2:-15000}; WIDTH=${3:-1554}; CAP=${4:-1800000}   # 1554 = 3*518, aspect exact
THRIFT=${THRIFT_DIR:-$HOME/projects/thriftsplat}
TPY=${THRIFT_PY:-$THRIFT/.venv/bin/python}
LPY=${LIVESPLAT_PY:-$HOME/venvs/livesplat/bin/python}
DATA=${LIVESPLAT_DATA:-$HOME/livesplat_data}
RUN="$DATA/runs/$NAME"
export HF_HUB_DISABLE_XET=1 CUDA_HOME=/usr/local/cuda-12.4
export PATH="$(dirname "$TPY"):$CUDA_HOME/bin:$PATH"   # ninja for the gsplat JIT lives in the venv
HERE=$(cd "$(dirname "$0")/.." && pwd)

# 1. Global SfM (CPU, pycolmap lives in the LiveSplat venv). Writes $RUN/thrift_sfm/scene.
[ -f "$RUN/thrift_sfm/scene/scene.json" ] || "$LPY" "$THRIFT/tools/sfm_poses.py" "$RUN/thrift" --out "$RUN/thrift_sfm" --threads 4

# 2. Train the splat on the SfM scene (ThriftSplat venv, GPU).
cd "$THRIFT"
PYTHONPATH="$THRIFT" "$TPY" -m thriftsplat.train_gsplat "$RUN/thrift_sfm/scene" \
  -o "$RUN/thrift_sfm/splat" --iters "$ITERS" --width "$WIDTH" --cap "$CAP"
ln -sf splat/splat.ply "$RUN/thrift_sfm/splat.ply"

# 3. Export cameras + keyframes for LiveSplat.
PYTHONPATH="$THRIFT" "$TPY" "$HERE/recon/export_thrift.py" "$RUN/thrift_sfm" --width 1036
echo "DONE $RUN/thrift_sfm"
