"""Spike 4: SAM 2.1 hiera-small video tracking on exported keyframes. Reports VRAM and time.

Run (WSL): python -m spikes.sam2_local <run_dir> [--n 40] [--box x0,y0,x1,y1]
Default box = centre third of frame 0.
"""
import argparse
import os
import shutil
import tempfile
import time
from pathlib import Path

import numpy as np
import torch

from backends.cache import data_dir
from scene_io.thrift import load_cameras


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--box", default=None)
    a = ap.parse_args()

    from sam2.build_sam import build_sam2_video_predictor
    ckpt = data_dir() / "models" / "sam2.1_hiera_small.pt"
    pred = build_sam2_video_predictor("configs/sam2.1/sam2.1_hiera_s.yaml", str(ckpt), device="cuda")

    cams = load_cameras(a.run_dir)
    frames = cams["frames"][:a.n]
    tmp = Path(tempfile.mkdtemp())
    for i, f in enumerate(frames):                 # SAM 2 wants a folder of 00000.jpg ...
        shutil.copy(f["image"], tmp / f"{i:05d}.jpg")
    W, H = cams["width"], cams["height"]
    box = (np.array([float(v) for v in a.box.split(",")]) if a.box
           else np.array([W / 3, H / 3, 2 * W / 3, 2 * H / 3]))

    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        st = pred.init_state(video_path=str(tmp), offload_video_to_cpu=True, offload_state_to_cpu=True)
        pred.add_new_points_or_box(st, frame_idx=0, obj_id=1, box=box)
        areas = []
        for fi, ids, logits in pred.propagate_in_video(st):
            areas.append(int((logits[0] > 0).sum()))
    dt = time.time() - t0
    print(f"{len(frames)} frames {W}x{H}: {dt:.1f}s ({dt / len(frames):.2f}s/frame), "
          f"peak VRAM {torch.cuda.max_memory_allocated() / 2**30:.2f} GB")
    print("mask area per frame (px):", areas[:10], "...")
    shutil.rmtree(tmp)


if __name__ == "__main__":
    main()
