"""Export a ThriftSplat run to the LiveSplat camera format.

Runs in the ThriftSplat venv (it imports thriftsplat, it does not copy it):
    $THRIFT_PY recon/export_thrift.py <run_dir> [--width 1036]

Writes <run_dir>/export/cameras.json and <run_dir>/export/keyframes/*.jpg.
Images go through ThriftSplat's own loader, so K matches the pixels exactly.
c2w = inverse(pose_delta @ w2c), the poses the splat was trained with.
"""
import argparse
import json
import os
import sys

import numpy as np
import torch
from PIL import Image


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("--width", type=int, default=1036, help="multiple of 518 keeps aspect exact")
    a = ap.parse_args()

    from thriftsplat.train_gsplat import load_views, pose_delta_to_mat

    scene_dir = os.path.join(a.run_dir, "scene")
    imgs, viewmats, Ks, W, H, names, _ = load_views(scene_dir, a.width, "cpu")
    pd_path = os.path.join(a.run_dir, "splat", "pose_delta.npy")
    if os.path.exists(pd_path):
        pd = torch.tensor(np.load(pd_path), dtype=torch.float32)
        assert len(pd) == len(names), f"pose_delta {len(pd)} != frames {len(names)}"
        viewmats = pose_delta_to_mat(pd) @ viewmats
    else:
        print("note: no pose_delta.npy, using stage-1 poses", file=sys.stderr)

    out = os.path.join(a.run_dir, "export")
    os.makedirs(os.path.join(out, "keyframes"), exist_ok=True)
    frames = []
    for i, n in enumerate(names):
        rel = f"keyframes/{n}.jpg"
        Image.fromarray(imgs[i].numpy()).save(os.path.join(out, rel), quality=92)
        c2w = np.linalg.inv(viewmats[i].double().numpy())
        frames.append({"name": n, "image": rel, "K": Ks[i].tolist(), "c2w": c2w.tolist()})
    json.dump({"width": W, "height": H, "convention": "opencv_c2w", "frames": frames},
              open(os.path.join(out, "cameras.json"), "w"), indent=1)
    print(f"exported {len(frames)} frames at {W}x{H} -> {out}")


if __name__ == "__main__":
    main()
