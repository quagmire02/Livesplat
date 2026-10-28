"""Load a ThriftSplat run. Cameras come from recon/export_thrift.py (cameras.json + keyframes/).

cameras.json: {"width", "height", "frames": [{"name", "image", "K" 3x3, "c2w" 4x4 OpenCV}]}
c2w already includes the pose_delta that gsplat training learned.
"""
import json
from pathlib import Path

import numpy as np

from scene_io import splat


def load_cameras(run_dir):
    d = json.loads((Path(run_dir) / "export" / "cameras.json").read_text())
    for f in d["frames"]:
        f["K"] = np.array(f["K"], np.float64)
        f["c2w"] = np.array(f["c2w"], np.float64)
        f["image"] = str(Path(run_dir) / "export" / f["image"])
    return d


def load_splat(run_dir):
    return splat.load_raw(Path(run_dir) / "splat.ply")


def project(points_w, K, c2w):
    """World points (N,3) -> pixel uv (N,2), camera depth z (N,). OpenCV convention."""
    w2c = np.linalg.inv(c2w)
    pc = points_w @ w2c[:3, :3].T + w2c[:3, 3]
    z = pc[:, 2]
    uv = pc @ K.T
    uv = uv[:, :2] / np.clip(uv[:, 2:3], 1e-9, None)
    return uv, z
