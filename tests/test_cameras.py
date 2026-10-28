"""Convention check: per-pixel world points from ThriftSplat must reproject onto their own pixels.

Needs a real run: set LIVESPLAT_TEST_RUN to a ThriftSplat run dir with export/ (skipped otherwise).
"""
import glob
import json
import os
from pathlib import Path

import numpy as np
import pytest

from scene_io.thrift import load_cameras, project

RUN = os.environ.get("LIVESPLAT_TEST_RUN")


@pytest.mark.skipif(not RUN, reason="LIVESPLAT_TEST_RUN not set")
def test_reprojection():
    cams = load_cameras(RUN)
    scene = json.loads((Path(RUN) / "scene" / "scene.json").read_text())
    errs = []
    for f in cams["frames"][::10]:
        z = np.load(Path(RUN) / "scene" / f"frame_{f['name']}.npz")
        P = z["pts3d_world"].reshape(-1, 3)
        h, w = z["pts3d_world"].shape[:2]
        sx, sy = cams["width"] / w, cams["height"] / h
        vv, uu = np.mgrid[0:h, 0:w]
        # Stage-1 pose (no pose_delta) must reproject exactly; it checks the convention.
        fr = scene["frames"][f["name"]]
        c2w = np.eye(4); c2w[:3, :3] = np.array(fr["R_c2w"]); c2w[:3, 3] = fr["centre"]
        uv, depth = project(P, f["K"], c2w)
        exp = np.stack([(uu.ravel() + 0.5) * sx - 0.5, (vv.ravel() + 0.5) * sy - 0.5], 1)
        ok = np.isfinite(uv).all(1) & (depth > 0)
        errs.append(np.median(np.linalg.norm(uv[ok] - exp[ok], axis=1)))
    print("median reprojection px per frame:", np.round(errs, 2))
    assert np.median(errs) < 2.0
