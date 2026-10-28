"""T4: render the sim package from the capture cameras at rest and compare with the real keyframes.

Run (WSL, needs CUDA_HOME for the gsplat JIT): python -m sim.check_render <sim_dir>
Writes <sim_dir>/checks/{render_vs_real_*.jpg, pushed_*.jpg, render_check.json}.
These cameras were also used to train the splat, so PSNR here is a fit check, not a held-out score.
"""
import json
import sys
from pathlib import Path

import mujoco
import numpy as np
from PIL import Image

from sim.render import SplatRenderer


def psnr(a, b):
    mse = np.mean((a.astype(np.float32) - b.astype(np.float32)) ** 2)
    return float(10 * np.log10(255 ** 2 / max(mse, 1e-9)))


def table_mask(meta, cam):
    """Pixels covered by the table top and the objects at rest (convex hull of their projections)."""
    import cv2
    c2w, K = np.array(cam["c2w_opencv"]), np.array(cam["K"])
    pts = [np.c_[np.array(meta["table"]["polygon_xy"]), np.zeros(len(meta["table"]["polygon_xy"]))]]
    for o in meta["objects"]:
        sx, sy, sz = o["size_xyz"]
        p = np.array(o["pose"]["pos"])
        pts.append(p + np.array([[dx * sx / 2, dy * sy / 2, z * sz] for dx in (-1, 1) for dy in (-1, 1) for z in (0, 1)]))
    P = np.concatenate(pts)
    w2c = np.linalg.inv(c2w)
    pc = P @ w2c[:3, :3].T + w2c[:3, 3]
    uv = (pc @ K.T)[:, :2] / pc[:, 2:3]
    mask = np.zeros((cam["height"], cam["width"]), np.uint8)
    cv2.fillConvexPoly(mask, cv2.convexHull(uv.astype(np.int32)), 1)
    return mask.astype(bool)


def main(sim_dir):
    sim = Path(sim_dir)
    out = sim / "checks"
    out.mkdir(exist_ok=True)
    r = SplatRenderer(sim)
    m = mujoco.MjModel.from_xml_path(str(sim / "scene.xml"))
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    prov = r.meta["provenance"]
    kf = Path(prov["run"]) / prov["source"] / "export" / "keyframes"
    rows = []
    for name, c in r.cams.items():
        img = r.render(d, name)
        real = np.asarray(Image.open(kf / f"{c['frame']}.jpg").convert("RGB"))
        v = psnr(img, real)
        mk = table_mask(r.meta, c)
        vt = psnr(img[mk], real[mk])
        rows.append({"camera": name, "psnr": v, "psnr_table": vt})
        Image.fromarray(np.concatenate([img, real], 1)).save(out / f"render_vs_real_{name}.jpg", quality=88)
    mean = float(np.mean([x["psnr"] for x in rows]))
    mean_t = float(np.mean([x["psnr_table"] for x in rows]))
    print(f"render vs real at rest over {len(rows)} cameras: full frame {mean:.2f} dB, "
          f"table + objects region {mean_t:.2f} dB")

    # Physics in photoreal: push the first object, step 1 s, render from the first camera.
    cam = next(iter(r.cams))
    body = r.meta["objects"][0]["body"]
    j = m.body(body).jntadr[0]
    d.qvel[m.jnt_dofadr[j]:m.jnt_dofadr[j] + 3] = (1.0, 0.0, 0.0)
    for t in range(3):
        for _ in range(int(0.33 / m.opt.timestep)):
            mujoco.mj_step(m, d)
        Image.fromarray(r.render(d, cam)).save(out / f"pushed_{t}.jpg", quality=88)
    json.dump({"mean_psnr": mean, "mean_psnr_table": mean_t, "cameras": rows, "note": "training cameras: fit check, not held-out"},
              open(out / "render_check.json", "w"), indent=1)


if __name__ == "__main__":
    main(sys.argv[1])
