"""Muse Image hole fill: realistic table texture under objects (paid, ~$0.01 per scene, cached).

Run (WSL, gsplat env): sim/run.sh viewer.holefill_ai <run_dir>
1. Render the table top from above (background splats only, objects removed) with gsplat.
2. Paint every object footprint magenta (Muse Image has no mask input).
3. Muse Image edits: replace magenta with the table surface, keep the rest.
4. Sample the edited image at each fill Gaussian -> <run>/gen/fill_rgb_<k>.npy (viewer uses it).
"""
import base64
import io
import json
import sys
from pathlib import Path

import numpy as np
import requests
import torch
from PIL import Image

from backends import cache
from backends.meta import budget, client
from viewer.app import Scene
from viewer.holefill import footprint

PX = 1024
PROMPT = ("Top-down photo of a wooden table. Replace every magenta area with clean wooden table surface that "
          "continues the surrounding wood grain and colour seamlessly. No reflections, no white or grey patches, "
          "no objects. Do not change anything outside the magenta areas.")


def top_camera(r):
    """Near-orthographic camera far above the table centre, looking down (OpenCV axes)."""
    h = 20 * r
    f = PX / 2 / (1.2 * r / h)                                 # 2.4 r across the image
    c2w = np.eye(4)
    c2w[:3, :3] = np.diag([1.0, -1.0, -1.0])                  # x right, y down (= -world y), z down
    c2w[:3, 3] = [0, 0, h]
    K = np.array([[f, 0, PX / 2], [0, f, PX / 2], [0, 0, 1]])
    return c2w, K


def to_px(xy, c2w, K):
    h = c2w[2, 3]
    return np.c_[K[0, 0] * xy[:, 0] / h + K[0, 2], K[1, 1] * -xy[:, 1] / h + K[1, 2]]


def render(bg, c2w, K):
    from gsplat import rasterization
    from scene_io import splat
    t = lambda x: torch.tensor(x, device="cuda", dtype=torch.float32)
    sel = bg["means"][:, 2] < 0.05 * np.abs(bg["means"][:, :2]).max()   # table top and below
    out, _, _ = rasterization(
        means=t(bg["means"][sel]), quats=t(bg["quats"][sel]), scales=t(bg["scales"][sel]),
        opacities=t(bg["opacity"][sel]), colors=t(bg["rgb"][sel]),
        viewmats=t(np.linalg.inv(c2w))[None], Ks=t(K)[None], width=PX, height=PX, sh_degree=None)
    return (out[0].clamp(0, 1) * 255).byte().cpu().numpy()


def muse_edit(img, prompt):
    req = {"model": "muse-image-1.0", "prompt": prompt, "n": 1,
           "images": [{"image_url": client.image_data_uri(Image.fromarray(img), fmt="PNG")}]}
    hit = cache.get(req)
    if hit is None:
        budget.check(0.01, "muse-image edit")
        r = requests.post(f"{client.BASE}/images/edits", headers=client._headers(), json=req, timeout=300)
        budget.record(0.01, "muse-image edit")
        if r.status_code != 200:
            sys.exit(f"Muse Image error {r.status_code}: {r.text[:300]}")
        hit = r.json()
        cache.put(req, hit, {"model": "muse-image-1.0"})
    return np.asarray(Image.open(io.BytesIO(base64.b64decode(hit["data"][0]["b64_json"]))).convert("RGB"))


def main(run_dir):
    run = Path(run_dir)
    sc = Scene(run)
    c2w, K = top_camera(sc.radius)
    top = render(sc.bg, c2w, K)
    painted = top.copy()
    import cv2
    for o in sc.objects:
        poly = to_px(footprint(o["hull"]) + o["base"][:2], c2w, K).astype(np.int32)
        cv2.fillPoly(painted, [poly], (255, 0, 255))
    (run / "gen").mkdir(exist_ok=True)
    Image.fromarray(painted).save(run / "gen" / "table_top_painted.png")
    edited = muse_edit(painted, PROMPT)
    edited = np.asarray(Image.fromarray(edited).resize((PX, PX)))
    Image.fromarray(edited).save(run / "gen" / "table_top_filled.png")
    # Sanity: outside the footprints the edit should match the input (else it moved/restyled the image).
    keep = np.all(painted != [255, 0, 255], axis=-1)
    tt = top[keep] / 255.0                                         # table colour: saturated (wood) pixels only
    wood = np.median(tt[(tt.max(1) - tt.min(1)) > 0.12], 0)
    err = float(np.abs(edited[keep].astype(float) - top[keep]).mean())
    for o in sc.objects:
        if o["fill"] is None:
            continue
        uv = np.clip(to_px(o["fill"]["means"][:, :2] + o["base"][:2], c2w, K).astype(int), 0, PX - 1)
        rgb = edited[uv[:, 1], uv[:, 0]] / 255.0
        grey = (rgb.max(1) - rgb.min(1)) < 0.08                   # reflections copied from around the hole
        rgb[grey] = wood
        np.save(run / "gen" / f"fill_rgb_{o['id']}.npy", rgb.astype(np.float32))
    json.dump({"mean_abs_diff_outside": err, "spent_total_usd": budget.spent()},
              open(run / "gen" / "holefill_ai.json", "w"), indent=1)
    print(f"Muse Image fill done. Outside-footprint difference {err:.1f}/255. spent ${budget.spent():.4f}")


if __name__ == "__main__":
    main(sys.argv[1])
