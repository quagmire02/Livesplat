"""Click to add a missed object: SAM 3.1 point prompts on keyframes -> new object from background splats.
Paid: one SAM image per keyframe used (default 6 -> ~$0.015), cached.
"""
import numpy as np
from PIL import Image

from backends.meta import client
from backends.meta.sam_parse import parse_image
from scene_io.thrift import project
from segment.sam_vote import zbuffer_visible


def ray_hit(sc, keep, ray_o, ray_d):
    """First opaque background Gaussian along the click ray (table frame), or None."""
    m = sc.bg["means"]
    d = np.asarray(ray_d, float) / np.linalg.norm(ray_d)
    rel = m - np.asarray(ray_o, float)
    t = rel @ d
    perp = np.linalg.norm(rel - np.outer(t, d), axis=1)
    ok = keep & (sc.bg["opacity"] > 0.3) & (t > 0) & (perp < 0.004 * t + 0.004 * sc.radius)
    if not ok.any():
        return None
    return m[np.flatnonzero(ok)[np.argmin(t[ok])]]


def segment_click(sc, cams, keep, p_table, n_frames=6, radius_frac=0.5):
    """Mask of background splats (in sc.bg order) that form the clicked object."""
    Tinv = np.linalg.inv(sc.T)
    to_w = lambda P: P @ Tinv[:3, :3].T + Tinv[:3, 3]
    W, H = cams["width"], cams["height"]
    allp = to_w(np.concatenate([sc.bg["means"][keep & (sc.bg["opacity"] > 0.3)], p_table[None]]))
    near = keep & (np.linalg.norm(sc.bg["means"] - p_table, axis=1) < radius_frac * sc.radius)
    cand = np.flatnonzero(near)
    Pw = to_w(sc.bg["means"][cand])
    views = []
    for f in cams["frames"]:                                   # frames where the clicked point is visible
        uv, z = project(allp, f["K"], f["c2w"])
        vis, _ = zbuffer_visible(uv, z, W, H)
        u, v = uv[-1]
        if vis[-1] and 5 < u < W - 5 and 5 < v < H - 5:
            views.append((f, u, v))
    if not views:
        return None, 0
    views = [views[i] for i in np.linspace(0, len(views) - 1, min(n_frames, len(views))).astype(int)]
    votes, seen = np.zeros(len(cand)), np.zeros(len(cand))
    for f, u, v in views:
        prompt = f"<0f>1<|point;x={int(u)};y={int(v)};w={W};h={H}|>"
        res = client.sam31_image(prompt, client.image_data_uri(Image.open(f["image"]), quality=90))
        if res["status"] != 200:
            continue
        masks = parse_image(res["raw"], (H, W))
        if not masks:
            continue
        mk = max(masks, key=lambda m: m["mask"][int(v), int(u)])["mask"]
        uv, z = project(np.concatenate([Pw, allp]), f["K"], f["c2w"])
        vis, _ = zbuffer_visible(uv, z, W, H)
        uv, vis = uv[:len(cand)], vis[:len(cand)]
        ui, vi = np.clip(uv[:, 0].astype(int), 0, W - 1), np.clip(uv[:, 1].astype(int), 0, H - 1)
        seen += vis
        votes += vis & mk[vi, ui]
    sel = np.zeros(len(keep), bool)
    sel[cand[(seen >= 2) & (votes >= 0.5 * seen)]] = True
    return sel, len(views)
