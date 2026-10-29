"""Isolate one object with SAM 3.1 in several source frames and diagnose the reconstruction against the masks.

Cache-first: frames come from every 3rd keyframe with the object's sam_prompt (the same requests as
segment.sam_vote), so already-tagged scenes cost $0. A cache miss needs live permission.
Per view: coverage = share of the SAM mask filled by the object's projected splats (holes, missing parts);
spill = object splats outside the mask (table bits, floaters). Suggested fix = splats outside the mask in
>= 2 views where they are visible.
"""
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from backends.meta import client
from backends.meta.sam_parse import parse_image
from scene_io.thrift import project
from segment.sam_vote import zbuffer_visible


def _views(points_w, cams, every=3, count=3):
    """Large, fully in-frame views with distinct directions, from the frames segment.sam_vote used."""
    centre = np.median(points_w, 0)
    W, H = cams["width"], cams["height"]
    cand = []
    for f in cams["frames"][::every]:
        uv, z = project(points_w, f["K"], f["c2w"])
        uv = uv[z > 0]
        if len(uv) < 50:
            continue
        lo, hi = np.percentile(uv, [2, 98], 0)
        if (lo < 0).any() or hi[0] > W or hi[1] > H:
            continue
        d = f["c2w"][:3, 3] - centre
        cand.append((float(np.prod(hi - lo)), f, d / max(np.linalg.norm(d), 1e-9)))
    out = []
    for c in sorted(cand, key=lambda x: -x[0]):
        if all(np.dot(c[2], p[2]) < 0.94 for p in out):
            out.append(c)
        if len(out) == count:
            break
    return [c[1] for c in out]


def isolate(points_w, prompt, cams, cache_only=True):
    """points_w: object splat centres (world frame, object order). Returns a report dict."""
    W, H = cams["width"], cams["height"]
    sub = np.arange(len(points_w))
    views, outside, seen = [], np.zeros(len(points_w)), np.zeros(len(points_w))
    for f in _views(points_w, cams):
        img = Image.open(f["image"]).convert("RGB")
        res = client.sam31_image(prompt, client.image_data_uri(img, quality=90), cache_only=cache_only)
        if res["status"] != 200:
            continue
        masks = parse_image(res["raw"], (H, W))
        uv, z = project(points_w, f["K"], f["c2w"])
        vis, inside = zbuffer_visible(uv, z, W, H, cell=3, tol=0.05)
        u = np.clip(uv[:, 0].astype(int), 0, W - 1)
        v = np.clip(uv[:, 1].astype(int), 0, H - 1)
        if not masks or vis.sum() < 30:
            continue
        mk = max(masks, key=lambda m: m["mask"][v[vis], u[vis]].mean())["mask"]
        if mk[v[vis], u[vis]].mean() < 0.3:
            continue                                         # SAM found something else here
        grown = cv2.dilate(mk.astype(np.uint8), np.ones((7, 7), np.uint8)).astype(bool)
        out_v = vis & ~grown[v, u]
        seen += vis
        outside += out_v
        filled = np.zeros((H, W), np.uint8)
        filled[v[inside], u[inside]] = 1
        filled = cv2.dilate(filled, np.ones((5, 5), np.uint8)).astype(bool)
        coverage = float((filled & mk).sum() / max(mk.sum(), 1))
        # Thumbnail (object only) and diagnosis overlay (green outline, red spill, magenta = not reconstructed).
        ys, xs = np.nonzero(mk)
        pad = int(0.15 * max(np.ptp(xs), np.ptp(ys)))
        x0, y0 = max(0, xs.min() - pad), max(0, ys.min() - pad)
        x1, y1 = min(W, xs.max() + pad), min(H, ys.max() + pad)
        arr = np.asarray(img)
        thumb = np.where(mk[..., None], arr, (arr * 0.25 + 96).astype(np.uint8))[y0:y1, x0:x1]
        over = arr.copy()
        over[mk & ~filled] = (0.4 * over[mk & ~filled] + 0.6 * np.array([255, 0, 255])).astype(np.uint8)
        cnt, _ = cv2.findContours(mk.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(over, cnt, -1, (60, 220, 60), 2)
        for x, y in zip(u[out_v], v[out_v]):
            cv2.circle(over, (int(x), int(y)), 1, (255, 40, 40), -1)
        views.append({"frame": Path(f["image"]).name, "coverage": round(coverage, 3),
                      "spill": round(float(out_v.sum() / max(vis.sum(), 1)), 3),
                      "thumb": Image.fromarray(thumb), "overlay": Image.fromarray(over[y0:y1, x0:x1])})
    spill_ids = sub[(seen >= 2) & (outside >= np.maximum(2, 0.6 * seen))]
    return {"views": views, "spill_ids": spill_ids, "prompt": prompt,
            "coverage_min": min((x["coverage"] for x in views), default=None)}


def metrics_text(rep):
    lines = [f"view {i + 1} ({x['frame']}): SAM mask filled by reconstruction {x['coverage'] * 100:.0f}%, "
             f"object splats outside the mask {x['spill'] * 100:.0f}%" for i, x in enumerate(rep["views"])]
    lines.append(f"splats outside the mask in 2+ views (candidate attachments): {len(rep['spill_ids'])}")
    return "\n".join(lines)


def save_report(run, k, rep):
    p = Path(run) / "gen" / f"sam_isolate_{k}.json"
    p.parent.mkdir(exist_ok=True)
    p.write_text(json.dumps({"prompt": rep["prompt"], "views": [{x: v[x] for x in ("frame", "coverage", "spill")}
                                                               for v in rep["views"]],
                             "spill_count": int(len(rep["spill_ids"]))}, indent=1))
