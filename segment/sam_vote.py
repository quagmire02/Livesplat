"""Stage 3: SAM 3.1 masks on keyframes -> per-Gaussian object labels by multi-view voting.

Run (WSL, paid, cached): python -m segment.sam_vote <run_dir> [--src thrift_sfm] [--every 3]
Needs labels_geo.npy (geometric labels, used to match mask instances to objects), props.json
(sam_prompt per object, from agent.properties), align.json. Writes labels.npy + sam_report.json.

Per keyframe and object: SAM 3.1 with the object's text prompt; the instance that overlaps the
object's projected Gaussians most is kept. A Gaussian votes only in frames where it is visible
(coarse z-buffer). Candidates = Gaussians above the table inside the table outline. The base band
and glossy reflections are re-attached afterwards (same rules as stage 2).
"""
import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

from backends.meta import budget, client
from backends.meta.sam_parse import parse_image
from scene_io import splat
from scene_io.thrift import load_cameras, project
from segment.cluster import attach_base_and_reflection, in_polygon


def zbuffer_visible(uv, z, W, H, cell=4, tol=0.03):
    """Visible mask: depth within tol (relative) of the nearest Gaussian in the same cell."""
    ij = np.floor(uv / cell).astype(int)
    ok = (z > 0) & (ij[:, 0] >= 0) & (ij[:, 1] >= 0) & (ij[:, 0] < W // cell) & (ij[:, 1] < H // cell)
    key = ij[:, 1] * (W // cell + 1) + ij[:, 0]
    zmin = np.full((W // cell + 1) * (H // cell + 1), np.inf)
    np.minimum.at(zmin, key[ok], z[ok])
    vis = np.zeros(len(z), bool)
    vis[ok] = z[ok] <= zmin[key[ok]] * (1 + tol)
    return vis, ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("--src", default="thrift_sfm")
    ap.add_argument("--every", type=int, default=3, help="use every Nth keyframe")
    a = ap.parse_args()
    run = Path(a.run_dir)
    cams = load_cameras(run / a.src)
    W, H = cams["width"], cams["height"]
    al = json.loads((run / "align.json").read_text())
    T, r, poly = np.array(al["T_world_to_table"]), al["radius"], np.array(al["polygon"])
    props = json.loads((run / "props.json").read_text())
    geo = np.load(run / "labels_geo.npy")
    raw = splat.load_raw(run / a.src / "splat.ply")
    means = raw["means"]
    opac = 1 / (1 + np.exp(-raw["opacity"]))
    pt = means @ T[:3, :3].T + T[:3, 3]
    ids = sorted(int(k) for k in props)
    cand = (geo > 0) | ((pt[:, 2] > 0.004 * r) & (pt[:, 2] < 1.5 * r) & in_polygon(pt[:, :2], poly * 1.05))
    occl = opac > 0.3                                     # Gaussians that can hide others
    idx = np.flatnonzero(cand | occl)
    votes = np.zeros((len(means), len(ids) + 1), np.float32)   # col 0 = visible but in no mask
    frames = cams["frames"][::a.every]
    print(f"{len(frames)} keyframes x {len(ids)} prompts = {len(frames) * len(ids)} SAM images "
          f"(~${len(frames) * len(ids) * 0.0025:.2f}); {cand.sum():,} candidate Gaussians")
    for f in frames:
        img = Image.open(f["image"])
        uri = client.image_data_uri(img, quality=90)
        uv, z = project(means[idx], f["K"], f["c2w"])
        vis, inside = zbuffer_visible(uv, z, W, H)
        u = np.clip(uv[:, 0].astype(int), 0, W - 1)
        v = np.clip(uv[:, 1].astype(int), 0, H - 1)
        seen = vis & cand[idx]
        votes[idx[seen], 0] += 1
        for j, k in enumerate(ids, 1):
            res = client.sam31_image(props[str(k)]["sam_prompt"], uri)
            if res["status"] != 200:
                continue
            masks = parse_image(res["raw"], (H, W))
            mine = vis & (geo[idx] == k)
            if not masks or mine.sum() < 20:
                continue
            best = max(masks, key=lambda mk: mk["mask"][v[mine], u[mine]].mean())
            if best["mask"][v[mine], u[mine]].mean() < 0.3:     # no instance covers this object here
                continue
            hit = seen & best["mask"][v, u]
            votes[idx[hit], j] += 1
            votes[idx[hit], 0] -= 1
    n_seen = votes.sum(1)
    frac = votes[:, 1:] / np.maximum(n_seen, 1)[:, None]
    best_k = np.array(ids)[np.argmax(frac, 1)]
    best_f = frac.max(1)
    # SAM may ADD Gaussians to an object and MOVE them between objects, but never sends object Gaussians
    # to the background: its masks miss parts of objects in some frames (half a bottle stayed behind).
    labels = np.maximum(geo, 0).astype(np.int32)
    seen = cand & (n_seen >= 3)
    add = seen & (labels == 0) & (best_f >= 0.5)
    labels[add] = best_k[add]
    cur = np.clip(labels - 1, 0, len(ids) - 1)
    cur_f = np.where(labels > 0, frac[np.arange(len(labels)), cur], 0)
    move = seen & (labels > 0) & (best_k != labels) & (best_f >= 0.6) & (cur_f < 0.2)
    labels[move] = best_k[move]
    labels = attach_base_and_reflection(pt, labels, r)
    np.save(run / "labels.npy", labels)
    rep = {"keyframes": len(frames), "prompts": {k: props[str(k)]["sam_prompt"] for k in ids},
           "per_object": {k: {"geo": int((geo == k).sum()), "sam": int((labels == k).sum()),
                              "added": int(((labels == k) & (geo != k)).sum()),
                              "removed": int(((geo == k) & (labels != k)).sum())} for k in ids},
           "spent_total_usd": budget.spent()}
    (run / "sam_report.json").write_text(json.dumps(rep, indent=1))
    for k, s in rep["per_object"].items():
        print(f"  obj {k} ({rep['prompts'][k]}): geo {s['geo']:,} -> sam {s['sam']:,} "
              f"(+{s['added']:,} / -{s['removed']:,})")
    print(f"spent total ${budget.spent():.4f}")


if __name__ == "__main__":
    main()
