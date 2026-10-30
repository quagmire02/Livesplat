"""3D model of one object for the user: real scan photos -> Hunyuan3D-2mv shape -> real-photo colours.

  python -m viewer.model3d build <run> [<id> ...]       # all three steps (viewer must be closed: RAM)
  python -m viewer.model3d prep <run> <id> [<id> ...]   # pick front/left/back/right photos, SAM 3.1 cut-outs
  ~/venvs/hy3d/bin/python segment/hy3d_shape.py <run>/gen/model_<id> ...   # shape (separate venv, GPU)
  python -m viewer.model3d paint <run> <id> [<id> ...]  # align to the scan, colour from real photos -> model.glb

Views: Muse picks which photo shows the FRONT (brand/main label); left/back/right are the real frames closest
to front -90/180/+90 deg azimuth (Hunyuan order: front, clockwise 90, back, clockwise 270), preferring level,
large views. Colours: every vertex takes the pixel of the real photo that sees it most head-on (z-buffered);
parts no photo saw (underside) copy their nearest seen neighbours. Output glb is y-up.
"""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from scene_io.thrift import load_cameras, project
from segment.sam_vote import zbuffer_visible

VIEWS = {"front": 0.0, "left": -90.0, "back": 180.0, "right": 90.0}     # deg from the front azimuth


def _setup(run, k, src):
    from viewer.app import Scene, body_world
    sc = Scene(run, src)
    o = next(o for o in sc.objects if o["id"] == k)
    return sc, o, body_world(sc, o), load_cameras(Path(run) / src)


def _candidates(sc, o, pw, cams):
    W, H = cams["width"], cams["height"]
    R, t = sc.T[:3, :3], sc.T[:3, 3]
    out = []
    for f in cams["frames"]:
        uv, z = project(pw, f["K"], f["c2w"])
        uv = uv[z > 0]
        if len(uv) < 50:
            continue
        lo, hi = np.percentile(uv, [1, 99], 0)
        if (lo < 2).any() or hi[0] > W - 2 or hi[1] > H - 2:
            continue                                            # object cut by the image border
        d = R @ f["c2w"][:3, 3] + t - o["base"]
        out.append({"f": f, "az": float(np.degrees(np.arctan2(d[1], d[0]))),
                    "el": float(np.degrees(np.arctan2(d[2], np.hypot(d[0], d[1])))),
                    "size": float(np.prod(hi - lo)), "lo": lo, "hi": hi})
    return out


def _crop(c, pad=0.25, W=None, H=None):
    lo, hi = c["lo"], c["hi"]
    p = pad * (hi - lo)
    return (max(0, int(lo[0] - p[0])), max(0, int(lo[1] - p[1])), min(W, int(hi[0] + p[0])), min(H, int(hi[1] + p[1])))


def _front(cands, name, W, H):
    """Muse picks the front among up to 6 views spread around the object."""
    from backends.meta import client
    bins = {}
    for c in cands:
        b = int((c["az"] + 180) // 60)
        if b not in bins or c["size"] > bins[b]["size"]:
            bins[b] = c
    pick = [bins[b] for b in sorted(bins)]
    content = [{"type": "text", "text": f'These {len(pick)} photos (index 0..{len(pick) - 1}) show the same "{name}" '
                'from different sides. Which photo shows its FRONT (brand, main label, or controls facing the '
                'camera)? Reply JSON only: {"front": index}'}]
    for c in pick:
        im = Image.open(c["f"]["image"]).convert("RGB").crop(_crop(c, 0.15, W, H))
        im.thumbnail((384, 384))
        content.append({"type": "image_url", "image_url": {"url": client.image_data_uri(im, quality=85)}})
    r = client.spark_chat([{"role": "user", "content": content}], max_tokens=1500, extra={"reasoning_effort": "low"})
    from backends.jsonfix import reply_json
    try:
        i = int((reply_json(r) or {})["front"])
        return pick[i]["az"] if 0 <= i < len(pick) else pick[0]["az"]
    except (ValueError, KeyError, TypeError):
        return max(pick, key=lambda c: c["size"])["az"]


def _cutout(c, pw, prompt, W, H):
    """SAM 3.1 mask of the object in this frame -> RGBA crop (alpha = mask, largest connected piece)."""
    from backends.meta import client
    from backends.meta.sam_parse import parse_image
    x0, y0, x1, y1 = _crop(c, 0.25, W, H)
    img = Image.open(c["f"]["image"]).convert("RGB").crop((x0, y0, x1, y1))
    res = client.sam31_image(prompt, client.image_data_uri(img, quality=92))
    if res["status"] != 200:
        return None
    masks = parse_image(res["raw"], (y1 - y0, x1 - x0))
    uv, z = project(pw, c["f"]["K"], c["f"]["c2w"])
    vis, _ = zbuffer_visible(uv, z, W, H, cell=3, tol=0.05)
    u = np.clip(uv[vis, 0] - x0, 0, x1 - x0 - 1).astype(int)
    v = np.clip(uv[vis, 1] - y0, 0, y1 - y0 - 1).astype(int)
    on = [m["mask"][v, u].mean() if len(u) else 0 for m in masks]
    if not masks or max(on) < 0.3:
        return None
    m = masks[int(np.argmax(on))]["mask"].astype(np.uint8)
    n, lab, stats, _ = cv2.connectedComponentsWithStats(m)
    if n > 2:
        m = (lab == 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))).astype(np.uint8)
    rgba = np.dstack([np.asarray(img), m * 255])
    ys, xs = np.nonzero(m)
    return Image.fromarray(rgba[ys.min():ys.max() + 1, xs.min():xs.max() + 1])


def prep(run, k, src="thrift_sfm"):
    sc, o, pw, cams = _setup(run, k, src)
    W, H = cams["width"], cams["height"]
    props = json.loads((Path(run) / "props.json").read_text())
    names = json.loads((Path(run) / "names.json").read_text())
    name = (names.get(str(k)) or [f"object {k}"])[0]
    prompt = props.get(str(k), {}).get("sam_prompt", name)
    cands = _candidates(sc, o, pw, cams)
    front = _front(cands, name, W, H)
    d = Path(run) / "gen" / f"model_{k}"
    d.mkdir(parents=True, exist_ok=True)
    for old in d.glob("*.png"):
        old.unlink()
    used = {}
    for view, off in VIEWS.items():
        want = front + off
        diff = lambda c: abs((c["az"] - want + 180) % 360 - 180)
        near = [c for c in cands if diff(c) <= 35]
        if not near:
            continue
        c = max(near, key=lambda c: c["size"] * np.cos(np.radians(diff(c))) * np.cos(np.radians(c["el"])) ** 2)
        cut = _cutout(c, pw, prompt, W, H)
        if cut is None:
            continue
        cut.save(d / f"{view}.png")
        used[view] = {"frame": Path(c["f"]["image"]).name, "azimuth_off_deg": round(diff(c), 1),
                      "elevation_deg": round(c["el"], 1)}
    (d / "views.json").write_text(json.dumps({"name": name, "front_azimuth": front, "views": used}, indent=1))
    print(f"obj {k} ({name}): views {used}", flush=True)
    return d


def paint(run, k, src="thrift_sfm"):
    import trimesh
    from scipy.spatial import cKDTree
    from segment.complete3d import align
    sc, o, pw, cams = _setup(run, k, src)
    W, H = cams["width"], cams["height"]
    d = Path(run) / "gen" / f"model_{k}"
    mesh = trimesh.load(d / "shape.glb", force="mesh")
    v = np.asarray(mesh.vertices) @ np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]]).T      # y-up -> z-up
    pts = o["capture_means"][o["body_ids"]] if "body_ids" in o else o["capture_means"]
    v, info = align(v, pts)                                                            # object frame (scan)
    m = trimesh.Trimesh(v, mesh.faces, process=False)
    R, t = sc.T[:3, :3], sc.T[:3, 3]
    Pw = (v + o["base"] - t) @ R                                                       # table -> world
    Nw = np.asarray(m.vertex_normals) @ R
    from viewer.app import body_world
    others = np.concatenate([body_world(sc, x)[::3] for x in sc.objects if x["id"] != k] or [np.zeros((0, 3))])
    allp = np.concatenate([Pw, others])                    # other objects can hide this one in a photo
    low = v[:, 2] < 0.04 * np.ptp(v[:, 2])                  # base rim sees the table: fill from neighbours
    best = np.full(len(v), -1.0)
    col = np.zeros((len(v), 3), np.uint8)
    for f in cams["frames"][::2]:
        uva, za = project(allp, f["K"], f["c2w"])
        uv, z = uva[:len(v)], za[:len(v)]
        inside = (z > 0) & (uv[:, 0] >= 0) & (uv[:, 1] >= 0) & (uv[:, 0] < W - 1) & (uv[:, 1] < H - 1)
        if inside.sum() < 50:
            continue
        vis, _ = zbuffer_visible(uva, za, W, H, cell=3, tol=0.02)
        vis = vis[:len(v)] & ~low
        to_cam = f["c2w"][:3, 3] - Pw
        to_cam /= np.linalg.norm(to_cam, axis=1, keepdims=True)
        s = (Nw * to_cam).sum(1)
        upd = inside & vis & (s > 0.25) & (s > best)
        if not upd.any():
            continue
        img = np.asarray(Image.open(f["image"]).convert("RGB"))
        sx, sy = img.shape[1] / W, img.shape[0] / H
        col[upd] = img[(uv[upd, 1] * sy).astype(int), (uv[upd, 0] * sx).astype(int)]
        best[upd] = s[upd]
    seen = best > 0
    if seen.any() and (~seen).any():                       # underside etc.: nearest seen colours
        _, j = cKDTree(v[seen]).query(v[~seen], k=4)
        col[~seen] = col[seen][j].mean(1).astype(np.uint8)
    out = trimesh.Trimesh(v @ np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]]).T, mesh.faces,     # back to y-up
                          vertex_colors=np.c_[col, np.full(len(col), 255, np.uint8)], process=False)
    out.export(d / "model.glb")
    # The complete shape is also the physics collision shape (viewer.app reads gen/obj_<id>.obj, object frame z-up,
    # with a flat base when it stands alone); it replaces the old TripoSR mesh.
    trimesh.Trimesh(v, mesh.faces, process=False).export(Path(run) / "gen" / f"obj_{k}.obj")
    rep = {**info, "vertices": int(len(v)), "seen_by_photos": round(float(seen.mean()), 3)}
    (d / "paint.json").write_text(json.dumps(rep, indent=1))
    print(f"obj {k}: {len(v):,} verts, {seen.mean() * 100:.0f}% coloured from photos, chamfer {info['chamfer']:.4f}",
          flush=True)


def build(run, ids, src="thrift_sfm"):
    """prep -> Hunyuan shape (separate venv) -> paint, for every id. The viewer must be closed (8 GB RAM)."""
    import os
    import subprocess
    import sys
    if subprocess.run(["pgrep", "-f", "viewer.app"], capture_output=True).returncode == 0:
        sys.exit("Close the viewer first: the shape model needs most of the 8 GB WSL RAM.")
    dirs = [prep(run, k, src) for k in ids]
    hy = Path(os.environ.get("HY3D_PY", Path.home() / "venvs" / "hy3d" / "bin" / "python"))
    script = Path(__file__).resolve().parents[1] / "segment" / "hy3d_shape.py"
    subprocess.run([str(hy), "-u", str(script), *map(str, dirs)], check=True,
                   env={**os.environ, "HF_HUB_DISABLE_XET": "1"})
    for k in ids:
        paint(run, k, src)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["prep", "paint", "build"])
    ap.add_argument("run_dir")
    ap.add_argument("ids", type=int, nargs="*", help="object ids (default: all)")
    ap.add_argument("--src", default="thrift_sfm")
    a = ap.parse_args()
    ids = a.ids or [int(k) for k in json.loads((Path(a.run_dir) / "names.json").read_text())]
    if a.step == "build":
        build(a.run_dir, ids, a.src)
        return
    for k in ids:
        (prep if a.step == "prep" else paint)(a.run_dir, k, a.src)


if __name__ == "__main__":
    main()
