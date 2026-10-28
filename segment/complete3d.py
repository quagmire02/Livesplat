"""Stage 3b: complete each object in 3D with TripoSR (image -> full mesh), aligned to the scan.
The scan never sees backs and undersides; the generated mesh fills them.

Run (WSL, local GPU, free): python -m segment.complete3d <run_dir> [--src thrift_sfm] [--res 192]
Needs labels (labels.npy or labels_geo.npy), align.json, export/ keyframes. TripoSR (MIT) in
~/livesplat_data/models/TripoSR (code + weights/). torchmcubes is replaced by scikit-image.
Writes <run>/gen/obj_<k>.obj (aligned, object frame) and obj_<k>_fill.ply (raw 3DGS, object frame:
Gaussians only where the scan has none). viewer.app adds the fill to the object splat.
"""
import argparse
import json
import sys
import types
from pathlib import Path

import numpy as np
import torch
import trimesh
from PIL import Image
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

from scene_io import splat
from scene_io.thrift import load_cameras, project

TSR_DIR = Path.home() / "livesplat_data" / "models" / "TripoSR"


def _mcubes_shim():
    """TripoSR imports torchmcubes (CUDA build fails here); same API on scikit-image."""
    from skimage.measure import marching_cubes as mc

    def marching_cubes(vol, thresh):
        v, f, _, _ = mc(vol.cpu().numpy(), thresh)
        return torch.from_numpy(v[:, ::-1].copy()).float(), torch.from_numpy(f.astype(np.int64))
    sys.modules["torchmcubes"] = types.SimpleNamespace(marching_cubes=marching_cubes)
    sys.modules.setdefault("rembg", types.SimpleNamespace(remove=None, new_session=None))   # unused: our mask


def sam_view(k, props, objs_world, cams, every=3, pad=0.12):
    """Best keyframe for object k from the cached SAM 3.1 masks: largest mask, inside the image, least
    covered by the other objects' masks. Returns a TripoSR-ready RGB image (object on grey)."""
    from backends.meta import client
    from backends.meta.sam_parse import parse_image
    W, H = cams["width"], cams["height"]
    best = None
    for f in cams["frames"][::every]:
        img = Image.open(f["image"])
        uri = client.image_data_uri(img, quality=90)              # same request as sam_vote -> cache hit
        chosen = {}
        for j, Pw in objs_world.items():
            res = client.sam31_image(props[str(j)]["sam_prompt"], uri)
            if res["status"] != 200:
                continue
            masks = parse_image(res["raw"], (H, W))
            uv, z = project(Pw, f["K"], f["c2w"])
            ok = (z > 0) & (uv[:, 0] >= 0) & (uv[:, 1] >= 0) & (uv[:, 0] < W) & (uv[:, 1] < H)
            if not masks or ok.sum() < 50:
                continue
            u, v = uv[ok, 0].astype(int), uv[ok, 1].astype(int)
            mk = max(masks, key=lambda m: m["mask"][v, u].mean())
            if mk["mask"][v, u].mean() > 0.3:
                chosen[j] = mk["mask"]
        if k not in chosen:
            continue
        m = chosen[k]
        ys, xs = np.nonzero(m)
        if xs.min() <= 1 or ys.min() <= 1 or xs.max() >= W - 2 or ys.max() >= H - 2:
            continue                                               # cut by the image border
        others = np.zeros_like(m)
        for j, mj in chosen.items():
            if j != k:
                others |= mj
        ring = cv2_dilate(m, 9) & ~m                              # another object touching the outline
        score = m.sum() * (1 - (ring & others).sum() / max(ring.sum(), 1))
        if best is None or score > best[0]:
            best = (score, f, m)
    _, f, m = best
    img = np.asarray(Image.open(f["image"]).convert("RGB"))
    ys, xs = np.nonzero(m)
    s = int(max(np.ptp(xs), np.ptp(ys)) * (1 + 2 * pad))
    cx, cy = (xs.min() + xs.max()) // 2, (ys.min() + ys.max()) // 2
    out = np.full((s, s, 3), 127, np.uint8)
    oy, ox = cy - s // 2, cx - s // 2
    yy, xx = ys - oy, xs - ox
    keep = (yy >= 0) & (xx >= 0) & (yy < s) & (xx < s)
    out[yy[keep], xx[keep]] = img[ys[keep], xs[keep]]
    return Image.fromarray(out).resize((512, 512)), f


def cv2_dilate(m, k):
    import cv2
    return cv2.dilate(m.astype(np.uint8), np.ones((k, k), np.uint8)).astype(bool)


def align(mesh_v, pts):
    """Upright similarity: TripoSR meshes are z-up; search yaw (5 deg) only, scale from height,
    xy from medians. No free tilt (ICP tilted and flipped objects)."""
    tree = cKDTree(pts)
    H = np.percentile(pts[:, 2], 98)
    sub = pts[::max(1, len(pts) // 3000)]
    def fit(yaw, pitch, roll):
        # TripoSR assumes a level camera; our frames look down, so small tilts are searched too.
        R = Rotation.from_euler("zyx", [yaw, pitch, roll]).as_matrix()
        v = mesh_v @ R.T
        v = v * (H / max(np.ptp(v[:, 2]), 1e-6))
        v = v - [*np.median(v[:, :2], 0), v[:, 2].min()] + [*np.median(pts[:, :2], 0), 0]
        return tree.query(v[::max(1, len(v) // 2000)])[0].mean() + cKDTree(v).query(sub)[0].mean(), v

    best = None
    for yaw in np.radians(np.arange(0, 360, 10)):
        for pitch in np.radians([-30, -15, 0, 15, 30]):
            for roll in np.radians([-30, -15, 0, 15, 30]):
                d, _ = fit(yaw, pitch, roll)
                if best is None or d < best[0]:
                    best = (d, yaw, pitch, roll)
    _, y0, p0, r0 = best
    for yaw in y0 + np.radians([-5, 0, 5]):                     # refine
        for pitch in p0 + np.radians([-5, 0, 5]):
            for roll in r0 + np.radians([-5, 0, 5]):
                d, _ = fit(yaw, pitch, roll)
                if d < best[0]:
                    best = (d, yaw, pitch, roll)
    d, yaw, pitch, roll = best
    _, v = fit(yaw, pitch, roll)
    return v, {"yaw_deg": float(np.degrees(yaw)), "tilt_deg": [float(np.degrees(pitch)), float(np.degrees(roll))],
               "chamfer": float(d)}


def fill_gaussians(mesh, pts, k_sh, gap, n=120000):
    """Flat, semi-transparent Gaussians on the generated surface where the scan has none."""
    P, fid = trimesh.sample.sample_surface(mesh, n)
    col = mesh.visual.vertex_colors[mesh.faces[fid, 0], :3] / 255.0 if mesh.visual.kind == "vertex" \
        else np.full((len(P), 3), 0.6)
    nrm = mesh.face_normals[fid]
    far = cKDTree(pts).query(P)[0] > gap
    P, col, nrm = P[far], col[far], nrm[far]
    z = np.array([0, 0, 1.0])                                     # rotate disc normal (z) onto the face normal
    axis = np.cross(z, nrm)
    s = np.linalg.norm(axis, axis=1, keepdims=True)
    ang = np.arctan2(s[:, 0], nrm @ z)
    axis = np.where(s > 1e-8, axis / np.maximum(s, 1e-8), [1, 0, 0])
    q = np.c_[np.cos(ang / 2), axis * np.sin(ang / 2)[:, None]].astype(np.float32)
    r = gap * 0.35
    return {"means": P.astype(np.float32), "sh0": ((col - 0.5) / splat.SH_C0).astype(np.float32),
            "shN": np.zeros((len(P), k_sh), np.float32), "opacity": np.full(len(P), 0.4, np.float32),
            "scales": np.log(np.tile([r, r, r * 0.1], (len(P), 1))).astype(np.float32), "quats": q}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("--src", default="thrift_sfm")
    ap.add_argument("--res", type=int, default=192, help="marching-cubes resolution (VRAM/time)")
    a = ap.parse_args()
    run = Path(a.run_dir)
    out = run / "gen"
    out.mkdir(exist_ok=True)
    _mcubes_shim()
    sys.path.insert(0, str(TSR_DIR))
    from tsr.system import TSR

    model = TSR.from_pretrained(str(TSR_DIR / "weights"), config_name="config.yaml", weight_name="model.ckpt")
    model.renderer.set_chunk_size(4096)
    model.to("cuda")
    cams = load_cameras(run / a.src)
    al = json.loads((run / "align.json").read_text())
    T = np.array(al["T_world_to_table"])
    labels = np.load(next(p for p in (run / "labels.npy", run / "labels_geo.npy") if p.exists()))
    raw = splat.load_raw(run / a.src / "splat.ply")
    op = 1 / (1 + np.exp(-raw["opacity"]))
    rep = {}
    props = json.loads((run / "props.json").read_text())
    objs_world = {k: raw["means"][(labels == k) & (op > 0.3)] for k in range(1, labels.max() + 1)}
    for k in range(1, labels.max() + 1):
        Pw = objs_world[k]
        Pt = Pw @ T[:3, :3].T + T[:3, 3]
        base = np.array([*np.median(Pt[:, :2], 0), 0.0])          # same object frame as viewer.app
        pts = Pt - base
        img, f = sam_view(k, props, objs_world, cams)
        img.save(out / f"obj_{k}_input.png")
        with torch.no_grad():
            codes = model([img], device="cuda")
            mesh = model.extract_mesh(codes, True, resolution=a.res)[0]
        torch.cuda.empty_cache()
        v, info = align(np.asarray(mesh.vertices), pts)
        mesh = trimesh.Trimesh(v, mesh.faces, vertex_colors=mesh.visual.vertex_colors, process=False)
        mesh.export(out / f"obj_{k}.obj")
        old_fill = out / f"obj_{k}_fill.ply"                   # visuals stay the scan splat
        if old_fill.exists():
            old_fill.unlink()
        rep[k] = {**info, "mesh_vertices": int(len(v)), "frame": f["name"],
                  "mesh_height": float(np.ptp(v[:, 2])), "scan_height": float(np.percentile(pts[:, 2], 98))}
        print(f"obj {k}: {len(v):,} verts, yaw {info['yaw_deg']:.0f}, chamfer {info['chamfer']:.4f}", flush=True)
    (out / "complete3d.json").write_text(json.dumps(rep, indent=1))


if __name__ == "__main__":
    main()
