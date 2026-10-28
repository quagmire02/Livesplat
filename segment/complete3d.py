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


def object_view(P_world, cams, pad=0.15):
    """Keyframe where the object is largest and fully inside; crop + mask from its projected points."""
    import cv2
    best = None
    for f in cams["frames"]:
        uv, z = project(P_world, f["K"], f["c2w"])
        uv = uv[z > 0]
        if len(uv) < 100:
            continue
        x0, y0 = uv.min(0)
        x1, y1 = uv.max(0)
        if x0 < 0 or y0 < 0 or x1 >= cams["width"] or y1 >= cams["height"]:
            continue
        if best is None or (x1 - x0) * (y1 - y0) > best[0]:
            best = ((x1 - x0) * (y1 - y0), f, uv)
    _, f, uv = best
    img = np.asarray(Image.open(f["image"]).convert("RGB"))
    mask = np.zeros(img.shape[:2], np.uint8)
    mask[np.clip(uv[:, 1].astype(int), 0, img.shape[0] - 1), np.clip(uv[:, 0].astype(int), 0, img.shape[1] - 1)] = 1
    mask = cv2.morphologyEx(cv2.dilate(mask, np.ones((5, 5), np.uint8)), cv2.MORPH_CLOSE, np.ones((15, 15), np.uint8))
    x0, y0 = uv.min(0).astype(int)
    x1, y1 = uv.max(0).astype(int)
    s = int(max(x1 - x0, y1 - y0) * (1 + 2 * pad))
    cx, cy = (x0 + x1) // 2, (y0 + y1) // 2
    out = np.full((s, s, 3), 127, np.uint8)                      # grey background, as TripoSR expects
    sx0, sy0 = max(0, cx - s // 2), max(0, cy - s // 2)
    sx1, sy1 = min(img.shape[1], cx + s // 2), min(img.shape[0], cy + s // 2)
    m = mask[sy0:sy1, sx0:sx1, None].astype(bool)
    out[sy0 - (cy - s // 2):sy1 - (cy - s // 2), sx0 - (cx - s // 2):sx1 - (cx - s // 2)] = \
        np.where(m, img[sy0:sy1, sx0:sx1], 127)
    return Image.fromarray(out).resize((512, 512)), f


def align(mesh_v, pts):
    """Similarity transform mesh -> object points (object frame, z up, base at z = 0).
    Coarse search: 6 up-axes x 12 yaws, scale from height; then ICP with scale."""
    best = None
    H = np.percentile(pts[:, 2], 98)
    tree = cKDTree(pts)
    ups = {"+x": [0, 1, 0, 0], "-x": [0, -1, 0, 0], "+y": [-1, 0, 0, 0], "-y": [1, 0, 0, 0], "+z": None, "-z": [0, 0, 1, 1]}
    for name, aa in ups.items():
        R0 = np.eye(3)
        if name in ("+x", "-x"):
            R0 = Rotation.from_euler("y", -90 if name == "+x" else 90, degrees=True).as_matrix()
        elif name in ("+y", "-y"):
            R0 = Rotation.from_euler("x", 90 if name == "+y" else -90, degrees=True).as_matrix()
        elif name == "-z":
            R0 = Rotation.from_euler("x", 180, degrees=True).as_matrix()
        for yaw in np.radians(np.arange(0, 360, 30)):
            R = Rotation.from_euler("z", yaw).as_matrix() @ R0
            v = mesh_v @ R.T
            s = H / max(np.ptp(v[:, 2]), 1e-6)
            v = v * s
            v = v - [*np.median(v[:, :2], 0), v[:, 2].min()] + [*np.median(pts[:, :2], 0), 0]
            d = tree.query(v[::max(1, len(v) // 3000)])[0].mean() + cKDTree(v).query(pts[::max(1, len(pts) // 3000)])[0].mean()
            if best is None or d < best[0]:
                best = (d, v, name, yaw)
    d0, v, up, yaw = best
    sub = pts[::max(1, len(pts) // 5000)]
    M, _, cost = trimesh.registration.icp(v, sub, scale=True, max_iterations=40)
    v2 = trimesh.transform_points(v, M)
    return v2, {"up": up, "yaw_deg": float(np.degrees(yaw)), "chamfer_coarse": float(d0), "icp_cost": float(cost)}


def fill_gaussians(mesh, pts, k_sh, gap, n=60000):
    """Surface samples of the generated mesh farther than `gap` from any scanned Gaussian -> raw 3DGS."""
    P, fid = trimesh.sample.sample_surface(mesh, n)
    col = mesh.visual.vertex_colors[mesh.faces[fid, 0], :3] / 255.0 if mesh.visual.kind == "vertex" \
        else np.full((len(P), 3), 0.6)
    far = cKDTree(pts).query(P)[0] > gap
    P, col = P[far].astype(np.float32), col[far].astype(np.float32)
    sc = np.full((len(P), 3), np.log(gap * 0.6), np.float32)
    return {"means": P, "sh0": ((col - 0.5) / splat.SH_C0).astype(np.float32),
            "shN": np.zeros((len(P), k_sh), np.float32), "opacity": np.full(len(P), 2.5, np.float32),
            "scales": sc, "quats": np.tile([1, 0, 0, 0], (len(P), 1)).astype(np.float32)}


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
    for k in range(1, labels.max() + 1):
        sel = (labels == k) & (op > 0.3)
        Pw = raw["means"][sel]
        Pt = Pw @ T[:3, :3].T + T[:3, 3]
        base = np.array([*np.median(Pt[:, :2], 0), 0.0])          # same object frame as viewer.app
        pts = Pt - base
        img, f = object_view(Pw, cams)
        img.save(out / f"obj_{k}_input.png")
        with torch.no_grad():
            codes = model([img], device="cuda")
            mesh = model.extract_mesh(codes, True, resolution=a.res)[0]
        torch.cuda.empty_cache()
        v, info = align(np.asarray(mesh.vertices), pts)
        mesh = trimesh.Trimesh(v, mesh.faces, vertex_colors=mesh.visual.vertex_colors, process=False)
        mesh.export(out / f"obj_{k}.obj")
        gap = 0.012 * al["radius"]
        fill = fill_gaussians(mesh, pts, raw["shN"].shape[1], gap)
        splat.save_raw(out / f"obj_{k}_fill.ply", fill)
        rep[k] = {**info, "mesh_vertices": int(len(v)), "fill_gaussians": int(len(fill["means"])),
                  "frame": f["name"], "mesh_height": float(np.ptp(v[:, 2])), "scan_height": float(np.percentile(pts[:, 2], 98))}
        print(f"obj {k}: {len(v):,} verts, up {info['up']}, icp {info['icp_cost']:.4f}, "
              f"fill {len(fill['means']):,} Gaussians", flush=True)
    (out / "complete3d.json").write_text(json.dumps(rep, indent=1))


if __name__ == "__main__":
    main()
