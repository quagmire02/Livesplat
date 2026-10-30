"""Camera test: what does a real depth/RGB sensor see from a given pose on this scanned table?

Per object (in the scan's table frame, metric via measure.scale):
  visible   share of its camera-facing surface that is inside the image and not hidden (z-buffer against all
            objects and the table/room splats), so occlusion by other objects counts
  pixels    size of the visible object in the image (short side, px)
  distance  median depth (m) and whether it is inside the sensor's working range
  depth_err expected depth noise there: stereo sigma_z = z^2 * sigma_d / (f * b); ToF a + b * z
Score per object = visible * min(1, pixels / 64) * in_range; the view's score is the weakest object's.
Auto placement searches 108 poses (12 azimuths x 3 elevations x 3 distances) looking at the objects' centre.
Also writes a synthetic depth map (sensor noise model) and an object-id mask for dataset export.

CLI: python -m lab.camera <run> [--camera realsense_d435] [--auto] [--pose x,y,z,lx,ly,lz (metres, table frame)]
"""
import argparse
import json
from pathlib import Path

import numpy as np
import yaml

from scene_io.thrift import project
from segment.sam_vote import zbuffer_visible

CAMERAS = Path(__file__).parent / "cameras"
MIN_PX = 64                                    # pixels across an object for reliable detection


def load_camera(name):
    p = Path(name)
    if not p.suffix:
        p = CAMERAS / f"{name}.yaml"
    return {**yaml.safe_load(p.read_text()), "id": p.stem}


def intrinsics(cam):
    W, H = cam["resolution"]
    f = W / 2 / np.tan(np.radians(cam["hfov_deg"]) / 2)
    return np.array([[f, 0, W / 2], [0, f, H / 2], [0, 0, 1.0]]), W, H, f


def look_at(pos, target):
    """camera-to-world 4x4, OpenCV axes (x right, y down, z forward), world z up."""
    z = np.asarray(target, float) - pos
    z /= np.linalg.norm(z)
    x = np.cross(z, [0, 0, 1.0])
    if np.linalg.norm(x) < 1e-6:
        x = np.array([1.0, 0, 0])
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    c2w = np.eye(4)
    c2w[:3, :3] = np.stack([x, y, z], 1)
    c2w[:3, 3] = pos
    return c2w


def scene_points(sc, run, n_obj=6000, n_bg=40000):
    """Object surface points + normals (table frame, current poses = capture poses) and background occluders."""
    from lab.gripper import _dense
    objs = {}
    rng = np.random.default_rng(0)
    for o in sc.objects:
        v, n = _dense(run, o["id"], o)
        i = rng.choice(len(v), size=min(n_obj, len(v)), replace=False)
        objs[o["id"]] = (v[i] + o["base"], n[i])
    bg = sc.bg["means"][sc.bg["opacity"] > 0.5]
    bg = bg[np.linalg.norm(bg[:, :2], axis=1) < 2.0 * sc.radius]
    bg = bg[rng.choice(len(bg), size=min(n_bg, len(bg)), replace=False)]
    return objs, bg


def evaluate(cam, c2w, objs, bg, mpu, names):
    K, W, H, f = intrinsics(cam)
    ids = list(objs)
    allp = np.concatenate([objs[k][0] for k in ids] + [bg])
    uv, z = project(allp, K, c2w)
    vis, _ = zbuffer_visible(uv, z, W, H, cell=max(2, W // 160), tol=0.02)
    inside = (z > 0) & (uv[:, 0] >= 0) & (uv[:, 1] >= 0) & (uv[:, 0] < W) & (uv[:, 1] < H)
    out, i0 = {}, 0
    for k in ids:
        p, n = objs[k]
        sl = slice(i0, i0 + len(p))
        i0 += len(p)
        facing = (n * (c2w[:3, 3] - p)).sum(1) > 0                # camera-facing surface (normals)
        if facing.sum() < 10:
            facing = np.ones(len(p), bool)
        v = vis[sl] & inside[sl] & facing
        frac = float(v.sum() / max(facing.sum(), 1))
        if v.sum() >= 5:
            box = uv[sl][v]
            px = float(min(np.ptp(box[:, 0]), np.ptp(box[:, 1])))
            dist = float(np.median(z[sl][v]) * mpu)
        else:
            px, dist = 0.0, float(np.median(z[sl]) * mpu)
        in_range = cam["min_m"] <= dist <= cam["max_m"]
        if cam["depth"] == "stereo":
            sig = dist ** 2 * cam["subpixel_px"] / (f * cam["baseline_m"])
        elif cam["depth"] == "tof":
            sig = cam["tof_sigma_m"] + cam["tof_sigma_rel"] * dist
        else:
            sig = None
        blockers = []
        if frac < 0.6:                                              # who hides it: other objects in front
            hidden = inside[sl] & facing & ~vis[sl]
            if hidden.any():
                for j in ids:
                    if j == k:
                        continue
                    pj = objs[j][0]
                    uvj, zj = project(pj, K, c2w)
                    if ((zj > 0).any() and np.median(zj) < np.median(z[sl]) and
                            np.abs(np.median(uvj, 0) - np.median(uv[sl][hidden], 0)).max() < 0.25 * W):
                        blockers.append(names.get(j, f"object {j}"))
        score = frac * min(1.0, px / MIN_PX) * float(in_range)
        out[str(k)] = {"name": names.get(k, f"object {k}"), "visible": round(frac, 2), "pixels": int(px),
                       "distance_m": round(dist, 3), "in_range": in_range,
                       "depth_sigma_mm": round(sig * 1000, 1) if sig is not None else None,
                       "hidden_by": blockers, "score": round(score, 2)}
    return {"score": round(min(v["score"] for v in out.values()), 2), "objects": out}


def auto_place(cam, objs, bg, mpu, names, sc):
    centre = np.mean([p.mean(0) for p, _ in objs.values()], 0)
    best = None
    for dist in (0.45, 0.8, 1.2):
        for el in (25, 45, 65):
            for az in range(0, 360, 30):
                a, e = np.radians(az), np.radians(el)
                pos = centre + dist / mpu * np.array([np.cos(e) * np.cos(a), np.cos(e) * np.sin(a), np.sin(e)])
                c2w = look_at(pos, centre)
                r = evaluate(cam, c2w, objs, bg, mpu, names)
                key = (r["score"], np.mean([v["score"] for v in r["objects"].values()]))
                if best is None or key > best[0]:
                    best = (key, c2w, r, {"azimuth_deg": az, "elevation_deg": el, "distance_m": dist})
    return best[1], best[2], best[3]


def depth_and_mask(cam, c2w, objs, bg, mpu, rng=None):
    """Synthetic sensor depth (metres, noise model, 0 = no return) and object-id mask at 1/4 resolution."""
    K, W, H, f = intrinsics(cam)
    s = 4
    K4 = K.copy()
    K4[:2] /= s
    W4, H4 = W // s, H // s
    ids = list(objs)
    pts = np.concatenate([objs[k][0] for k in ids] + [bg])
    lab = np.concatenate([np.full(len(objs[k][0]), k) for k in ids] + [np.zeros(len(bg), int)])
    uv, z = project(pts, K4, c2w)
    ok = (z > 0) & (uv[:, 0] >= 0) & (uv[:, 1] >= 0) & (uv[:, 0] < W4) & (uv[:, 1] < H4)
    u, v, zz, ll = uv[ok, 0].astype(int), uv[ok, 1].astype(int), z[ok] * mpu, lab[ok]
    order = np.argsort(-zz)                                         # far first, near overwrite
    depth = np.zeros((H4, W4), np.float32)
    mask = np.zeros((H4, W4), np.uint8)
    depth[v[order], u[order]] = zz[order]
    mask[v[order], u[order]] = ll[order]
    import cv2
    holes = depth == 0
    depth = np.where(holes, cv2.dilate(depth, np.ones((3, 3), np.uint8)), depth)
    mask = np.where(mask == 0, cv2.dilate(mask, np.ones((3, 3), np.uint8)), mask)
    rng = rng or np.random.default_rng(0)
    if cam["depth"] == "stereo":
        sig = depth ** 2 * cam["subpixel_px"] / (f * cam["baseline_m"])
    elif cam["depth"] == "tof":
        sig = cam["tof_sigma_m"] + cam["tof_sigma_rel"] * depth
    else:
        return None, mask
    noisy = depth + rng.normal(size=depth.shape) * sig
    noisy[(depth < cam["min_m"]) | (depth > cam["max_m"])] = 0                     # out of range: no return
    return noisy.astype(np.float32), mask


def suggest(res, cam, auto_res=None):
    weak = [v for v in res["objects"].values() if v["score"] < 0.5]
    tips = []
    for v in weak:
        if not v["in_range"]:
            tips.append(f"{v['name']} is at {v['distance_m']} m, outside {cam['min_m']}-{cam['max_m']} m: "
                        + ("move the camera back." if v["distance_m"] < cam["min_m"] else "move closer."))
        elif v["hidden_by"]:
            tips.append(f"{v['name']} is {int((1 - v['visible']) * 100)}% hidden by {', '.join(v['hidden_by'])}: "
                        "raise the camera or move it to the side.")
        elif v["pixels"] < MIN_PX:
            tips.append(f"{v['name']} is only {v['pixels']} px across: move closer or use a narrower lens.")
        else:
            tips.append(f"{v['name']}: only {int(v['visible'] * 100)}% visible from here.")
    if auto_res is not None and auto_res["score"] > res["score"] + 0.1:
        tips.append(f"Best position found: score {auto_res['score']} vs {res['score']} here.")
    return tips


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("--src", default="thrift_sfm")
    ap.add_argument("--camera", nargs="+", default=["realsense_d435", "realsense_d455", "femto_bolt"])
    a = ap.parse_args()
    from measure.scale import load
    from viewer.app import Scene
    sc = Scene(a.run_dir, a.src)
    mpu = load(a.run_dir).get("metres_per_unit") or 0.6
    names = {int(k): v[0] for k, v in json.loads((Path(a.run_dir) / "names.json").read_text()).items()}
    objs, bg = scene_points(sc, a.run_dir)
    d = Path(a.run_dir) / "lab"
    d.mkdir(exist_ok=True)
    for c in a.camera:
        cam = load_camera(c)
        c2w, res, pose = auto_place(cam, objs, bg, mpu, names, sc)
        res.update(camera=cam["name"], pose=pose, c2w=np.round(c2w, 5).tolist(), suggestions=suggest(res, cam))
        (d / f"camera_{cam['id']}.json").write_text(json.dumps(res, indent=1))
        print(f"{cam['name']}: best pose {pose}, score {res['score']}")
        for v in res["objects"].values():
            print(f"  {v['name']}: visible {int(v['visible'] * 100)}%, {v['pixels']} px, {v['distance_m']} m, "
                  f"depth sigma {v['depth_sigma_mm']} mm")


if __name__ == "__main__":
    main()
