"""Export a LiveSplat run as a sim package (MJCF + URDF + meshes + splats + metadata).

Run (WSL): python -m sim.export <run_dir> [--src thrift_sfm] [--out <dir>] [--cam-every 8] [--test]
Default out: <run_dir>/sim. --test loads scene.xml in plain MuJoCo and checks that objects settle.
Frame: table frame (Z up, origin = table centre on the top surface). Units: scene units (~m).
"""
import argparse
import json
import shutil
import time
from pathlib import Path

import numpy as np
import trimesh

from scene_io import splat
from viewer.app import Scene, solid_points  # noqa: F401  (Scene builds hulls, props, splats)

FRAME_NOTE = ("Table frame: Z up, origin at the table centre on the top surface. Units are scene units "
              "from MapAnything (approximately metres, NOT calibrated: no reference object in this capture).")


def _f(v):
    return " ".join(f"{x:.6g}" for x in np.ravel(v))


def camera_xml(name, c2w, K, H):
    """OpenCV camera-to-world -> MuJoCo camera (looks along -z, y up)."""
    x, y = c2w[:3, 0], -c2w[:3, 1]
    fovy = np.degrees(2 * np.arctan(H / (2 * K[1, 1])))
    return f'<camera name="{name}" pos="{_f(c2w[:3, 3])}" xyaxes="{_f(x)} {_f(y)}" fovy="{fovy:.4g}"/>'


def inertia_of(hull_pts, mass):
    m = trimesh.convex.convex_hull(hull_pts)
    m.density = mass / max(m.volume, 1e-12)
    return m.center_mass, m.moment_inertia


def urdf(name, mesh_rel, vis_rel, rgba, mass, com, I, mu):
    return f"""<?xml version="1.0"?>
<robot name="{name}">
  <link name="{name}">
    <inertial><origin xyz="{_f(com)}"/><mass value="{mass:.6g}"/>
      <inertia ixx="{I[0, 0]:.6g}" ixy="{I[0, 1]:.6g}" ixz="{I[0, 2]:.6g}" iyy="{I[1, 1]:.6g}" iyz="{I[1, 2]:.6g}" izz="{I[2, 2]:.6g}"/></inertial>
    <visual><geometry><mesh filename="{vis_rel}"/></geometry>
      <material name="{name}_mat"><color rgba="{_f(rgba)}"/></material></visual>
    <collision><geometry><mesh filename="{mesh_rel}"/></geometry></collision>
  </link>
  <!-- friction (Coulomb, vs wood, estimate): {mu:.3g}. Spawn as a floating base at the pose in metadata.json. -->
</robot>
"""


def export(run_dir, src="thrift_sfm", out=None, cam_every=8):
    run = Path(run_dir)
    out = Path(out) if out else run / "sim"
    if out.exists():
        shutil.rmtree(out)
    for d in ("meshes", "urdf", "splats"):
        (out / d).mkdir(parents=True)

    sc = Scene(run, src)
    names = json.loads((run / "names.json").read_text()) if (run / "names.json").exists() else {}
    props = json.loads((run / "props.json").read_text()) if (run / "props.json").exists() else {}
    r = sc.radius

    # ---- meshes ----------------------------------------------------------------
    table_h = 0.03 * r / 0.487 if r else 0.03                     # thin top slab
    poly = sc.polygon
    tv = np.concatenate([np.c_[poly, np.zeros(len(poly))], np.c_[poly, np.full(len(poly), -table_h)]])
    trimesh.convex.convex_hull(tv).export(out / "meshes" / "table.obj")

    # ---- splats (raw values, body frames) --------------------------------------
    labels = np.load(next(p for p in (run / src / "labels.npy", run / "labels.npy", run / "labels_geo.npy")
                          if p.exists()))
    raw = splat.transform(splat.load_raw(run / src / "splat.ply"), sc.T)
    m = raw["means"]
    near = (np.linalg.norm(m[:, :2], axis=1) < 1.6 * r) & (m[:, 2] > -2.5 * r) & (m[:, 2] < 2.0 * r)
    splat.save_raw(out / "splats" / "background.ply", splat.subset(raw, (labels == 0) & near))

    objs_meta, bodies, assets = [], [], []
    for o in sc.objects:
        k = o["id"]
        nm = (names.get(str(k)) or [f"object {k}"])
        tag = f"obj_{k}"
        hull = trimesh.convex.convex_hull(o["hull"])
        hull.export(out / "meshes" / f"{tag}_collision.obj")
        hull.export(out / "meshes" / f"{tag}_visual.obj")
        rgba = [*np.median(o["splat"]["rgb"], 0).round(3), 1.0]
        com, I = inertia_of(o["hull"], o["mass"])
        mu, e = o["friction"], o["bounce"]
        for part, lab in (("", k), ("_reflection", -k)):
            sel = labels == lab
            if sel.any():
                s = splat.subset(raw, sel)
                s["means"] = (s["means"] - o["base"]).astype(np.float32)
                splat.save_raw(out / "splats" / f"{tag}{part}.ply", s)
        (out / "urdf" / f"{tag}.urdf").write_text(
            urdf(tag, f"../meshes/{tag}_collision.obj", f"../meshes/{tag}_visual.obj", rgba, o["mass"], com, I, mu))
        assets += [f'<mesh name="{tag}_col" file="{tag}_collision.obj"/>',
                   f'<mesh name="{tag}_vis" file="{tag}_visual.obj"/>']
        damp = max(0.05, 1.0 - e)
        bodies.append(
            f'<body name="{tag}" pos="{_f(o["base"])}"><freejoint name="{tag}_free"/>'
            f'<geom name="{tag}_col" type="mesh" mesh="{tag}_col" mass="{o["mass"]:.6g}" '
            f'friction="{mu:.4g} 0.005 0.0001" solref="0.004 {damp:.3g}" condim="4" group="3" rgba="{_f(rgba)}"/>'
            f'<geom name="{tag}_vis" type="mesh" mesh="{tag}_vis" contype="0" conaffinity="0" group="1" '
            f'mass="0" rgba="{_f(rgba)}"/></body>')
        p = props.get(str(k), {})
        ext = o["hull"].max(0) - o["hull"].min(0)
        objs_meta.append({
            "id": k, "body": tag, "name": nm[0], "aliases": nm[1:],
            "pose": {"pos": o["base"].tolist(), "quat_wxyz": [1, 0, 0, 0]},
            "size_xyz": ext.round(5).tolist(), "volume": float(hull.volume),
            "mass": {"value": o["mass"], "source": "spark_vision" if "mass_kg" in p else "default_density_x_hull"},
            "friction": {"value": mu, "source": "spark_vision" if "friction" in p else "default"},
            "bounce": {"value": e, "source": "spark_vision" if "bounce" in p else "default"},
            "material": p.get("material"), "hollow": p.get("hollow"),
            "center_of_mass_local": np.asarray(com).round(5).tolist(),
            "files": {"collision": f"meshes/{tag}_collision.obj", "visual": f"meshes/{tag}_visual.obj",
                      "urdf": f"urdf/{tag}.urdf", "splat": f"splats/{tag}.ply"},
            "uncertainty": "collision = convex hull of visible surface; flat footprint (underside not seen); "
                           "mass/friction are estimates",
        })

    # ---- cameras -----------------------------------------------------------------
    cams = json.loads((run / src / "export" / "cameras.json").read_text())
    H = cams["height"]
    cam_xml, cam_meta = [], []
    for i, f in enumerate(cams["frames"]):
        if i % cam_every:
            continue
        c2w = sc.T @ np.array(f["c2w"])
        nm_ = f"cam_{f['name']}"
        cam_xml.append(camera_xml(nm_, c2w, np.array(f["K"]), H))
        cam_meta.append({"name": nm_, "frame": f["name"], "c2w_opencv": c2w.round(6).tolist(),
                         "K": np.array(f["K"]).round(4).tolist(), "width": cams["width"], "height": H})

    xml = f"""<mujoco model="livesplat_{run.name}">
  <compiler meshdir="meshes" angle="radian"/>
  <option timestep="0.002" gravity="0 0 -9.81" integrator="implicitfast"/>
  <visual><global offwidth="{cams['width']}" offheight="{H}"/></visual>
  <asset>
    <mesh name="table" file="table.obj"/>
    {chr(10).join('    ' + a for a in assets).strip()}
  </asset>
  <worldbody>
    <light pos="0 0 {3 * r:.4g}" dir="0 0 -1" diffuse="0.8 0.8 0.8"/>
    <geom name="floor" type="plane" pos="0 0 {sc.floor_z:.6g}" size="0 0 0.05" rgba="0.6 0.6 0.6 1"/>
    <geom name="table" type="mesh" mesh="table" rgba="0.35 0.22 0.12 1" friction="0.6 0.005 0.0001"/>
    {chr(10).join('    ' + b for b in bodies).strip()}
    {chr(10).join('    ' + c for c in cam_xml).strip()}
  </worldbody>
</mujoco>
"""
    (out / "scene.xml").write_text(xml)
    meta = {
        "format": "livesplat-sim/1", "created": time.strftime("%Y-%m-%d %H:%M:%S"),
        "frame": FRAME_NOTE, "units": "scene units (~m, uncalibrated)",
        "conventions": {"quaternion": "wxyz", "camera": "OpenCV c2w (x right, y down, z forward); "
                        "MJCF cameras converted to MuJoCo convention"},
        "table": {"polygon_xy": poly.round(5).tolist(), "top_z": 0.0, "thickness": table_h,
                  "radius": r, "friction": 0.6},
        "floor_z": sc.floor_z,
        "objects": objs_meta,
        "cameras": cam_meta,
        "splats": {"background": "splats/background.ply",
                   "note": "3DGS PLY, raw values (log scale, logit opacity). Object splats are in body frames: "
                           "world = body_pose * point. obj_<k>_reflection.ply = mirror image in a glossy table "
                           "(show only while the object stands at rest)."},
        "provenance": {"run": str(run), "source": src,
                       "build": json.loads((run / "build.json").read_text()) if (run / "build.json").exists() else None},
    }
    (out / "metadata.json").write_text(json.dumps(meta, indent=1))
    (out / "README.md").write_text(PACKAGE_README.format(name=run.name, n=len(objs_meta)))
    return out


PACKAGE_README = """# LiveSplat sim package: {name}
{n} objects on a table, made from one phone video.

Load in MuJoCo:
```python
import mujoco
m = mujoco.MjModel.from_xml_path("scene.xml")
d = mujoco.MjData(m)
```
- Collision geoms are group 3; visual geoms are group 1.
- Cameras `cam_*` are the real capture cameras (use them to compare sim renders with real frames).
- URDFs in urdf/ load each object as a floating body (PyBullet, Isaac, Gazebo). Poses: metadata.json.
- Splats in splats/ give photoreal images of the sim state (render.py, needs gsplat).
- Read metadata.json for units, frames, sources and uncertainty of every value.
"""


def self_test(out, seconds=2.0):
    import mujoco
    m = mujoco.MjModel.from_xml_path(str(out / "scene.xml"))
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    bodies = [m.body(i).name for i in range(m.nbody) if m.body(i).name.startswith("obj_")]
    p0 = {b: d.body(b).xpos.copy() for b in bodies}
    for _ in range(int(seconds / m.opt.timestep)):
        mujoco.mj_step(m, d)
    worst = 0.0
    for b in bodies:
        moved = np.linalg.norm(d.body(b).xpos - p0[b])
        tilt = np.degrees(2 * np.arccos(min(1.0, abs(d.body(b).xquat[0]))))
        worst = max(worst, moved)
        print(f"  {b}: moved {moved * 100:.2f} cm, tilt {tilt:.1f} deg")
    print(f"  {len(bodies)} bodies, {m.ncam} cameras, {m.nmesh} meshes; worst drift {worst * 100:.2f} cm")
    # Camera convention: MuJoCo camera axes must equal OpenCV axes with y and z flipped.
    meta = json.loads((out / "metadata.json").read_text())
    cam_err = 0.0
    for c in meta["cameras"]:
        c2w = np.array(c["c2w_opencv"])
        R = d.cam(c["name"]).xmat.reshape(3, 3)
        cam_err = max(cam_err, np.abs(R - c2w[:3, :3] @ np.diag([1, -1, -1])).max(),
                      np.abs(d.cam(c["name"]).xpos - c2w[:3, 3]).max())
    print(f"  camera convention max error {cam_err:.2e}")
    return worst < 0.02 and cam_err < 1e-4


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("--src", default="thrift_sfm")
    ap.add_argument("--out")
    ap.add_argument("--cam-every", type=int, default=8)
    ap.add_argument("--test", action="store_true")
    a = ap.parse_args()
    out = export(a.run_dir, a.src, a.out, a.cam_every)
    size = sum(f.stat().st_size for f in out.rglob("*") if f.is_file()) / 1e6
    print(f"sim package -> {out} ({size:.0f} MB)")
    if a.test:
        ok = self_test(out)
        print("self-test", "PASS" if ok else "FAIL")


if __name__ == "__main__":
    main()
