"""T5-T6: add a Franka Panda to a sim package, choose its base, check top-down reach and grasp width.

Run (WSL): python -m sim.robot <sim_dir> [--units-per-metre 1.0] [--base auto|x,y,yaw_deg]
Writes <sim_dir>/robot/franka_emika_panda/ (Apache-2.0 copy), <sim_dir>/robot_scene.xml,
<sim_dir>/checks/robot_report.json.
The Panda is metric. Scene units are metres * units_per_metre (1.0 until a reference object
calibrates the scene), so reach results are approximate without calibration.
"""
import argparse
import json
import shutil
from pathlib import Path

import mujoco
import numpy as np
from scipy.spatial import ConvexHull

MENAGERIE = Path.home() / "livesplat_data" / "models" / "mujoco_menagerie" / "franka_emika_panda"
TCP = np.array([0.0, 0.0, 0.1034])        # fingertip centre in the hand frame (Panda hand)
GRIP_MAX = 0.08                           # metres, fingers fully open
CLEAR = 0.05                              # pre-grasp height above the object top (metres)
GRASP_DEPTH = 0.025                       # finger pads this far below the object top (palm ends ~3.8 cm above the TCP)
P = "panda/"


def build(sim, base_xyz, yaw, upm):
    """robot_scene.xml = scene.xml + Panda attached at base (scene units scaled by upm)."""
    dst = sim / "robot" / "franka_emika_panda"
    if not dst.exists():
        shutil.copytree(MENAGERIE, dst, ignore=shutil.ignore_patterns("mjx_*", "*.png"))
    scene = mujoco.MjSpec.from_file(str(sim / "scene.xml"))
    panda = mujoco.MjSpec.from_file(str(dst / "panda.xml"))
    # A uniform scale of the robot would change its dynamics; instead the robot stays metric and the
    # caller states how many scene units one metre is (upm). With upm != 1 the scene should be rescaled.
    q = [np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)]
    frame = scene.worldbody.add_frame(pos=list(base_xyz), quat=q)
    frame.attach_body(panda.body("link0"), P, "")
    scene.option.cone = mujoco.mjtCone.mjCONE_ELLIPTIC          # stable grasps (as in Menagerie scenes)
    scene.option.impratio = 10
    absdir = str((dst / "assets").resolve()) + "/"   # compile resolves from the cwd: use absolute paths,
    for mesh in scene.meshes:                       # then write paths relative to the package meshdir
        if mesh.name.startswith(P) and not mesh.file.startswith("/"):
            mesh.file = absdir + Path(mesh.file).name
    m = scene.compile()
    xml = scene.to_xml().replace(absdir, "../robot/franka_emika_panda/assets/")
    (sim / "robot_scene.xml").write_text(xml)
    return m


def arm_index(m):
    names = [f"{P}joint{i}" for i in range(1, 8)]
    qadr = np.array([m.joint(n).qposadr[0] for n in names])
    dadr = np.array([m.joint(n).dofadr[0] for n in names])
    lo, hi = np.array([m.joint(n).range for n in names]).T
    return qadr, dadr, lo, hi


def down_frame(yaw):
    """Hand rotation: z points down, x along yaw. Columns = hand axes in world."""
    z = np.array([0, 0, -1.0])
    x = np.array([np.cos(yaw), np.sin(yaw), 0.0])
    return np.stack([x, np.cross(z, x), z], 1)


def ik(m, d, target, Rd, iters=400, lam=1e-2, step=0.5):
    qadr, dadr, lo, hi = arm_index(m)
    hand = m.body(f"{P}hand").id
    jp, jr = np.zeros((3, m.nv)), np.zeros((3, m.nv))
    for _ in range(iters):
        mujoco.mj_kinematics(m, d)
        mujoco.mj_comPos(m, d)
        R = d.xmat[hand].reshape(3, 3)
        tcp = d.xpos[hand] + R @ TCP
        ep = target - tcp
        er = 0.5 * sum(np.cross(R[:, i], Rd[:, i]) for i in range(3))
        if np.linalg.norm(ep) < 2e-3 and np.linalg.norm(er) < 2e-2:
            break
        mujoco.mj_jac(m, d, jp, jr, tcp, hand)
        J = np.vstack([jp[:, dadr], jr[:, dadr]])
        e = np.r_[ep, er]
        dq = J.T @ np.linalg.solve(J @ J.T + lam * np.eye(6), e)
        d.qpos[qadr] = np.clip(d.qpos[qadr] + step * dq, lo, hi)
    mujoco.mj_kinematics(m, d)
    R = d.xmat[hand].reshape(3, 3)
    tcp = d.xpos[hand] + R @ TCP
    ang = np.degrees(np.arccos(np.clip((np.trace(Rd.T @ R) - 1) / 2, -1, 1)))
    return float(np.linalg.norm(target - tcp)), float(ang)


def grasp_width(hull_pts, quat_wxyz, depth):
    """Narrowest width (and its yaw) of the object slice `depth` below its top, over gripper yaws."""
    from scipy.spatial.transform import Rotation
    w, x, y, z = quat_wxyz
    pts = hull_pts @ Rotation.from_quat([x, y, z, w]).as_matrix().T
    import trimesh
    top = pts[:, 2].max()
    # True cross-section of the hull where the finger pads touch (`depth` below the top).
    sec = trimesh.convex.convex_hull(pts).section(plane_origin=[0, 0, top - depth], plane_normal=[0, 0, 1])
    sl = np.asarray(sec.vertices)[:, :2] if sec is not None and len(sec.vertices) >= 3 else pts[:, :2]
    best = (np.inf, 0.0, np.zeros(2))
    for yaw in np.radians(np.arange(0, 180, 5)):
        a = np.array([-np.sin(yaw), np.cos(yaw)])          # finger axis = hand y for this yaw
        b = np.array([np.cos(yaw), np.sin(yaw)])
        pa, pb = sl @ a, sl @ b
        if np.ptp(pa) < best[0]:
            centre = a * (pa.min() + pa.max()) / 2 + b * (pb.min() + pb.max()) / 2
            best = (np.ptp(pa), yaw, centre)                 # centre: offset from the body origin (xy)
    return best


def reach_report(m, meta, hulls, upm):
    d = mujoco.MjData(m)
    mujoco.mj_resetData(m, d)
    qadr, *_ = arm_index(m)
    home = np.array([0, 0, 0, -1.571, 0, 1.571, -0.785])
    rows = []
    for o in meta["objects"]:
        pos, quat = np.array(o["pose"]["pos"]), o["pose"]["quat_wxyz"]
        top = pos[2] + o["size_xyz"][2]
        width, yaw, ctr = grasp_width(hulls[o["body"]], quat, depth=min(GRASP_DEPTH * upm, 0.5 * o["size_xyz"][2]))
        target = np.array([pos[0] + ctr[0], pos[1] + ctr[1], top + CLEAR * upm])
        d.qpos[qadr] = home
        err, ang = ik(m, d, target, down_frame(yaw))
        ok_reach = err < 0.01 * upm and ang < 5
        rows.append({"object": o["name"], "body": o["body"], "reachable_top_down": bool(ok_reach),
                     "ik_pos_error_m": round(err / upm, 4), "ik_angle_error_deg": round(ang, 1),
                     "grasp_width_m": round(width / upm, 4), "grasp_yaw_deg": round(np.degrees(yaw), 1),
                     "fits_gripper": bool(width / upm <= GRIP_MAX - 0.005),
                     "grasp_centre_offset_xy": np.round(ctr, 5).tolist(),
                     "arm_q": np.round(d.qpos[qadr], 4).tolist()})
    return rows


def edge_distance(poly, ang):
    u = np.array([np.cos(ang), np.sin(ang)])
    return float((poly @ u).max())


def choose_base(sim, meta, hulls, upm, n=24, gap=0.12):
    """Try n base positions around the table edge, facing the centre; keep the best."""
    poly = np.array(meta["table"]["polygon_xy"])
    best = None
    for ang in np.linspace(0, 2 * np.pi, n, endpoint=False):
        rad = edge_distance(poly, ang) + gap * upm
        base = (rad * np.cos(ang), rad * np.sin(ang), 0.0)
        m = build(sim, base, ang + np.pi, upm)
        rows = reach_report(m, meta, hulls, upm)
        score = (sum(r["reachable_top_down"] for r in rows), -sum(r["ik_pos_error_m"] for r in rows))
        if best is None or score > best[0]:
            best = (score, base, ang + np.pi, rows)
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("sim_dir")
    ap.add_argument("--units-per-metre", type=float, default=1.0)
    ap.add_argument("--base", default="auto", help="auto or x,y,yaw_deg (scene units)")
    a = ap.parse_args()
    sim = Path(a.sim_dir)
    meta = json.loads((sim / "metadata.json").read_text())
    upm = a.units_per_metre
    hulls = {}
    for o in meta["objects"]:
        import trimesh
        hulls[o["body"]] = np.asarray(trimesh.load(sim / o["files"]["collision"]).vertices)
    if a.base == "auto":
        score, base, yaw, rows = choose_base(sim, meta, hulls, upm)
    else:
        x, y, yd = (float(v) for v in a.base.split(","))
        base, yaw = (x, y, 0.0), np.radians(yd)
        rows = reach_report(build(sim, base, yaw, upm), meta, hulls, upm)
    build(sim, base, yaw, upm)                              # leave the chosen layout on disk
    rep = {"robot": "franka_emika_panda (MuJoCo Menagerie, Apache-2.0)", "units_per_metre": upm,
           "calibrated": False, "base": {"pos": list(base), "yaw_deg": round(float(np.degrees(yaw)), 1)},
           "tcp_offset_m": TCP.tolist(), "gripper_max_m": GRIP_MAX, "pregrasp_clearance_m": CLEAR,
           "objects": rows,
           "note": "Top-down pre-grasp reach (IK) and finger-width check. Scene scale is not calibrated."}
    (sim / "checks").mkdir(exist_ok=True)
    (sim / "checks" / "robot_report.json").write_text(json.dumps(rep, indent=1))
    print(f"base at ({base[0]:.3f}, {base[1]:.3f}), yaw {np.degrees(yaw):.0f} deg")
    for r in rows:
        print(f"  {r['object']:<20} reach {'YES' if r['reachable_top_down'] else 'no ':<3} "
              f"(err {r['ik_pos_error_m'] * 100:.1f} cm, {r['ik_angle_error_deg']:.0f} deg)  "
              f"width {r['grasp_width_m'] * 100:.1f} cm -> {'fits' if r['fits_gripper'] else 'TOO WIDE'} "
              f"at yaw {r['grasp_yaw_deg']:.0f}")


if __name__ == "__main__":
    main()
