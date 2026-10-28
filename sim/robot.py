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


SIDE_MIN_H = 0.045                       # side grasp: lowest pad height above the table (hand clearance)
SIDE_MAX_IN = 0.035                      # fingertip centre at most this far past the near surface (palm)


def side_frame(approach):
    """Hand rotation for a horizontal approach: z along `approach`, fingers close horizontally."""
    z = np.array([approach[0], approach[1], 0.0])
    z /= np.linalg.norm(z)
    y = np.array([-z[1], z[0], 0.0])
    return np.stack([np.cross(y, z), y, z], 1)


def robot_collisions(m, d, q, ignore=()):
    """Names of non-robot bodies the arm touches at joint config q (objects stay at their poses)."""
    qadr, *_ = arm_index(m)
    d.qpos[qadr] = q
    mujoco.mj_forward(m, d)
    hits = set()
    for c in d.contact[:d.ncon]:
        n1, n2 = m.body(m.geom_bodyid[c.geom1]).name, m.body(m.geom_bodyid[c.geom2]).name
        r1, r2 = n1.startswith(P), n2.startswith(P)
        if r1 != r2:
            other = n2 if r1 else n1
            if other not in ignore:
                hits.add(other or "table/floor")
    return sorted(hits)


def side_plan(hull_pts, pos, quat_wxyz, approach, upm):
    """Best side grasp: height, width across the fingers, TCP point. None if nothing fits."""
    import trimesh
    from scipy.spatial.transform import Rotation
    w, x, y, z = quat_wxyz
    mesh = trimesh.convex.convex_hull(hull_pts @ Rotation.from_quat([x, y, z, w]).as_matrix().T)
    H = mesh.bounds[1, 2]
    a = np.array(approach[:2]) / np.linalg.norm(approach[:2])
    perp = np.array([-a[1], a[0]])
    best = None
    for h in np.arange(max(SIDE_MIN_H * upm, 0.15 * H), H - 0.02 * upm, 0.01 * upm):
        sec = mesh.section(plane_origin=[0, 0, h], plane_normal=[0, 0, 1])
        if sec is None or len(sec.vertices) < 3:
            continue
        xy = np.asarray(sec.vertices)[:, :2]
        pa, pp = xy @ a, xy @ perp
        width = np.ptp(pp)
        if width / upm > GRIP_MAX - 0.005:
            continue
        along = min((pa.min() + pa.max()) / 2, pa.min() + SIDE_MAX_IN * upm)
        ctr = a * along + perp * (pp.min() + pp.max()) / 2
        score = abs(h - 0.4 * H)
        if best is None or score < best["score"]:
            best = {"score": score, "height": float(h), "width": float(width),
                    "tcp": [pos[0] + ctr[0], pos[1] + ctr[1], pos[2] + h], "approach": a.tolist()}
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
        q_top = d.qpos[qadr].copy()
        top_block = robot_collisions(m, d, q_top) if ok_reach else []
        # Top-down at the grasp point (fingers open around the object: the target itself may touch).
        base_xy = d.body(f"{P}link0").xpos[:2]
        base_dir = pos[:2] - base_xy
        side_row, blocked_by = {"feasible": False}, set()
        for dang in np.radians([0, -20, 20, -40, 40, -60, 60]):
            c_, s_ = np.cos(dang), np.sin(dang)
            appr = np.array([c_ * base_dir[0] - s_ * base_dir[1], s_ * base_dir[0] + c_ * base_dir[1]])
            side = side_plan(hulls[o["body"]], pos, quat, appr, upm)
            if not side:
                continue
            Rd = side_frame(side["approach"])
            a3 = np.array([*side["approach"], 0.0])
            d.qpos[qadr] = home
            e_p, a_p = ik(m, d, np.array(side["tcp"]) - 0.12 * upm * a3, Rd)
            q_pre = d.qpos[qadr].copy()
            e_s, a_s = ik(m, d, np.array(side["tcp"]), Rd)
            q_at = d.qpos[qadr].copy()
            ok_ik = max(e_p, e_s) < 0.01 * upm and max(a_p, a_s) < 5
            hits = sorted(set(robot_collisions(m, d, q_pre)) | set(robot_collisions(m, d, q_at, ignore=(o["body"],))))
            cand = {"feasible": bool(ok_ik and not hits), "height_m": round(side["height"] / upm, 4),
                    "width_m": round(side["width"] / upm, 4), "tcp": np.round(side["tcp"], 5).tolist(),
                    "approach_xy": np.round(side["approach"], 4).tolist(), "approach_offset_deg": round(float(np.degrees(dang))),
                    "ik_pos_error_m": round(e_s / upm, 4), "ik_angle_error_deg": round(a_s, 1), "blocked_by": hits}
            blocked_by |= set(hits)
            if cand["feasible"] and (not side_row["feasible"] or cand["width_m"] < side_row["width_m"]):
                side_row = cand
            elif not side_row["feasible"] and "height_m" not in side_row:
                side_row = cand
        if not side_row["feasible"] and blocked_by:
            side_row["blocked_by"] = sorted(blocked_by)
        d.qpos[qadr] = q_top
        rows.append({"object": o["name"], "body": o["body"], "reachable_top_down": bool(ok_reach and not top_block),
                     "top_blocked_by": top_block, "side": side_row,
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
        s = r["side"]
        print(f"  {'':<20} side: " + (f"{'YES' if s['feasible'] else 'no'} at {s['height_m'] * 100:.0f} cm high, "
                                      f"width {s['width_m'] * 100:.1f} cm" + (f", blocked by {s['blocked_by']}" if s.get('blocked_by') else '')
                                      if "height_m" in s else "nothing fits the gripper"))


if __name__ == "__main__":
    main()
