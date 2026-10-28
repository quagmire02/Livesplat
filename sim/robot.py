"""Simulation mode: add a robot arm (any sim/robots/<name>.yaml) to a sim package, choose its base,
check top-down and side grasps with collision checks.

Run (WSL): python -m sim.robot <sim_dir> [--robot panda|so100|path.yaml] [--units-per-metre 1.0]
                                         [--base auto|x,y,yaw_deg]
Writes <sim_dir>/robot/<name>/ (model copy + licence), robot_scene_<name>.xml (+ robot_scene.xml),
checks/robot_report_<name>.json (+ robot_report.json).
Robots are metric. Scene units = metres * units_per_metre (1.0 until a reference object calibrates
the scene), so results are approximate without calibration.
"""
import argparse
import json
import shutil
from pathlib import Path

import mujoco
import numpy as np

from sim.robot_spec import RobotSpec

CLEAR = 0.05                  # pre-grasp height above the object top (metres)
SIDE_MIN_H = 0.045            # side grasp: lowest pad height above the table (hand clearance)


# ---- scene ------------------------------------------------------------------------------------
def build(sim, spec, base_xyz, yaw):
    """robot_scene_<name>.xml = scene.xml + the robot attached at base_xyz, facing yaw."""
    dst = sim / "robot" / spec.name
    if not dst.exists():
        shutil.copytree(spec.model.parent, dst, ignore=shutil.ignore_patterns("mjx_*", "*.png"))
    scene = mujoco.MjSpec.from_file(str(sim / "scene.xml"))
    child = mujoco.MjSpec.from_file(str(dst / spec.model.name))
    q = [np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)]
    frame = scene.worldbody.add_frame(pos=list(base_xyz), quat=q)
    frame.attach_body(child.body(spec.root_body), spec.prefix, "")
    scene.option.cone = mujoco.mjtCone.mjCONE_ELLIPTIC              # stable grasps (as in Menagerie scenes)
    scene.option.impratio = 10
    meshdir = dst / (child.meshdir or ".")
    absdir = str(meshdir.resolve()) + "/"                          # compile resolves from the cwd: absolute
    for mesh in scene.meshes:                                      # paths, then relative ones in the file
        if mesh.name.startswith(spec.prefix) and not mesh.file.startswith("/"):
            mesh.file = absdir + mesh.file
    m = scene.compile()
    xml = scene.to_xml().replace(absdir, f"../robot/{spec.name}/{child.meshdir or '.'}/".replace("/./", "/"))
    (sim / f"robot_scene_{spec.name}.xml").write_text(xml)
    (sim / "robot_scene.xml").write_text(xml)
    return m


def arm_index(m, spec):
    names = [spec.prefix + n for n in spec.arm_joints]
    qadr = np.array([m.joint(n).qposadr[0] for n in names])
    dadr = np.array([m.joint(n).dofadr[0] for n in names])
    lo, hi = np.array([m.joint(n).range for n in names]).T
    return qadr, dadr, lo, hi


def tcp_pose(m, d, spec):
    b = m.body(spec.prefix + spec.tcp_body).id
    R = d.xmat[b].reshape(3, 3)
    return b, R, d.xpos[b] + R @ spec.tcp_offset


# ---- grasp directions (world) -----------------------------------------------------------------
def top_dirs(yaw):
    """Top-down: approach straight down; jaws close along the horizontal direction at `yaw`."""
    return np.array([0, 0, -1.0]), np.array([-np.sin(yaw), np.cos(yaw), 0.0])


def side_dirs(approach_xy):
    a = np.array([approach_xy[0], approach_xy[1], 0.0])
    a /= np.linalg.norm(a)
    return a, np.array([-a[1], a[0], 0.0])


def ik(m, d, spec, target, approach_w, close_w, iters=400, lam=1e-2, step=0.5):
    """Damped least squares on the TCP. full: position + approach + close axes. axis (5-DoF arms):
    position + approach axis; the close axis is only a soft preference."""
    qadr, dadr, lo, hi = arm_index(m, spec)
    wc = 1.0 if spec.ik == "full" else 0.0       # 5-DoF: no freedom left for the close axis
    wr = 1.0 if spec.ik == "full" else 0.3       # 5-DoF: position wins over the approach direction
    jp, jr = np.zeros((3, m.nv)), np.zeros((3, m.nv))
    for _ in range(iters):
        mujoco.mj_kinematics(m, d)
        mujoco.mj_comPos(m, d)
        b, R, tcp = tcp_pose(m, d, spec)
        ep = target - tcp
        er = wr * (np.cross(R @ spec.approach_axis, approach_w) + wc * np.cross(R @ spec.close_axis, close_w))
        if np.linalg.norm(ep) < 2e-3 and np.linalg.norm(er) < 2e-2:
            break
        mujoco.mj_jac(m, d, jp, jr, tcp, b)
        J = np.vstack([jp[:, dadr], jr[:, dadr]])
        dq = J.T @ np.linalg.solve(J @ J.T + lam * np.eye(6), np.r_[ep, er])
        d.qpos[qadr] = np.clip(d.qpos[qadr] + step * dq, lo, hi)
    mujoco.mj_kinematics(m, d)
    _, R, tcp = tcp_pose(m, d, spec)
    ang = lambda u, v: np.degrees(np.arccos(np.clip(u @ v, -1, 1)))
    a_err = ang(R @ spec.approach_axis, approach_w)
    c_err = min(ang(R @ spec.close_axis, close_w), ang(R @ spec.close_axis, -close_w))
    return float(np.linalg.norm(target - tcp)), float(a_err if spec.ik != "full" else max(a_err, c_err))


def ik_multi(m, d, spec, target, approach_w, close_w, seeds=3, rng=None):
    """IK from the current pose, then from random seeds; keeps the best. Avoids local minima (5-DoF arms)."""
    qadr, _, lo, hi = arm_index(m, spec)
    rng = rng or np.random.default_rng(0)
    q0 = d.qpos[qadr].copy()
    best = ik(m, d, spec, target, approach_w, close_w) + (d.qpos[qadr].copy(),)
    for _ in range(seeds):
        if best[0] < 2e-3 and best[1] < 2:
            break
        d.qpos[qadr] = np.clip(q0 + rng.normal(0, 0.6, len(q0)), lo, hi)
        r = ik(m, d, spec, target, approach_w, close_w) + (d.qpos[qadr].copy(),)
        if r[0] + r[1] / 200 < best[0] + best[1] / 200:
            best = r
    d.qpos[qadr] = best[2]
    return best[0], best[1]


# ---- geometry ---------------------------------------------------------------------------------
def _posed_mesh(hull_pts, quat_wxyz):
    import trimesh
    from scipy.spatial.transform import Rotation
    w, x, y, z = quat_wxyz
    return trimesh.convex.convex_hull(hull_pts @ Rotation.from_quat([x, y, z, w]).as_matrix().T)


def width_along(hull_pts, quat_wxyz, depth, close_xy):
    """Object width along a given horizontal closing direction, `depth` below the top."""
    mesh = _posed_mesh(hull_pts, quat_wxyz)
    sec = mesh.section(plane_origin=[0, 0, mesh.bounds[1, 2] - depth], plane_normal=[0, 0, 1])
    sl = np.asarray(sec.vertices)[:, :2] if sec is not None and len(sec.vertices) >= 3 else np.asarray(mesh.vertices)[:, :2]
    c = np.asarray(close_xy, float) / max(np.linalg.norm(close_xy), 1e-9)
    return float(np.ptp(sl @ c))


def top_plan(hull_pts, quat_wxyz, depth):
    """Narrowest width over jaw yaws of the true cross-section `depth` below the top."""
    mesh = _posed_mesh(hull_pts, quat_wxyz)
    top = mesh.bounds[1, 2]
    sec = mesh.section(plane_origin=[0, 0, top - depth], plane_normal=[0, 0, 1])
    sl = np.asarray(sec.vertices)[:, :2] if sec is not None and len(sec.vertices) >= 3 else np.asarray(mesh.vertices)[:, :2]
    best = (np.inf, 0.0, np.zeros(2))
    for yaw in np.radians(np.arange(0, 180, 5)):
        c = np.array([-np.sin(yaw), np.cos(yaw)])            # closing direction
        b = np.array([np.cos(yaw), np.sin(yaw)])
        pc, pb = sl @ c, sl @ b
        if np.ptp(pc) < best[0]:
            best = (np.ptp(pc), yaw, c * (pc.min() + pc.max()) / 2 + b * (pb.min() + pb.max()) / 2)
    return best


def side_plan(spec, hull_pts, pos, quat_wxyz, approach, upm):
    """Best side grasp: height, width across the jaws, TCP point. None if nothing fits."""
    mesh = _posed_mesh(hull_pts, quat_wxyz)
    H = mesh.bounds[1, 2]
    a = np.array(approach[:2]) / np.linalg.norm(approach[:2])
    perp = np.array([-a[1], a[0]])
    best = None
    for h in np.arange(max(spec.side_min_height * upm, 0.15 * H), H - 0.01 * upm, 0.01 * upm):
        sec = mesh.section(plane_origin=[0, 0, h], plane_normal=[0, 0, 1])
        if sec is None or len(sec.vertices) < 3:
            continue
        xy = np.asarray(sec.vertices)[:, :2]
        pa, pp = xy @ a, xy @ perp
        width = np.ptp(pp)
        if width / upm > spec.fit_width:
            continue
        along = min((pa.min() + pa.max()) / 2, pa.min() + spec.side_max_in * upm)
        ctr = a * along + perp * (pp.min() + pp.max()) / 2
        score = abs(h - 0.4 * H)
        if best is None or score < best["score"]:
            best = {"score": score, "height": float(h), "width": float(width),
                    "tcp": [pos[0] + ctr[0], pos[1] + ctr[1], pos[2] + h], "approach": a.tolist()}
    return best


def robot_collisions(m, d, spec, q, ignore=()):
    """Names of non-robot bodies the arm touches at joint config q (objects stay at their poses)."""
    qadr, *_ = arm_index(m, spec)
    d.qpos[qadr] = q
    mujoco.mj_forward(m, d)
    hits = set()
    for c in d.contact[:d.ncon]:
        n1, n2 = m.body(m.geom_bodyid[c.geom1]).name, m.body(m.geom_bodyid[c.geom2]).name
        r1, r2 = n1.startswith(spec.prefix), n2.startswith(spec.prefix)
        if spec.prefix + spec.root_body in (n1, n2):            # the mounted base touches its support
            continue
        if r1 != r2:
            other = n2 if r1 else n1
            if other not in ignore:
                hits.add(other or "table/floor")
    return sorted(hits)


# ---- report -----------------------------------------------------------------------------------
def reach_report(m, spec, meta, hulls, upm):
    d = mujoco.MjData(m)
    mujoco.mj_resetData(m, d)
    qadr, *_ = arm_index(m, spec)
    rows = []
    for o in meta["objects"]:
        pos, quat = np.array(o["pose"]["pos"]), o["pose"]["quat_wxyz"]
        top = pos[2] + o["size_xyz"][2]
        depth = min(spec.top_depth * upm, 0.5 * o["size_xyz"][2])
        width, yaw, ctr = top_plan(hulls[o["body"]], quat, depth)
        target = np.array([pos[0] + ctr[0], pos[1] + ctr[1], top + CLEAR * upm])
        d.qpos[qadr] = spec.home
        err, ang = ik_multi(m, d, spec, target, *top_dirs(yaw))
        ok_reach = err < 0.01 * upm and ang < spec.approach_tol
        q_top = d.qpos[qadr].copy()
        if spec.ik != "full":                                   # jaw yaw is whatever the arm reached:
            _, R_, _ = tcp_pose(m, d, spec)                     # re-measure the width along it
            c_ = R_ @ spec.close_axis
            if np.linalg.norm(c_[:2]) > 0.3:
                width = width_along(hulls[o["body"]], quat, depth, c_[:2])
                yaw = float(np.arctan2(-c_[0], c_[1]))
        top_block = robot_collisions(m, d, spec, q_top) if ok_reach else []

        base_xy = d.body(spec.prefix + spec.root_body).xpos[:2]
        base_dir = pos[:2] - base_xy
        side_row, blocked_by = {"feasible": False}, set()
        for dang in np.radians([0, -20, 20, -40, 40, -60, 60]):
            c_, s_ = np.cos(dang), np.sin(dang)
            side = side_plan(spec, hulls[o["body"]], pos, quat,
                             [c_ * base_dir[0] - s_ * base_dir[1], s_ * base_dir[0] + c_ * base_dir[1]], upm)
            if not side:
                continue
            aw, cw = side_dirs(side["approach"])
            d.qpos[qadr] = spec.home
            e_p, a_p = ik_multi(m, d, spec, np.array(side["tcp"]) - 0.12 * upm * aw, aw, cw)
            q_pre = d.qpos[qadr].copy()
            e_s, a_s = ik_multi(m, d, spec, np.array(side["tcp"]), aw, cw)
            q_at = d.qpos[qadr].copy()
            ok_ik = max(e_p, e_s) < 0.01 * upm and max(a_p, a_s) < spec.approach_tol
            hits = sorted(set(robot_collisions(m, d, spec, q_pre)) |
                          set(robot_collisions(m, d, spec, q_at, ignore=(o["body"],))))
            cand = {"feasible": bool(ok_ik and not hits), "height_m": round(side["height"] / upm, 4),
                    "width_m": round(side["width"] / upm, 4), "tcp": np.round(side["tcp"], 5).tolist(),
                    "approach_xy": np.round(side["approach"], 4).tolist(),
                    "approach_offset_deg": round(float(np.degrees(dang))),
                    "ik_pos_error_m": round(e_s / upm, 4), "ik_angle_error_deg": round(a_s, 1), "blocked_by": hits}
            blocked_by |= set(hits)
            if cand["feasible"] and (not side_row["feasible"] or cand["width_m"] < side_row["width_m"]):
                side_row = cand
            elif not side_row["feasible"] and "height_m" not in side_row:
                side_row = cand
        if not side_row["feasible"] and blocked_by:
            side_row["blocked_by"] = sorted(blocked_by)
        rows.append({"object": o["name"], "body": o["body"], "reachable_top_down": bool(ok_reach and not top_block),
                     "top_blocked_by": top_block, "side": side_row,
                     "ik_pos_error_m": round(err / upm, 4), "ik_angle_error_deg": round(ang, 1),
                     "grasp_width_m": round(width / upm, 4), "grasp_yaw_deg": round(np.degrees(yaw), 1),
                     "fits_gripper": bool(width / upm <= spec.fit_width),
                     "grasp_centre_offset_xy": np.round(ctr, 5).tolist(), "arm_q": np.round(q_top, 4).tolist()})
    return rows


def edge_distance(poly, ang):
    u = np.array([np.cos(ang), np.sin(ang)])
    return float((poly @ u).max())


def choose_base(sim, spec, meta, hulls, upm, n=24):
    """Try n base positions around the table edge, facing the centre; keep the one that can grasp most."""
    poly = np.array(meta["table"]["polygon_xy"])
    best = None
    for ang in np.linspace(0, 2 * np.pi, n, endpoint=False):
        rad = edge_distance(poly, ang) + spec.base_gap * upm
        base = (rad * np.cos(ang), rad * np.sin(ang), 0.0)
        rows = reach_report(build(sim, spec, base, ang + np.pi), spec, meta, hulls, upm)
        ok = sum((r["reachable_top_down"] and r["fits_gripper"]) or r["side"]["feasible"] for r in rows)
        score = (ok, sum(r["reachable_top_down"] for r in rows), -sum(r["ik_pos_error_m"] for r in rows))
        if best is None or score > best[0]:
            best = (score, base, ang + np.pi, rows)
    return best


def main():
    import trimesh
    ap = argparse.ArgumentParser()
    ap.add_argument("sim_dir")
    ap.add_argument("--robot", default="panda", help="name in sim/robots/ or a path to a robot yaml")
    ap.add_argument("--units-per-metre", type=float, default=1.0)
    ap.add_argument("--base", default="auto", help="auto or x,y,yaw_deg (scene units)")
    a = ap.parse_args()
    sim = Path(a.sim_dir)
    spec = RobotSpec.load(a.robot)
    meta = json.loads((sim / "metadata.json").read_text())
    upm = a.units_per_metre
    hulls = {o["body"]: np.asarray(trimesh.load(sim / o["files"]["collision"]).vertices) for o in meta["objects"]}
    if a.base == "auto":
        _, base, yaw, rows = choose_base(sim, spec, meta, hulls, upm)
    else:
        x, y, yd = (float(v) for v in a.base.split(","))
        base, yaw = (x, y, 0.0), np.radians(yd)
        rows = reach_report(build(sim, spec, base, yaw), spec, meta, hulls, upm)
    build(sim, spec, base, yaw)                              # leave the chosen layout on disk
    rep = {"robot": spec.name, "robot_yaml": str(a.robot), "units_per_metre": upm, "calibrated": False,
           "base": {"pos": list(base), "yaw_deg": round(float(np.degrees(yaw)), 1)},
           "gripper": {"type": spec.gripper_type, "max_opening_m": spec.max_opening},
           "pregrasp_clearance_m": CLEAR, "objects": rows,
           "note": "Top-down and side grasp planning (IK + collision check). Scene scale is not calibrated."}
    (sim / "checks").mkdir(exist_ok=True)
    for f in (f"robot_report_{spec.name}.json", "robot_report.json"):
        (sim / "checks" / f).write_text(json.dumps(rep, indent=1))
    print(f"{spec.name}: base at ({base[0]:.3f}, {base[1]:.3f}), yaw {np.degrees(yaw):.0f} deg")
    for r in rows:
        print(f"  {r['object']:<20} top: reach {'YES' if r['reachable_top_down'] else 'no '} "
              f"(err {r['ik_pos_error_m'] * 100:.1f} cm, {r['ik_angle_error_deg']:.0f} deg), "
              f"width {r['grasp_width_m'] * 100:.1f} cm -> {'fits' if r['fits_gripper'] else 'TOO WIDE'}"
              + (f", blocked by {r['top_blocked_by']}" if r["top_blocked_by"] else ""))
        s = r["side"]
        print(f"  {'':<20} side: " + (f"{'YES' if s['feasible'] else 'no'} at {s['height_m'] * 100:.0f} cm high, "
                                      f"width {s['width_m'] * 100:.1f} cm"
                                      + (f", blocked by {s['blocked_by']}" if s.get("blocked_by") else "")
                                      if "height_m" in s else "nothing fits the gripper"))


if __name__ == "__main__":
    main()
