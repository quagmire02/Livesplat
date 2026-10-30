"""Simulation mode: dynamic grasp tests with any robot spec (position actuators, real contacts).

Run (WSL): python -m sim.grasp <sim_dir> [--mode auto|top|side|both] [--render]
Needs robot_scene.xml + checks/robot_report.json from sim.robot (the robot named in the report is used).
top : above -> down -> close -> lift.   side: back off 12 cm -> in -> close -> lift.
Success = the object rises >= 5 cm with the hand and stays near it. Results -> checks/grasp_report.json.
"""
import argparse
import json
from pathlib import Path

import mujoco
import numpy as np

from sim.robot import CLEAR, arm_index, ik, side_dirs, tcp_pose, top_dirs
from sim.robot_spec import RobotSpec


class Arm:
    """Actuator view of one robot inside a compiled scene."""

    def __init__(self, m, spec):
        self.m, self.spec = m, spec
        self.arm = np.array([m.actuator(spec.prefix + n).id for n in spec.arm_actuators])
        self.grip = m.actuator(spec.prefix + spec.gripper_actuator).id
        self.qadr = arm_index(m, spec)[0]
        if spec.grip_kp > 0:                                    # position actuator: force = kp * (ctrl - q)
            m.actuator_gainprm[self.grip, 0] = spec.grip_kp
            m.actuator_biasprm[self.grip, 1] = -spec.grip_kp
        self._gain0 = m.actuator_gainprm[self.grip].copy()
        self._bias0 = m.actuator_biasprm[self.grip].copy()
        self._frange0 = m.actuator_forcerange[self.grip].copy()
        self.set_force_scale(spec.force_scale)
        # Robot joints whose zero lies outside their range (Trossen fingers: 15-37 mm) must start inside it,
        # or the fingers start interpenetrating and jam.
        self.limited = [(m.jnt_qposadr[j], *m.jnt_range[j]) for j in range(m.njnt)
                        if m.jnt_limited[j] and m.joint(j).name.startswith(spec.prefix)
                        and int(m.jnt_type[j]) in (int(mujoco.mjtJoint.mjJNT_HINGE), int(mujoco.mjtJoint.mjJNT_SLIDE))]

    def set_force_scale(self, s):
        """Scale the gripper actuator's strength (gain, stiffness, damping) keeping its open/closed positions."""
        m = self.m
        m.actuator_gainprm[self.grip] = self._gain0
        m.actuator_biasprm[self.grip] = self._bias0
        m.actuator_gainprm[self.grip, 0] = self._gain0[0] * s
        m.actuator_biasprm[self.grip, 1:3] = self._bias0[1:3] * s
        if m.actuator_forcelimited[self.grip]:
            m.actuator_forcerange[self.grip] = self._frange0 * s

    def reset(self):
        d = mujoco.MjData(self.m)
        mujoco.mj_resetData(self.m, d)
        for adr, lo, hi in self.limited:
            d.qpos[adr] = np.clip(d.qpos[adr], lo, hi)
        d.qpos[self.qadr] = self.spec.home
        d.ctrl[self.arm] = self.spec.home
        d.ctrl[self.grip] = self.spec.grip_open
        mujoco.mj_forward(self.m, d)
        return d

    def drive(self, d, q_target, grip, seconds, frames=None, every=0.05):
        """Interpolate arm position targets over `seconds` while stepping physics."""
        q0 = d.ctrl[self.arm].copy()
        n = int(seconds / self.m.opt.timestep)
        for i in range(n):
            a = min(1.0, (i + 1) / max(1, int(0.8 * n)))
            d.ctrl[self.arm] = q0 + a * (q_target - q0)
            d.ctrl[self.grip] = grip
            mujoco.mj_step(self.m, d)
            if frames is not None and i % int(every / self.m.opt.timestep) == 0:
                frames.append(d.qpos.copy())

    def solve(self, target, dirs, seed):
        d = mujoco.MjData(self.m)
        d.qpos[self.qadr] = seed
        err, ang = ik(self.m, d, self.spec, target, *dirs)
        return d.qpos[self.qadr].copy(), err, ang


def _result(arm, d, body, z0, row, mode, point, err_at, upm):
    spec = arm.spec
    rise = d.body(body).xpos[2] - z0
    _, _, tcp = tcp_pose(arm.m, d, spec)
    held = np.linalg.norm(d.body(body).xpos[:2] - tcp[:2]) < 0.15 * upm
    return {"object": row["object"], "robot": spec.name, "mode": mode, "lifted_m": round(float(rise / upm), 4),
            "success": bool(rise > 0.05 * upm and held), "grasp_point": point,
            "ik_error_at_grasp_m": round(err_at / upm, 4)}


def try_top(arm, body, row, meta_obj, upm, frames=None):
    spec = arm.spec
    d = arm.reset()
    arm.drive(d, spec.home, spec.grip_open, 0.3, frames)            # let objects settle under the arm
    pos = d.body(body).xpos.copy()
    top = pos[2] + meta_obj["size_xyz"][2]
    dirs = top_dirs(np.radians(row["grasp_yaw_deg"]))
    depth = min(spec.top_depth * upm, 0.5 * meta_obj["size_xyz"][2])
    pos[:2] += np.array(row["grasp_centre_offset_xy"])              # centre of the slice, not the base origin
    above = np.array([pos[0], pos[1], top + CLEAR * upm])
    at = np.array([pos[0], pos[1], top - depth])
    q_above, *_ = arm.solve(above, dirs, spec.home)
    q_at, err_at, _ = arm.solve(at, dirs, q_above)
    q_lift, *_ = arm.solve(above + np.array([0, 0, 0.10 * upm]), dirs, q_above)
    arm.drive(d, q_above, spec.grip_open, 1.5, frames)
    arm.drive(d, q_at, spec.grip_open, 1.0, frames)
    arm.drive(d, q_at, spec.grip_closed, 0.8, frames)
    z0 = d.body(body).xpos[2]
    arm.drive(d, q_lift, spec.grip_closed, 1.5, frames)
    arm.drive(d, q_lift, spec.grip_closed, 0.5, frames)
    return _result(arm, d, body, z0, row, "top", f"{depth / upm * 100:.1f} cm below the object top (top-down)",
                   err_at, upm)


def try_side(arm, body, row, upm, frames=None):
    spec, s = arm.spec, row["side"]
    d = arm.reset()
    arm.drive(d, spec.home, spec.grip_open, 0.3, frames)
    dirs = side_dirs(s["approach_xy"])
    tcp = np.array(s["tcp"])
    q_pre, *_ = arm.solve(tcp - s.get("backoff_m", 0.12) * upm * dirs[0], dirs, spec.home)
    q_at, err_at, _ = arm.solve(tcp, dirs, q_pre)
    q_lift, *_ = arm.solve(tcp + np.array([0, 0, 0.10 * upm]), dirs, q_at)
    arm.drive(d, q_pre, spec.grip_open, 1.5, frames)
    arm.drive(d, q_at, spec.grip_open, 1.2, frames)
    arm.drive(d, q_at, spec.grip_closed, 0.8, frames)
    z0 = d.body(body).xpos[2]
    arm.drive(d, q_lift, spec.grip_closed, 1.5, frames)
    arm.drive(d, q_lift, spec.grip_closed, 0.5, frames)
    return _result(arm, d, body, z0, row, "side", f"{s['height_m'] * 100:.0f} cm above the table (side)", err_at, upm)


def run_grasp(arm, row, meta_obj, upm, mode="auto", frames=None):
    """mode: top | side | auto (top-down if it fits and holds, else side)."""
    top_ok = row["reachable_top_down"] and row["fits_gripper"]
    side_ok = row.get("side", {}).get("feasible", False)
    if mode == "top" or (mode == "auto" and top_ok):
        if not top_ok:
            why = ("too wide for the gripper" if row["reachable_top_down"] else
                   f"blocked by {row['top_blocked_by']}" if row.get("top_blocked_by") else "not reachable")
            return {"object": row["object"], "robot": arm.spec.name, "mode": "top", "success": False, "reason": why}
        r = try_top(arm, row["body"], row, meta_obj, upm, frames)
        if r["success"] or mode == "top" or not side_ok:
            return r
        if frames is not None:
            frames.clear()
    if not side_ok:
        why = f"blocked by {row['side']['blocked_by']}" if row.get("side", {}).get("blocked_by") else "no side grasp fits"
        return {"object": row["object"], "robot": arm.spec.name, "mode": "side", "success": False, "reason": why}
    return try_side(arm, row["body"], row, upm, frames)


def load(sim):
    sim = Path(sim)
    meta = json.loads((sim / "metadata.json").read_text())
    rep = json.loads((sim / "checks" / "robot_report.json").read_text())
    spec = RobotSpec.load(rep.get("robot_yaml", rep["robot"]))
    m = mujoco.MjModel.from_xml_path(str(sim / f"robot_scene_{spec.name}.xml"))
    return meta, rep, Arm(m, spec)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("sim_dir")
    ap.add_argument("--render", action="store_true")
    ap.add_argument("--mode", choices=["auto", "top", "side", "both"], default="both")
    a = ap.parse_args()
    sim = Path(a.sim_dir)
    meta, rep, arm = load(sim)
    upm = rep["units_per_metre"]
    out = []
    for row in rep["objects"]:
        o = next(x for x in meta["objects"] if x["body"] == row["body"])
        for mode in (["top", "side"] if a.mode == "both" else [a.mode]):
            frames = [] if a.render else None
            r = run_grasp(arm, row, o, upm, mode, frames)
            out.append(r)
            if frames:
                np.save(sim / "checks" / f"grasp_{arm.spec.name}_{row['body']}_{r['mode']}_qpos.npy", np.array(frames))
    (sim / "checks" / f"grasp_report_{arm.spec.name}.json").write_text(
        json.dumps({"robot": arm.spec.name, "results": out, "units_per_metre": upm}, indent=1))
    for r in out:
        print(f"  {r['object']:<20} {r['mode']:<4} {'SUCCESS' if r['success'] else 'fail   '} "
              + (f"lifted {r['lifted_m'] * 100:.1f} cm, {r['grasp_point']}" if "lifted_m" in r else r.get("reason", "")))


if __name__ == "__main__":
    main()
