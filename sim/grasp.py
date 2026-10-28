"""T7: dynamic top-down grasp test with the Panda (position actuators, real contacts).

Run (WSL): python -m sim.grasp <sim_dir> [--render]
Needs robot_scene.xml and checks/robot_report.json from sim.robot.
For each reachable object: home -> above -> down -> close fingers -> lift. Success = the object
rises >= 5 cm with the hand and is still between the fingers. Results -> checks/grasp_report.json.
"""
import argparse
import json
from pathlib import Path

import mujoco
import numpy as np

from sim.robot import CLEAR, GRASP_DEPTH, P, arm_index, down_frame, ik

HOME = np.array([0, 0, 0, -1.571, 0, 1.571, -0.785])
OPEN, CLOSED = 255.0, 0.0


def drive(m, d, q_target, grip, seconds, frames=None, every=0.05):
    """Interpolate arm position targets over `seconds` while stepping physics."""
    qadr, *_ = arm_index(m)
    q0 = d.ctrl[:7].copy()
    n = int(seconds / m.opt.timestep)
    for i in range(n):
        a = min(1.0, (i + 1) / max(1, int(0.8 * n)))
        d.ctrl[:7] = q0 + a * (q_target - q0)
        d.ctrl[7] = grip
        mujoco.mj_step(m, d)
        if frames is not None and i % int(every / m.opt.timestep) == 0:
            frames.append(d.qpos.copy())


def solve(m, target, yaw, seed):
    d = mujoco.MjData(m)
    qadr, *_ = arm_index(m)
    d.qpos[qadr] = seed
    err, ang = ik(m, d, target, down_frame(yaw))
    return d.qpos[qadr].copy(), err, ang


def try_grasp(m, body, row, meta_obj, upm, frames=None):
    d = mujoco.MjData(m)
    mujoco.mj_resetData(m, d)
    qadr, *_ = arm_index(m)
    d.qpos[qadr] = HOME
    d.ctrl[:7], d.ctrl[7] = HOME, OPEN
    mujoco.mj_forward(m, d)
    drive(m, d, HOME, OPEN, 0.3, frames)                       # let objects settle under the arm
    pos = d.body(body).xpos.copy()
    top = pos[2] + meta_obj["size_xyz"][2]
    yaw = np.radians(row["grasp_yaw_deg"])
    depth = min(GRASP_DEPTH * upm, 0.5 * meta_obj["size_xyz"][2])
    pos[:2] += np.array(row["grasp_centre_offset_xy"])     # centre of the slice, not the base origin
    above = np.array([pos[0], pos[1], top + CLEAR * upm])
    at = np.array([pos[0], pos[1], top - depth])
    lift = above + np.array([0, 0, 0.10 * upm])
    q_above, *_ = solve(m, above, yaw, HOME)
    q_at, err_at, _ = solve(m, at, yaw, q_above)
    q_lift, *_ = solve(m, lift, yaw, q_above)
    drive(m, d, q_above, OPEN, 1.5, frames)
    drive(m, d, q_at, OPEN, 1.0, frames)
    drive(m, d, q_at, CLOSED, 0.8, frames)
    z0 = d.body(body).xpos[2]
    drive(m, d, q_lift, CLOSED, 1.5, frames)
    drive(m, d, q_lift, CLOSED, 0.5, frames)                   # hold
    rise = d.body(body).xpos[2] - z0
    hand = d.body(f"{P}hand").xpos
    held = np.linalg.norm(d.body(body).xpos[:2] - hand[:2]) < 0.06 * upm
    fingers = d.qpos[m.joint(f"{P}finger_joint1").qposadr[0]]
    return {"object": row["object"], "lifted_m": round(float(rise / upm), 4),
            "success": bool(rise > 0.05 * upm and held), "finger_gap_m": round(float(2 * fingers), 4),
            "grasp_point": f"{depth / upm * 100:.0f} cm below the object top (top-down)", "ik_error_at_grasp_m": round(err_at / upm, 4)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("sim_dir")
    ap.add_argument("--render", action="store_true")
    a = ap.parse_args()
    sim = Path(a.sim_dir)
    meta = json.loads((sim / "metadata.json").read_text())
    rep = json.loads((sim / "checks" / "robot_report.json").read_text())
    upm = rep["units_per_metre"]
    m = mujoco.MjModel.from_xml_path(str(sim / "robot_scene.xml"))
    out = []
    for row in rep["objects"]:
        o = next(x for x in meta["objects"] if x["body"] == row["body"])
        if not row["reachable_top_down"]:
            out.append({"object": row["object"], "success": False, "reason": "not reachable"})
            continue
        if not row["fits_gripper"]:
            out.append({"object": row["object"], "success": False, "reason": "too wide for the gripper"})
            continue
        frames = [] if a.render else None
        r = try_grasp(m, row["body"], row, o, upm, frames)
        out.append(r)
        if frames:
            np.save(sim / "checks" / f"grasp_{row['body']}_qpos.npy", np.array(frames))
    (sim / "checks" / "grasp_report.json").write_text(json.dumps({"results": out, "units_per_metre": upm}, indent=1))
    for r in out:
        print(f"  {r['object']:<20} {'SUCCESS' if r['success'] else 'fail   '} "
              + (f"lifted {r['lifted_m'] * 100:.1f} cm, finger gap {r['finger_gap_m'] * 100:.1f} cm"
                 if "lifted_m" in r else r.get("reason", "")))


if __name__ == "__main__":
    main()
