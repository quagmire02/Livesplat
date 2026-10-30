"""Calibrate a robot model's grip force to the real gripper's.

Off-the-shelf models can be far off in the quantity that decides grasps: on a 5 cm block the Menagerie Panda
squeezes ~1 N per finger (real Franka Hand: 70 N continuous), the xArm7 ~400 N. This closes the gripper on a 5 cm
block in a synthetic scene, measures the normal force per finger, and prints the actuator scale that makes it
match gripper.force_n in the robot yaml. Scaling gain and bias (stiffness, damping) together keeps the open and
closed positions and only changes the strength. sim.grasp.Arm applies gripper.force_scale.

  python -m sim.grip_calibrate panda xarm7 ...      -> measured N/finger, target, force_scale to put in the yaml
"""
import sys
from pathlib import Path

import mujoco
import numpy as np


def squeeze(robot, scale=None):
    """Normal force per finger (N) on a 5 cm block after closing, with an optional force_scale override."""
    from lab.grasp import setup
    from sim.robot import CLEAR, top_dirs
    from sim.synthetic import make
    sim = make(Path("/tmp/livesplat_grip_cal"), "block:0.05,0.05,0.06@0.0,0.0")
    spec, meta, rows, arm, _ = setup(Path(sim), robot)
    if scale is not None:
        arm.set_force_scale(scale)
    row, m = rows[0], arm.m
    bid = m.body(row["body"]).id
    d = arm.reset()
    arm.drive(d, spec.home, spec.grip_open, 0.3)
    pos = d.body(row["body"]).xpos.copy()
    dirs = top_dirs(np.radians(row["grasp_yaw_deg"]))
    top = pos[2] + meta["objects"][0]["size_xyz"][2]
    pos[:2] += np.array(row["grasp_centre_offset_xy"])
    q_above, *_ = arm.solve(np.array([pos[0], pos[1], top + CLEAR]), dirs, spec.home)
    q_at, *_ = arm.solve(np.array([pos[0], pos[1], top - 0.02]), dirs, q_above)
    arm.drive(d, q_above, spec.grip_open, 1.5)
    arm.drive(d, q_at, spec.grip_open, 1.0)
    arm.drive(d, q_at, spec.grip_closed, 1.0)
    f6, total = np.zeros(6), 0.0
    for i in range(d.ncon):
        c = d.contact[i]
        b1, b2 = m.geom_bodyid[c.geom1], m.geom_bodyid[c.geom2]
        if bid in (b1, b2) and any(m.body(b).name.startswith(spec.prefix) for b in (b1, b2)):
            mujoco.mj_contactForce(m, d, i, f6)
            total += abs(f6[0])
    return total / 2, spec


def main():
    for robot in sys.argv[1:]:
        f, spec = squeeze(robot, 1.0)
        target = spec.force_n
        if not target:
            print(f"{robot}: {f:.1f} N per finger (no gripper.force_n in the yaml to calibrate to)")
            continue
        # The squeeze is not always proportional to the scale (force limits, finger linkages): search.
        lo, hi = (1.0, max(2.0, 4 * target / max(f, 1e-6))) if f < target else (min(0.5, 0.25 * target / f), 1.0)
        best = (abs(f - target), 1.0, f)
        for _ in range(9):
            s = np.sqrt(lo * hi)                                     # bisection in log space
            fs, _ = squeeze(robot, s)
            if abs(fs - target) < best[0]:
                best = (abs(fs - target), s, fs)
            if fs < target:
                lo = s
            else:
                hi = s
        print(f"{robot}: model {f:.1f} N/finger, target {target:g} N -> force_scale: {best[1]:.4g} "
              f"(gives {best[2]:.1f} N/finger)", flush=True)


if __name__ == "__main__":
    main()
