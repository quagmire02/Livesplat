"""Draft a robot yaml (sim/robots/) from any MuJoCo arm model: joints, actuators, gripper, TCP and axes.

  python -m sim.robot_probe <model.xml> --name myarm [--prefix myarm/]   -> prints a yaml draft
The gripper actuator is the one driving a finger/gripper joint or tendon. The TCP is the midpoint between the two
finger bodies' geoms when the gripper is closed, expressed in their common parent body; close_axis points from
one finger to the other; approach_axis from the parent body origin to the TCP. Opening = finger gap when open.
Check the draft in the viewer (Lab tab) or with sim.synthetic before trusting it.
"""
import argparse
import re

import mujoco
import numpy as np

FINGER = re.compile(r"finger|pad|jaw|gripper_(left|right)|left|right", re.I)
GRIP_ACT = re.compile(r"finger|grip|jaw|hand|actuator8", re.I)


def _subtree(m, root):
    out, stack = [], [root]
    while stack:
        b = stack.pop()
        out.append(b)
        stack += [c for c in range(m.nbody) if m.body_parentid[c] == b and c != b]
    return out


def _settle(m, d, ctrl_i, value, seconds=1.5):
    g = m.opt.gravity.copy()
    m.opt.gravity[:] = 0
    for _ in range(int(seconds / m.opt.timestep)):
        d.ctrl[ctrl_i] = value
        mujoco.mj_step(m, d)
    m.opt.gravity[:] = g


def probe(xml, name, prefix):
    m = mujoco.MjModel.from_xml_path(xml)
    d = mujoco.MjData(m)
    if m.nkey:
        mujoco.mj_resetDataKeyframe(m, d, 0)
    mujoco.mj_forward(m, d)
    acts = [m.actuator(i).name for i in range(m.nu)]
    targets = []
    for i in range(m.nu):
        t, tid = m.actuator_trntype[i], m.actuator_trnid[i, 0]
        targets.append(m.joint(tid).name if t == mujoco.mjtTrn.mjTRN_JOINT else
                       m.tendon(tid).name if t == mujoco.mjtTrn.mjTRN_TENDON else "")
    gi = next((i for i in range(m.nu) if GRIP_ACT.search(acts[i]) or GRIP_ACT.search(targets[i])), m.nu - 1)
    arm_acts = [a for i, a in enumerate(acts) if i != gi]
    arm_joints = [targets[i] for i in range(m.nu) if i != gi]
    lo, hi = m.actuator_ctrlrange[gi]
    cand = [b for b in range(m.nbody) if FINGER.search(m.body(b).name) and m.body_geomnum[b] > 0]
    named = [b for b in cand if re.search(r"finger|pad", m.body(b).name, re.I)] or cand   # not knuckles/links
    left = [b for b in named if re.search(r"left|_l\b|1$", m.body(b).name, re.I)]
    right = [b for b in named if re.search(r"right|_r\b|2$", m.body(b).name, re.I)]
    fingers = [left[0], right[0]] if left and right else named[:2]
    if len(fingers) < 2:
        raise SystemExit(f"found finger bodies {[m.body(b).name for b in fingers]}: name them with --fingers")

    def finger_pts():
        return [np.mean([d.geom_xpos[g] for g in range(m.ngeom) if m.geom_bodyid[g] == b], 0) for b in fingers]

    gaps = {}
    for label, v in (("a", lo), ("b", hi)):
        _settle(m, d, gi, v)
        p = finger_pts()
        gaps[label] = (float(np.linalg.norm(p[0] - p[1])), v, p)
    (g_open, v_open, _), (g_closed, v_closed, pc) = sorted(gaps.values(), key=lambda x: -x[0])
    parent = m.body_parentid[fingers[0]]
    while fingers[1] not in _subtree(m, parent):
        parent = m.body_parentid[parent]
    R = d.xmat[parent].reshape(3, 3)
    tcp_w = (pc[0] + pc[1]) / 2
    off = R.T @ (tcp_w - d.xpos[parent])
    close = R.T @ (pc[1] - pc[0])
    approach = off / max(np.linalg.norm(off), 1e-9)
    root = next(m.body(b).name for b in range(1, m.nbody) if m.body_parentid[b] == 0)
    home = [round(float(d.qpos[m.joint(j).qposadr[0]]), 3) for j in arm_joints] if m.nkey else [0.0] * len(arm_joints)
    f = lambda v: "[" + ", ".join(f"{x:.4f}" for x in v) + "]"
    print(f"""# {name}: drafted by sim.robot_probe from {xml}. Check before trusting.
name: {name}
model: {xml}
root_body: {root}
prefix: "{prefix}"
arm_joints: [{', '.join(arm_joints)}]
arm_actuators: [{', '.join(arm_acts)}]
home: {home}
tcp: {{body: {m.body(parent).name}, offset: {f(off)}}}
approach_axis: {f(approach)}
close_axis: {f(close / max(np.linalg.norm(close), 1e-9))}
ik: {'full' if len(arm_joints) >= 6 else 'axis'}
gripper:
  type: parallel
  actuator: {acts[gi]}
  open: {v_open:.4g}
  closed: {v_closed:.4g}
  max_opening_m: {g_open:.3f}          # geom-centre gap; pad faces are a little closer
  palm_clearance_m: 0.04
base_gap_m: 0.10
side_min_height_m: 0.04
approach_tolerance_deg: {5 if len(arm_joints) >= 6 else 30}""")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("xml")
    ap.add_argument("--name", required=True)
    ap.add_argument("--prefix")
    a = ap.parse_args()
    probe(a.xml, a.name, a.prefix or f"{a.name}/")


if __name__ == "__main__":
    main()
