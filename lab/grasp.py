"""Arm grasp test: can <robot> pick up each scanned object, how (top / side), and how reliably?

Builds on sim/ (export -> robot -> grasp): a metric sim package of the scan, the arm's base placed automatically
at the table edge (most objects reachable), then real dynamic grasps (position actuators, contacts, lift 10 cm).
Every grasp runs over mass x0.7/1/1.3 and grip friction x0.7/1/1.3; unless the scale is calibrated, the whole
scene is also exported at (1 -+ spread) size with the arm base fixed in metres (arms are never scaled: they
are the manufacturers' models). Grip friction: MuJoCo uses the larger of two surfaces, so each trial sets the
robot's and the object's friction to the combined sqrt(pad * object) value.

CLI: python -m lab.grasp <run> [--robot so100 panda ...] [--quick]   -> <run>/lab/grasp_<robot>.json
"""
import argparse
import itertools
import json
from pathlib import Path

import mujoco
import numpy as np

PAD = {"so100": 0.35, "panda": 0.5}             # fingertip friction (bare PLA, Panda plastic); others 0.6
MASS_F, FRIC_F = (0.7, 1.0, 1.3), (0.7, 1.0, 1.3)


def package(run, mpu, tag):
    from sim.export import export
    out = Path(run) / "lab" / f"sim_{tag}"
    if not (out / "metadata.json").exists():
        export(run, out=out, scale=mpu, scale_source=f"lab grasp test ({tag})", bg_crop=1.5)
    return out


def _hulls(sim, meta):
    import trimesh
    return {o["body"]: np.asarray(trimesh.load(sim / o["files"]["collision"]).vertices) for o in meta["objects"]}


def _setup(sim, robot, base=None):
    from sim.grasp import Arm
    from sim.robot import build, choose_base, reach_report
    from sim.robot_spec import RobotSpec
    spec = RobotSpec.load(robot)
    meta = json.loads((sim / "metadata.json").read_text())
    hulls = _hulls(sim, meta)
    if base is None:
        _, pos, yaw, rows = choose_base(sim, spec, meta, hulls, 1.0)
    else:
        pos, yaw = base
        rows = reach_report(build(sim, spec, pos, yaw), spec, meta, hulls, 1.0)
    m = build(sim, spec, pos, yaw)
    return spec, meta, rows, Arm(m, spec), (tuple(pos), float(yaw))


def _trial(arm, row, obj_meta, mode, mass_f, fric_f, pad):
    from sim.grasp import run_grasp
    m = arm.m
    bid = m.body(row["body"]).id
    mass0, inert0, fric0 = m.body_mass[bid], m.body_inertia[bid].copy(), m.geom_friction[:, 0].copy()
    obj_g = m.geom_bodyid == bid
    robot_g = np.array([m.body(int(b)).name.startswith(arm.spec.prefix) for b in m.geom_bodyid])
    mu = float(np.sqrt(pad * obj_meta["friction"]["value"] * fric_f))
    try:
        m.body_mass[bid] = mass0 * mass_f
        m.body_inertia[bid] = inert0 * mass_f
        m.geom_friction[obj_g | robot_g, 0] = mu
        return run_grasp(arm, row, obj_meta, 1.0, mode)
    finally:
        m.body_mass[bid], m.body_inertia[bid] = mass0, inert0
        m.geom_friction[:, 0] = fric0


def run(run_dir, robot, quick=False):
    from measure.scale import load
    s = load(run_dir)
    mpu, spread = s.get("metres_per_unit") or 0.6, s.get("spread") or 0.1
    calibrated = str(s.get("source", "")).startswith("user")
    scales = [1.0] if calibrated or quick else [1.0, 1 - spread, 1 + spread]
    pad = PAD.get(Path(robot).stem, 0.6)
    res = {"robot": robot, "scale": {"metres_per_unit": mpu, "spread": spread, "calibrated": calibrated},
           "objects": {}}
    base = None
    for sf in scales:
        sim = package(run_dir, mpu * sf, f"{round(mpu * sf * 1000)}")
        spec, meta, rows, arm, b = _setup(sim, robot, base)
        res.setdefault("body_names", {o["body"]: o["name"] for o in meta["objects"]})
        if base is None:
            base = b
            res["base"] = {"pos_m": [round(float(v), 3) for v in b[0]], "yaw_deg": round(float(np.degrees(b[1])), 1)}
            res["robot_name"] = spec.name
        for row in rows:
            o = next(x for x in meta["objects"] if x["body"] == row["body"])
            rec = res["objects"].setdefault(row["body"], {"name": row["object"], "modes": {}, "reach": []})
            rec["reach"].append({"scale_f": round(sf, 3), "top_reachable": row["reachable_top_down"],
                                 "fits_top": row["fits_gripper"], "top_width_mm": round(row["grasp_width_m"] * 1000, 1),
                                 "side_feasible": row["side"].get("feasible", False),
                                 "side_blocked_by": row["side"].get("blocked_by", [])})
            for mode in ("top", "side"):
                cases = [(1.0, 1.0)] if quick else list(itertools.product(MASS_F, FRIC_F))
                for mf, ff in cases:
                    r = _trial(arm, row, o, mode, mf, ff, pad)
                    md = rec["modes"].setdefault(mode, {"trials": 0, "success": 0, "reasons": {}, "lifted_cm": []})
                    md["trials"] += 1
                    md["success"] += int(r["success"])
                    if not r["success"]:
                        why = r.get("reason") or ("slipped / dropped" if r.get("lifted_m", 0) < 0.05 else "lost it")
                        md["reasons"][why] = md["reasons"].get(why, 0) + 1
                    if "lifted_m" in r:
                        md["lifted_cm"].append(round(r["lifted_m"] * 100, 1))
                    if "grasp_point" in r:
                        md["grasp_point"] = r["grasp_point"]
        print(f"  {spec.name} at scale x{sf:.2f}: done", flush=True)
    for rec in res["objects"].values():
        for md in rec["modes"].values():
            md["rate"] = round(md["success"] / max(md["trials"], 1), 2)
        best = max(rec["modes"].items(), key=lambda kv: kv[1]["rate"])
        rec["best_mode"], rec["best_rate"] = best[0], best[1]["rate"]
        rec["suggestion"] = _suggest(rec, res)
    return res


def _readable(text, res):
    """obj_2 / ['obj_1', 'world'] -> object names and 'the table'."""
    import re
    names = {**res.get("body_names", {}), "world": "the table"}
    text = re.sub(r"[\[\]']", "", str(text))
    return re.sub(r"\b(obj_\d+|world)\b", lambda m: names.get(m.group(1), m.group(1)), text)


def _suggest(rec, res):
    return _readable(_suggest_raw(rec, res), res)


def _suggest_raw(rec, res):
    top, side = rec["modes"].get("top", {}), rec["modes"].get("side", {})
    r0 = rec["reach"][0]
    if rec["best_rate"] >= 0.9:
        return f"Reliable with a {rec['best_mode']} grasp ({int(rec['best_rate'] * 100)}% of trials)."
    if rec["best_rate"] == 0:
        if not r0["top_reachable"] and not r0["side_feasible"]:
            blk = r0["side_blocked_by"]
            return ("Out of reach from the chosen base: move the object closer or the arm base nearer to it." if not blk
                    else f"Blocked by {', '.join(blk)}: clear the space around it.")
        if r0["top_reachable"] and not r0["fits_top"]:
            return (f"Too wide for a top grasp ({r0['top_width_mm']} mm); no side grasp works either. "
                    "Use a wider gripper or grasp a narrower part.")
        return "Every attempt slipped: add rubber/TPU fingertip pads or use a stronger gripper."
    reasons = {}
    for md in (top, side):
        for k, v in md.get("reasons", {}).items():
            reasons[k] = reasons.get(k, 0) + v
    worst = max(reasons, key=reasons.get) if reasons else "unknown"
    return (f"Works in {int(rec['best_rate'] * 100)}% of trials ({rec['best_mode']} grasp); most failures: {worst}. "
            + ("Rubber/TPU pads would help." if "slip" in worst or "dropped" in worst else
               "Calibrate the scale: the result depends on the true size." if not res["scale"]["calibrated"] else ""))


def summary(res):
    lines = [f"{res.get('robot_name', res['robot'])} grasp test (base {res.get('base', {}).get('pos_m')}, "
             f"scale {'calibrated' if res['scale']['calibrated'] else 'estimated ±' + str(int(res['scale']['spread'] * 100)) + '%'})"]
    for rec in res["objects"].values():
        modes = ", ".join(f"{k} {int(v['rate'] * 100)}%" for k, v in rec["modes"].items())
        lines.append(f"- {rec['name']}: {modes}. {rec['suggestion']}")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("--robot", nargs="+", default=["so100", "panda"])
    ap.add_argument("--quick", action="store_true", help="nominal case only (1 trial per grasp)")
    a = ap.parse_args()
    d = Path(a.run_dir) / "lab"
    d.mkdir(exist_ok=True)
    for robot in a.robot:
        res = run(a.run_dir, robot, a.quick)
        (d / f"grasp_{Path(robot).stem}.json").write_text(json.dumps(res, indent=1))
        print(summary(res), "\n", flush=True)


if __name__ == "__main__":
    main()
