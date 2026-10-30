"""Arm grasp test: can <robot> pick up each scanned object, how (top / side), how reliably, why, and what helps?

Builds on sim/ (export -> robot -> grasp): a metric sim package of the scan, the arm's base placed automatically
at the table edge (most objects reachable), then real dynamic grasps (position actuators, contacts, lift 10 cm).
Every grasp runs over mass x0.7/1/1.3 and grip friction x0.7/1/1.3; unless the scale is calibrated, the whole
scene is also exported at (1 -+ spread) size with the arm base fixed in metres (arms are never scaled: they
are the manufacturers' models). Grip friction: MuJoCo uses the larger of two surfaces, so each trial sets the
robot's and the object's friction to the combined sqrt(pad * object) value. Every trial is recorded, so the
report can say what decides the outcome (lab.advisor), and Muse's grip choice and proposed changes are run as
experiments with measured results (--advise).

CLI: python -m lab.grasp <run> [--robot so100 panda ...] [--quick] [--advise]   -> <run>/lab/grasp_<robot>.json
"""
import argparse
import itertools
import json
from pathlib import Path

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


def setup(sim, robot, base=None):
    """(spec, meta, reach rows, Arm, (base pos, yaw)). base=None: search the best base at the table edge."""
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
    return spec, meta, rows, Arm(m, spec), (tuple(float(v) for v in pos), float(yaw))


_setup = setup                                   # older name


def trial(arm, row, obj_meta, mode, mass_f, fric_f, pad, kp_f=1.0, frames=None):
    """One dynamic grasp with this object mass / grip friction / gripper stiffness. Model restored after."""
    from sim.grasp import run_grasp
    m = arm.m
    bid = m.body(row["body"]).id
    mass0, inert0, fric0 = m.body_mass[bid], m.body_inertia[bid].copy(), m.geom_friction[:, 0].copy()
    gain0, bias0 = m.actuator_gainprm[arm.grip].copy(), m.actuator_biasprm[arm.grip].copy()
    obj_g = m.geom_bodyid == bid
    robot_g = np.array([m.body(int(b)).name.startswith(arm.spec.prefix) for b in m.geom_bodyid])
    mu = float(np.sqrt(pad * obj_meta["friction"]["value"] * fric_f))
    try:
        m.body_mass[bid] = mass0 * mass_f
        m.body_inertia[bid] = inert0 * mass_f
        m.geom_friction[obj_g | robot_g, 0] = mu
        if kp_f != 1.0:
            m.actuator_gainprm[arm.grip, 0] *= kp_f
            m.actuator_biasprm[arm.grip, 1] *= kp_f
        return run_grasp(arm, row, obj_meta, 1.0, mode, frames)
    finally:
        m.body_mass[bid], m.body_inertia[bid] = mass0, inert0
        m.geom_friction[:, 0] = fric0
        m.actuator_gainprm[arm.grip], m.actuator_biasprm[arm.grip] = gain0, bias0


def why_failed(r):
    if r["success"]:
        return None
    if r.get("reason"):
        return r["reason"]
    return "slipped / dropped" if r.get("lifted_m", 0) < 0.05 else "lost it"


def rate(arm, row, o, mode, pad, kp_f=1.0, cases=None):
    """Success rate over the mass x friction grid (nominal scale) + failure reasons."""
    cases = cases or list(itertools.product(MASS_F, FRIC_F))
    ok, reasons = 0, {}
    for mf, ff in cases:
        r = trial(arm, row, o, mode, mf, ff, pad, kp_f)
        ok += int(r["success"])
        w = why_failed(r)
        if w:
            reasons[w] = reasons.get(w, 0) + 1
    return round(ok / len(cases), 2), reasons


def run(run_dir, robot, quick=False):
    from measure.scale import load
    s = load(run_dir)
    mpu, spread = s.get("metres_per_unit") or 0.6, s.get("spread") or 0.1
    calibrated = str(s.get("source", "")).startswith("user")
    scales = [1.0] if calibrated or quick else [1.0, 1 - spread, 1 + spread]
    pad = PAD.get(Path(robot).stem, 0.6)
    res = {"robot": robot, "pad_friction": pad,
           "scale": {"metres_per_unit": mpu, "spread": spread, "calibrated": calibrated}, "objects": {}}
    base = None
    for sf in scales:
        sim = package(run_dir, mpu * sf, f"{round(mpu * sf * 1000)}")
        spec, meta, rows, arm, b = setup(sim, robot, base)
        res.setdefault("body_names", {o["body"]: o["name"] for o in meta["objects"]})
        if base is None:
            base = b
            res["base"] = {"pos_m": [round(v, 3) for v in b[0]], "yaw_deg": round(float(np.degrees(b[1])), 1)}
            res["robot_name"] = spec.name
            res["gripper"] = {"type": spec.gripper_type, "max_opening_m": spec.max_opening,
                              "force_n": spec.force_n, "force_scale": spec.force_scale}
            res["nominal_package"] = str(sim)
        for row in rows:
            o = next(x for x in meta["objects"] if x["body"] == row["body"])
            rec = res["objects"].setdefault(row["body"], {
                "name": row["object"], "modes": {}, "reach": [], "mass_kg": o["mass"]["value"],
                "friction": o["friction"]["value"], "size_m": o["size_xyz"]})
            rec["reach"].append({"scale_f": round(sf, 3), "top_reachable": row["reachable_top_down"],
                                 "fits_top": row["fits_gripper"], "top_width_mm": round(row["grasp_width_m"] * 1000, 1),
                                 "side_feasible": row["side"].get("feasible", False),
                                 "side_height_m": row["side"].get("height_m"),
                                 "side_blocked_by": row["side"].get("blocked_by", [])})
            for mode in ("top", "side"):
                cases = [(1.0, 1.0)] if quick else list(itertools.product(MASS_F, FRIC_F))
                md = rec["modes"].setdefault(mode, {"trials": 0, "success": 0, "reasons": {}, "lifted_cm": [],
                                                    "cases": []})
                for mf, ff in cases:
                    r = trial(arm, row, o, mode, mf, ff, pad)
                    md["trials"] += 1
                    md["success"] += int(r["success"])
                    md["cases"].append({"mass_f": mf, "fric_f": ff, "scale_f": round(sf, 3), "ok": bool(r["success"])})
                    w = why_failed(r)
                    if w:
                        md["reasons"][w] = md["reasons"].get(w, 0) + 1
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
    g = res.get("gripper", {})
    grip = (f"grip {g['force_n']:g} N/finger (calibrated)" if g.get("force_n") and g.get("force_scale", 1) != 1
            else "grip force: model default (not calibrated)")
    lines = [f"{res.get('robot_name', res['robot'])} grasp test (base {res.get('base', {}).get('pos_m')}, {grip}, "
             f"scale {'calibrated' if res['scale']['calibrated'] else 'estimated ±' + str(int(res['scale']['spread'] * 100)) + '%'})"]
    for rec in res["objects"].values():
        modes = ", ".join(f"{k} {int(v['rate'] * 100)}%" for k, v in rec["modes"].items())
        lines.append(f"- {rec['name']}: {modes}. {rec['suggestion']}")
        if rec.get("decides"):
            lines.append(f"  · What decides it: {rec['decides']}")
        if rec.get("muse_grip"):
            g = rec["muse_grip"]
            lines.append(f"  · Muse's grip ({g['mode']} at {int(g['band'][0] * 100)}-{int(g['band'][1] * 100)}% height: "
                         f"{g['why']}) -> {int(g['rate'] * 100)}% vs planner "
                         f"{int(g.get('planner_nominal', rec['best_rate']) * 100)}% (same 9 cases, estimated size)")
        for e in rec.get("experiments", []):
            lines.append(f"  · Experiment: {e['label']} -> {int(e['rate'] * 100)}% "
                         f"(was {int(e['baseline'] * 100)}%). {e['why']}")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("--robot", nargs="+", default=["so100", "panda"])
    ap.add_argument("--quick", action="store_true", help="nominal case only (1 trial per grasp)")
    ap.add_argument("--advise", action="store_true", help="Muse grip choice + proposed experiments, simulated")
    a = ap.parse_args()
    d = Path(a.run_dir) / "lab"
    d.mkdir(exist_ok=True)
    for robot in a.robot:
        res = run(a.run_dir, robot, a.quick)
        if a.advise:
            from lab.advisor import advise
            advise(a.run_dir, robot, res)
        (d / f"grasp_{Path(robot).stem}.json").write_text(json.dumps(res, indent=1))
        print(summary(res), "\n", flush=True)


if __name__ == "__main__":
    main()
