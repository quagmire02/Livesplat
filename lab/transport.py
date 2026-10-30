"""Transport test: how hard can a mobile robot / tray / cart accelerate or brake before each object slides or tips?

Physics: in the carrier's frame a horizontal acceleration a acts like gravity tilted by a/g. For every object
(alone, on its scanned collision hull) and every direction, a bisection finds the smallest a/g that makes it
SLIDE (> 5 mm of travel during the pulse) or TIP (> 25 deg). The answer is a ratio, so it does not depend on the
scan's metric scale. Trials vary friction (estimate +-30%) and the carrier surface. Assumes uniform fill (a
half-full bottle has a lower centre of mass and tips later).

CLI: python -m lab.transport <run> [--surface wood|plastic|rubber]   -> <run>/lab/transport_<surface>.json
"""
import argparse
import copy
import json
from pathlib import Path

import mujoco
import numpy as np

from physics.world import World

G = 9.81
SURFACES = {"wood": 1.0, "plastic": 0.75, "rubber": 1.8}          # x the object's friction estimate (vs wood)
PULSE, SETTLE = 0.6, 0.8                                          # seconds of braking, then settling
DIRS = {"forward": 0.0, "sideways": 90.0, "diagonal": 45.0}
FRICTION_RANGE = (0.7, 1.0, 1.3)


def _one(o, polygon, floor_z, radius, ratio, angle_deg, mu):
    """Apply a/g = ratio along angle for PULSE s. Returns ('ok'|'slide'|'tip', slide distance, tilt)."""
    obj = copy.deepcopy(o)
    obj["friction"] = float(np.clip(mu, 0.05, 2.0))
    w = World([obj], polygon, floor_z=floor_z)
    w.m.opt.cone = mujoco.mjtCone.mjCONE_ELLIPTIC                # same grip in every direction (pyramid: -30% diagonal)
    k = obj["id"]
    w.step(0.3)                                                  # settle on the table first
    p0 = w.pose(k)[0].copy()
    a = np.radians(angle_deg)
    w.m.opt.gravity[:] = (ratio * w.gravity * np.cos(a), ratio * w.gravity * np.sin(a), -w.gravity)
    w.step(PULSE)
    moved = float(np.linalg.norm((w.pose(k)[0] - p0)[:2]))
    w.m.opt.gravity[:] = (0, 0, -w.gravity)
    w.step(SETTLE)
    q = w.pose(k)[1]
    tilt = float(np.degrees(np.arccos(np.clip(1 - 2 * (q[1] ** 2 + q[2] ** 2), -1, 1))))
    slid = moved > 0.02 * radius                                 # ~5 mm on a typical table
    if tilt > 25:
        return "tip", moved, tilt
    return ("slide" if slid else "ok"), moved, tilt


def limit(o, polygon, floor_z, radius, angle, mu, lo=0.0, hi=1.5, iters=7):
    """Smallest a/g in [lo, hi] that is not 'ok', and what happens there."""
    what = "ok"
    for _ in range(iters):
        mid = (lo + hi) / 2
        r, _, _ = _one(o, polygon, floor_z, radius, mid, angle, mu)
        if r == "ok":
            lo = mid
        else:
            hi, what = mid, r
    if what == "ok":                                             # never failed up to 1.5 g
        return hi, "holds"
    return hi, what


def run(sc, names, surface="wood", run_dir=None):
    from lab.objects import lab_objects
    out = {"surface": surface, "objects": {}}
    for o, shape in lab_objects(sc, run_dir) if run_dir else [(o, "scan hull") for o in sc.objects]:
        k = o["id"]
        rows = []
        for dname, ang in DIRS.items():
            for f in FRICTION_RANGE:
                mu = o["friction"] * SURFACES[surface] * f
                lim, what = limit(o, sc.polygon, sc.floor_z, sc.radius, ang, mu)
                rows.append({"direction": dname, "friction": round(mu, 2), "limit_g": round(lim, 3), "fails_by": what})
        worst = min(rows, key=lambda r: r["limit_g"])
        nominal = [r for r in rows if r["friction"] == round(o["friction"] * SURFACES[surface], 2)]
        out["objects"][str(k)] = {"name": names.get(k, f"object {k}"), "shape": shape, "trials": rows,
                                  "worst_g": worst["limit_g"], "worst_mode": worst["fails_by"],
                                  "worst_direction": worst["direction"],
                                  "nominal_g": min(r["limit_g"] for r in nominal) if nominal else None}
    ws = [v["worst_g"] for v in out["objects"].values()]
    out["safe_g"] = round(0.8 * min(ws), 3) if ws else None       # 20% margin under the weakest object
    out["safe_ms2"] = round(out["safe_g"] * G, 2) if ws else None
    return out


def summary(res):
    lines = [f"Transport test on {res['surface']}: keep acceleration/braking under {res['safe_ms2']} m/s² "
             f"({res['safe_g']} g, 20% margin)."]
    for k, v in sorted(res["objects"].items(), key=lambda kv: kv[1]["worst_g"]):
        how = {"slide": "slides", "tip": "tips over", "holds": "holds up to 1.5 g"}[v["worst_mode"]]
        lines.append(f"- {v['name']}: {how} at {v['worst_g']} g ({v['worst_g'] * G:.1f} m/s², "
                     f"{v['worst_direction']}, low-friction case; shape: {v.get('shape', 'scan hull')})")
    tippers = [v["name"] for v in res["objects"].values() if v["worst_mode"] == "tip"]
    sliders = [v["name"] for v in res["objects"].values() if v["worst_mode"] == "slide"]
    if sliders and res["surface"] != "rubber":
        lines.append(f"Suggestion: a rubber mat raises the slide limit for {', '.join(sliders)}.")
    if tippers:
        lines.append(f"Suggestion: {', '.join(tippers)} tip first; lay them down, add a fence, or accelerate gently.")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("--src", default="thrift_sfm")
    ap.add_argument("--surface", default="wood", choices=list(SURFACES))
    a = ap.parse_args()
    from viewer.app import Scene
    sc = Scene(a.run_dir, a.src)
    names = {int(k): v[0] for k, v in json.loads((Path(a.run_dir) / "names.json").read_text()).items()}
    res = run(sc, names, a.surface, a.run_dir)
    d = Path(a.run_dir) / "lab"
    d.mkdir(exist_ok=True)
    (d / f"transport_{a.surface}.json").write_text(json.dumps(res, indent=1))
    print(summary(res))


if __name__ == "__main__":
    main()
