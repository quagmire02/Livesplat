"""Gripper fit test (no arm needed): does the object fit this gripper, where, and will it hold?

Parallel jaws: the object's full shape (3D model if built, else the scan) is sliced at 15..85% of its height; in
each slice the narrowest width over 18 angles is the candidate grip. It fits if width + 5 mm clearance <= the
opening. It holds if 2 * mu * F >= m * (g + 0.5 g lift) * 1.5 safety, mu = sqrt(pad * object friction).
Suction: a patch fits if the surface under the cup deviates <= the cup's tolerance from flat (a 5 cm can side
is too curved for a 30 mm cup) and the material is not porous; it holds if vacuum * area * efficiency >= the
same load. Every candidate is tried over mass x0.7/1/1.3, friction x0.7/1/1.3 and, unless the scale is
calibrated, size x(1 -+ spread): the success rate is over those 27 cases.

CLI: python -m lab.gripper <run> [--gripper panda_hand|robotiq_2f85|so100_jaw|suction_30mm|my.yaml]
"""
import argparse
import itertools
import json
from pathlib import Path

import numpy as np
import yaml

G, LIFT, SAFETY, CLEAR = 9.81, 0.5, 1.5, 0.005
GRIPPERS = Path(__file__).parent / "grippers"
POROUS = ("paper", "cardboard", "fabric", "cloth", "foam", "sponge", "tissue")


def load_gripper(name):
    p = Path(name)
    if not p.suffix:
        p = GRIPPERS / f"{name}.yaml"
    return {**yaml.safe_load(p.read_text()), "id": p.stem}


def _cases(spread, calibrated):
    sizes = (1.0,) if calibrated else (1 - spread, 1.0, 1 + spread)
    return list(itertools.product((0.7, 1.0, 1.3), (0.7, 1.0, 1.3), sizes))


def _slices(pts):
    """[(height fraction, angle deg, width in scene units)]: narrowest width per height band."""
    H = pts[:, 2].max()
    out = []
    for f in np.arange(0.15, 0.86, 0.1):
        band = pts[np.abs(pts[:, 2] - f * H) < 0.05 * H][:, :2]
        if len(band) < 20:
            continue
        band = band - band.mean(0)
        ang = np.radians(np.arange(0, 180, 10))
        widths = [np.ptp(band @ np.array([np.cos(a), np.sin(a)])) for a in ang]
        i = int(np.argmin(widths))
        out.append((round(float(f), 2), int(np.degrees(ang[i])), float(widths[i])))
    return out


def parallel(o, pts, gr, mpu, spread, calibrated):
    cands = []
    for f, ang, w in _slices(pts):
        ok = fit_fail = hold_fail = 0
        cases = _cases(spread, calibrated)
        for mm, fm, sm in cases:
            width = w * mpu * sm
            mass = o["mass"] * mm                             # Muse's real-product estimate: not tied to scan scale
            mu = np.sqrt(gr["pad_friction"] * o["friction"] * fm)
            fits = gr.get("min_opening_m", 0) <= width <= gr["max_opening_m"] - CLEAR
            holds = 2 * mu * gr["grip_force_n"] >= mass * (G + LIFT * G) * SAFETY
            ok += fits and holds
            fit_fail += not fits
            hold_fail += fits and not holds
        need = o["mass"] * (G + LIFT * G) * SAFETY / (2 * np.sqrt(gr["pad_friction"] * o["friction"]))
        cands.append({"height_frac": f, "angle_deg": ang, "width_mm": round(w * mpu * 1000, 1),
                      "success": round(ok / len(cases), 2), "force_needed_n": round(float(need), 1),
                      "fails": "size" if fit_fail >= hold_fail else "grip force"})
    return sorted(cands, key=lambda c: (-c["success"], abs(c["height_frac"] - 0.5)))


def suction(o, mesh_v, mesh_n, gr, mpu, spread, calibrated, material):
    if any(p in str(material).lower() for p in POROUS):
        return [], f"{material} is porous: a suction cup will not seal"
    from scipy.spatial import cKDTree
    r = gr["cup_diameter_m"] / 2 / mpu                         # cup radius in scene units
    tree = cKDTree(mesh_v)
    rng = np.random.default_rng(0)
    idx = rng.choice(len(mesh_v), size=min(400, len(mesh_v)), replace=False)
    H = mesh_v[:, 2].max()
    cands = []
    for i in idx:
        hf = mesh_v[i, 2] / H
        nz = abs(mesh_n[i, 2])
        if hf < 0.1 or (nz > 0.8 and hf < 0.6) or 0.5 <= nz <= 0.8:
            continue                                           # underside / table / slanted shoulders
        where = "top" if nz > 0.8 else "side"
        nb = mesh_v[tree.query_ball_point(mesh_v[i], r)]
        if len(nb) < 12:
            continue
        c = nb.mean(0)
        n = np.linalg.svd(nb - c, full_matrices=False)[2][2]
        dev = np.abs((nb - c) @ n).max() * mpu
        if dev > gr["max_curvature_dev_m"]:
            continue
        cands.append({"where": where, "height_frac": round(float(hf), 2),
                      "flatness_mm": round(dev * 1000, 2)})
    if not cands:
        return [], f"no patch flat enough for a {gr['cup_diameter_m'] * 1000:.0f} mm cup"
    force = gr["vacuum_kpa"] * 1000 * np.pi * (gr["cup_diameter_m"] / 2) ** 2 * gr["efficiency"]
    cases = _cases(spread, calibrated)
    ok = sum(force >= o["mass"] * mm * (G + LIFT * G) * SAFETY for mm, _, _ in cases)
    best = min(cands, key=lambda c: (c["where"] != "top", c["flatness_mm"]))
    best.update(success=round(ok / len(cases), 2), force_available_n=round(float(force), 1),
                force_needed_n=round(float(o["mass"] * (G + LIFT * G) * SAFETY), 1), patches=len(cands))
    return [best], ""


def run(sc, run_dir, gripper, names, props):
    from lab.objects import lab_objects
    from measure.scale import load
    s = load(run_dir)
    mpu, spread = s.get("metres_per_unit") or 0.6, s.get("spread") or 0.1
    calibrated = str(s.get("source", "")).startswith("user")
    gr = load_gripper(gripper)
    out = {"gripper": gr["name"], "scale": {"metres_per_unit": mpu, "spread": spread, "calibrated": calibrated,
                                            "source": s.get("source")}, "objects": {}}
    for o, shape in lab_objects(sc, run_dir):
        k = o["id"]
        mesh_v, mesh_n = _dense(run_dir, k, o)
        if gr["type"] == "suction":
            cands, why = suction(o, mesh_v, mesh_n, gr, mpu, spread, calibrated, props.get(str(k), {}).get("material"))
        else:
            cands, why = parallel(o, mesh_v, gr, mpu, spread, calibrated), ""
        best = cands[0] if cands else None
        out["objects"][str(k)] = {"name": names.get(k, f"object {k}"), "shape": shape, "best": best,
                                  "candidates": cands[:6], "why_not": why,
                                  "suggestion": _suggest(o, best, gr, mpu, why)}
    return out


def _dense(run_dir, k, o):
    """Dense surface points + normals (object frame, z-up): the 3D model if built, else the scan splats."""
    p = Path(run_dir) / "gen" / f"model_{k}" / "model.glb"
    if p.exists():
        import trimesh
        m = trimesh.load(p, force="mesh")
        v = np.asarray(m.vertices, float)
        n = np.asarray(m.vertex_normals, float)
        v, n = np.c_[v[:, 0], -v[:, 2], v[:, 1]], np.c_[n[:, 0], -n[:, 2], n[:, 1]]
        v[:, 2] -= v[:, 2].min()
        return v, n
    pts = o["capture_means"][o["body_ids"]] if "body_ids" in o else o["capture_means"]
    return np.asarray(pts, float), np.tile([0.0, 0.0, 1.0], (len(pts), 1))


def _suggest(o, best, gr, mpu, why):
    if gr["type"] == "suction":
        if best is None:
            return f"Use a parallel gripper ({why})." if "porous" in why else \
                f"Try a smaller or bellows cup ({why}), or a parallel gripper."
        if best["success"] < 1:
            return f"Cup force {best['force_available_n']} N is marginal; use a larger cup or stronger vacuum."
        return (f"Pick from the {best['where']}" + (f" at {int(best['height_frac'] * 100)}% height."
                                                     if best["where"] == "side" else "."))
    if best is None or best["success"] == 0:
        narrow = best["width_mm"] if best else None
        if best and best["width_mm"] / 1000 > gr["max_opening_m"] - CLEAR:
            return (f"Too wide: narrowest grip {narrow} mm vs {gr['max_opening_m'] * 1000:.0f} mm opening. "
                    "Use a wider gripper (e.g. Robotiq 2F-85) or grip a narrower part.")
        if best:
            pads = best["force_needed_n"] * np.sqrt(gr["pad_friction"]) / np.sqrt(0.8)
            return (f"Fits, but needs {best['force_needed_n']} N of grip vs {gr['grip_force_n']} N. "
                    f"Rubber/TPU pads (mu 0.8) would need ~{pads:.0f} N.")
        return "No grip found."
    if best["success"] < 1 and best.get("fails") == "size":
        return (f"Works in {int(best['success'] * 100)}% of cases: {best['width_mm']} mm is close to the "
                f"{gr['max_opening_m'] * 1000:.0f} mm opening, so it depends on the true size. Calibrate the scale "
                "(type the object's real size) or use a wider gripper.")
    if best["success"] < 1:
        return (f"Works in {int(best['success'] * 100)}% of cases (heavy or slippery ones slip); rubber/TPU pads "
                f"or more grip force (> {best['force_needed_n']} N) make it reliable.")
    return f"Grip at {int(best['height_frac'] * 100)}% height, {best['width_mm']} mm wide: reliable in all cases."


def summary(res):
    sc = res["scale"]
    lines = [f"{res['gripper']} fit test (scale {'calibrated' if sc['calibrated'] else 'estimated ±' + str(int(sc['spread'] * 100)) + '%'})"]
    for v in res["objects"].values():
        b = v["best"]
        if b is None:
            lines.append(f"- {v['name']}: no grip. {v['suggestion']}")
        elif "width_mm" in b:
            lines.append(f"- {v['name']}: {int(b['success'] * 100)}% at {int(b['height_frac'] * 100)}% height, "
                         f"{b['width_mm']} mm wide. {v['suggestion']}")
        else:
            lines.append(f"- {v['name']}: {int(b['success'] * 100)}% from the {b['where']} "
                         f"({b['patches']} flat patches). {v['suggestion']}")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("--src", default="thrift_sfm")
    ap.add_argument("--gripper", nargs="+", default=["so100_jaw", "panda_hand", "robotiq_2f85", "suction_30mm"])
    a = ap.parse_args()
    from viewer.app import Scene
    sc = Scene(a.run_dir, a.src)
    run_p = Path(a.run_dir)
    names = {int(k): v[0] for k, v in json.loads((run_p / "names.json").read_text()).items()}
    props = json.loads((run_p / "props.json").read_text()) if (run_p / "props.json").exists() else {}
    (run_p / "lab").mkdir(exist_ok=True)
    for g in a.gripper:
        res = run(sc, a.run_dir, g, names, props)
        (run_p / "lab" / f"gripper_{load_gripper(g)['id']}.json").write_text(json.dumps(res, indent=1))
        print(summary(res), "\n")


if __name__ == "__main__":
    main()
