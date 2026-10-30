"""Lab advisor: turns a grasp test into something a lab learns from.

1. What decides it: from every recorded trial, which uncertain factor (object mass, grip friction, true size)
   moves the success rate most, with the numbers ("friction x0.7: 0%, x1.3: 100%; mass barely matters").
2. Muse's grip: Muse sees the object's photos and the gripper and picks a grip (top/side, height band, what to
   avoid). The simulator runs it on the same 9 cases as the planner's choice: does the model's judgement help?
3. Experiments: Muse reads the results and proposes up to 3 changes from a fixed menu (fingertip pads, grip
   force, grip height, top/side, base closer). Each is simulated on the same cases; the report shows measured
   before -> after. Muse only proposes; the simulator decides.
"""
import json
from pathlib import Path

import numpy as np

from lab.grasp import MASS_F, FRIC_F, rate, setup

MENU = """Allowed changes (pick the ones most likely to help, max 3, no duplicates):
- {"change": "pads", "value": fingertip friction 0.6-1.2 (rubber/TPU pads ~0.9)}
- {"change": "grip_force", "value": stiffness multiplier 1.5-4 (stronger servo / higher current limit)}
- {"change": "grip_height", "value": [lo, hi] fractions of the object height for a side grasp}
- {"change": "mode", "value": "top" or "side"}
- {"change": "base_closer", "value": metres 0.05-0.2 to move the arm base toward this object}"""


def _muse(content, tokens=2500):
    from backends.meta import client
    r = client.spark_chat([{"role": "user", "content": content}], max_tokens=tokens, extra={"reasoning_effort": "low"})
    from backends.jsonfix import reply_json
    j = reply_json(r)
    if isinstance(j, dict):
        return j
    else:
        return {}


def _photos(run, k, n=3):
    from PIL import Image
    from backends.meta import client
    d = Path(run) / "gen" / f"model_{k}"
    paths = [d / f"{v}.png" for v in ("front", "left", "back", "right") if (d / f"{v}.png").exists()]
    paths = paths or [p for p in [Path(run) / f"crop_obj{k}.jpg"] if p.exists()]
    out = []
    for p in paths[:n]:
        im = Image.open(p).convert("RGB")
        im.thumbnail((384, 384))
        out.append({"type": "image_url", "image_url": {"url": client.image_data_uri(im, quality=85)}})
    return out


def decides(rec):
    """Which factor moves the success rate most (best mode's trials)."""
    md = rec["modes"].get(rec["best_mode"], {})
    cases = md.get("cases", [])
    if not cases:
        return ""
    total = np.mean([c["ok"] for c in cases])
    if total in (0.0, 1.0):
        reasons = ", ".join(rec["modes"][rec["best_mode"]].get("reasons", {})) or "works"
        return (f"nothing in the tested range changes it: it {'always works' if total == 1 else 'always fails'}"
                + ("" if total == 1 else f" ({reasons})") + ".")
    parts = []
    for key, label in (("fric_f", "grip friction"), ("mass_f", "object mass"), ("scale_f", "true size")):
        levels = sorted({c[key] for c in cases})
        if len(levels) < 2:
            continue
        rates = {lv: np.mean([c["ok"] for c in cases if c[key] == lv]) for lv in levels}
        parts.append((max(rates.values()) - min(rates.values()), label, rates))
    parts.sort(key=lambda p: -p[0])
    top = parts[0]
    txt = f"{top[1]} (" + ", ".join(f"x{lv:g}: {int(r * 100)}%" for lv, r in top[2].items()) + ")"
    weak = [p[1] for p in parts[1:] if p[0] < 0.2]
    return txt + (f"; {' and '.join(weak)} barely matter" if weak else "") + "."


def _baseline(rec):
    md = rec["modes"].get(rec["best_mode"], {})
    nom = [c["ok"] for c in md.get("cases", []) if c["scale_f"] == 1.0]
    return round(float(np.mean(nom)), 2) if nom else rec["best_rate"]


def _env(res, robot, band=None, body=None, base_shift=None):
    """Nominal package with the test's base (optionally shifted) and an optional side-grasp band."""
    from sim import robot as srobot
    sim = Path(res["nominal_package"])
    pos, yaw = list(res["base"]["pos_m"]), np.radians(res["base"]["yaw_deg"])
    if base_shift is not None:
        pos = [pos[0] + base_shift[0], pos[1] + base_shift[1], pos[2]]
    srobot.GRIP_BANDS.clear()
    if band is not None:
        srobot.GRIP_BANDS[body] = tuple(band)
    try:
        spec, meta, rows, arm, _ = setup(sim, robot, (tuple(pos), yaw))
    finally:
        srobot.GRIP_BANDS.clear()
    return meta, rows, arm


def _measure(res, robot, body, mode, pad, kp_f=1.0, band=None, base_shift=None):
    meta, rows, arm = _env(res, robot, band, body, base_shift)
    row = next(r for r in rows if r["body"] == body)
    o = next(x for x in meta["objects"] if x["body"] == body)
    modes = [mode] if mode in ("top", "side") else ["top", "side"]
    best = (0.0, {})
    for md in modes:
        r, why = rate(arm, row, o, md, pad, kp_f)
        if r >= best[0]:
            best = (r, why)
    return best


def _facts(rec, res):
    from lab.grasp import _readable
    r0 = rec["reach"][0]
    top = rec["modes"].get("top", {})
    side = rec["modes"].get("side", {})
    H = rec["size_m"][2]
    txt = (f"top grasp: {'reachable' if r0['top_reachable'] else 'NOT reachable'}, grabs "
           f"{top.get('grasp_point', 'the top')} where it is {r0['top_width_mm']} mm wide"
           f"{'' if r0['fits_top'] else ' (too wide)'}, success {int(top.get('rate', 0) * 100)}% "
           f"({', '.join(top.get('reasons', {})) or 'no failures'}). Side grasp: "
           + (f"feasible at {r0['side_height_m'] / H * 100:.0f}% of the height" if r0["side_feasible"] else
              f"NOT feasible" + (f" (blocked by {', '.join(r0['side_blocked_by'])})" if r0["side_blocked_by"] else ""))
           + f", success {int(side.get('rate', 0) * 100)}%.")
    return _readable(txt, res)


def grip_hint(run, k, rec, res):
    g = res.get("gripper", {})
    q = (f'A robot arm ({res.get("robot_name")}) with a {g.get("type", "parallel")} gripper '
         f'(opens {g.get("max_opening_m", 0) * 1000:.0f} mm, fingertip friction {res["pad_friction"]}) must pick up '
         f'this object: "{rec["name"]}", {rec["mass_kg"]:.2f} kg, about '
         f'{" x ".join(f"{v * 100:.0f}" for v in rec["size_m"])} cm (W x D x H). Look at the photos.\n'
         f'What the motion planner found: {_facts(rec, res)}\n'
         'Where should it grip so it holds and lifts reliably? Use the planner facts (a side grasp that is blocked '
         'or out of reach cannot work; a top grasp grabs the topmost part, e.g. a pump or cap). Consider which '
         'parts are narrow, curved, slippery or likely to come off. Reply JSON only: {"mode": "top" or "side", '
         '"band": [lo, hi] fractions of the height to grip within (side grasps), "avoid": "part to avoid", '
         '"why": "under 20 words"}')
    j = _muse([{"type": "text", "text": q}] + _photos(run, k))
    band = j.get("band") if isinstance(j.get("band"), list) and len(j["band"]) == 2 else [0.3, 0.6]
    band = [float(np.clip(min(band), 0.05, 0.95)), float(np.clip(max(band), 0.1, 0.98))]
    mode = j.get("mode") if j.get("mode") in ("top", "side") else "side"
    return {"mode": mode, "band": band, "avoid": str(j.get("avoid", ""))[:60], "why": str(j.get("why", ""))[:160]}


def propose(rec, res, baseline):
    modes = {m: {"rate": v["rate"], "failures": v.get("reasons", {})} for m, v in rec["modes"].items()}
    q = (f'Lab grasp test in simulation. Arm: {res.get("robot_name")}, gripper opens '
         f'{res["gripper"]["max_opening_m"] * 1000:.0f} mm, fingertip friction {res["pad_friction"]}. Object: '
         f'"{rec["name"]}", {rec["mass_kg"]:.2f} kg, {" x ".join(f"{v * 100:.0f}" for v in rec["size_m"])} cm. '
         f'Results per grasp mode (success rate over mass x0.7-1.3, friction x0.7-1.3): {json.dumps(modes)}. '
         f'Best now: {int(baseline * 100)}%. What decides it: {rec.get("decides", "")}. '
         f'Planner facts: {_facts(rec, res)}\n{MENU}\nReply JSON only: {{"experiments": [{{"change": ..., '
         '"value": ..., "why": "under 15 words"}]}')
    return (_muse([{"type": "text", "text": q}]).get("experiments") or [])[:3]


def _run_change(res, robot, body, rec, e):
    ch, v = e.get("change"), e.get("value")
    pad, mode = res["pad_friction"], rec["best_mode"] if rec["best_rate"] > 0 else "any"
    try:
        if ch == "pads":
            p = float(np.clip(float(v), 0.3, 1.5))
            return f"fingertip pads (friction {p:g})", _measure(res, robot, body, mode, p)
        if ch == "grip_force":
            f = float(np.clip(float(v), 1.0, 6.0))
            return f"grip force x{f:g}", _measure(res, robot, body, mode, pad, kp_f=f)
        if ch == "grip_height":
            band = [float(np.clip(min(v), 0.05, 0.95)), float(np.clip(max(v), 0.1, 0.98))]
            return (f"side grasp at {int(band[0] * 100)}-{int(band[1] * 100)}% height",
                    _measure(res, robot, body, "side", pad, band=band))
        if ch == "mode" and v in ("top", "side"):
            return f"{v} grasp", _measure(res, robot, body, v, pad)
        if ch == "base_closer":
            dist = float(np.clip(float(v), 0.02, 0.3))
            b = np.array(res["base"]["pos_m"][:2])
            o = np.array(rec.get("pos_m", [0, 0])[:2])
            u = (o - b) / max(np.linalg.norm(o - b), 1e-9)
            return f"arm base {dist * 100:.0f} cm closer", _measure(res, robot, body, mode, pad,
                                                                    base_shift=(u * dist).tolist())
    except (TypeError, ValueError):
        pass
    return None, None


def advise(run, robot, res, threshold=0.9):
    """Adds decides / muse_grip / experiments to each object record in res (in place)."""
    sim = Path(res["nominal_package"])
    meta = json.loads((sim / "metadata.json").read_text())
    for body, rec in res["objects"].items():
        o = next(x for x in meta["objects"] if x["body"] == body)
        rec["pos_m"] = o["pose"]["pos"]
        from lab.grasp import _readable
        rec["decides"] = _readable(decides(rec), res)
        if rec["decides"].startswith("true size") and 0 < rec["best_rate"] < 0.9:
            rec["suggestion"] = (f"Works in {int(rec['best_rate'] * 100)}% of trials ({rec['best_mode']} grasp), and the "
                                 "true size decides it: type the object's real size (e.g. 'the can is 15 cm tall') "
                                 "to get a definite answer.")
        elif rec["decides"].startswith("grip friction") and 0 < rec["best_rate"] < 0.9:
            rec["suggestion"] = (f"Works in {int(rec['best_rate'] * 100)}% of trials; grip friction decides it: "
                                 "rubber/TPU fingertip pads should make it reliable (see the experiments).")
        baseline = _baseline(rec)
        k = int(o["id"])
        try:
            hint = grip_hint(run, k, rec, res)
            r, why = _measure(res, robot, body, hint["mode"], res["pad_friction"],
                              band=hint["band"] if hint["mode"] == "side" else None)
            rec["muse_grip"] = {**hint, "rate": r, "failures": why, "planner_nominal": baseline}
            print(f"  {rec['name']}: Muse grip {hint['mode']} {hint['band']} -> {r} (planner {baseline})", flush=True)
        except Exception as e:
            rec["muse_grip_error"] = str(e)[:200]
        if baseline >= threshold:
            continue
        exps = []
        for e in propose(rec, res, baseline):
            label, out = _run_change(res, robot, body, rec, e)
            if label:
                exps.append({"label": label, "rate": out[0], "failures": out[1], "baseline": baseline,
                             "why": str(e.get("why", ""))[:120], "change": e})
                print(f"  {rec['name']}: {label} -> {out[0]} (was {baseline})", flush=True)
        rec["experiments"] = sorted(exps, key=lambda x: -x["rate"])
    return res
