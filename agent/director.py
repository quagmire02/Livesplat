"""Director mode: goal -> Muse proposes several action plans -> each is simulated headless -> Muse scores the
real outcomes and revises -> the best plan is played back live.

The simulator is the judge of what happens; Muse only proposes plans and scores the reported outcomes.
Compute-heavy by design (hackathon credit): about 1 call per round, several plans per call.
Run headless (WSL): python -m agent.director <run_dir> "stack everything into a tower" [--provider baseten]
"""
import json
import time
from pathlib import Path

import mujoco
import numpy as np

from agent.planner import scene_summary
from agent.schema import ACTIONS, TOOLS, validate

SETTLE = 2.5                        # seconds of physics after the last action
DEFAULT_WAIT = 0.6                  # seconds between actions if the plan gives none

PROMPT = """You direct a deterministic physics simulation of real scanned objects on a table.
GOAL: {goal}

Objects: {objects}
Table: roughly a disc of radius {radius:.2f} scene units around the origin; z is up; the table top is z=0.
Lengths are scene units (about 1.65 scene units per metre); speeds in scene units/s.
Current state:
{state}

Available actions (JSON schema):
{tools}
Each action may have "wait": seconds to let physics run before the next action (default {wait}).
After the last action physics runs {settle}s, then the outcome is measured.
Physics facts: push sets an initial speed v; the object then slides about v*v/(2*friction*g) before friction stops
it (g = {g:.2f}). So moving an object a distance d needs v ~ sqrt(2*friction*g*d) (e.g. d=0.2, friction 0.5 -> v~1.4).
Tall, narrow objects tip over when pushed fast if friction is high; a push on one object can hit another.
slide is a steady hand-guided move (stays upright, stops after the distance; with a target and no distance it stops
just short of the target): use it for precise placement. push is a flick: use it to knock, launch or hit things.
surface_gaps = horizontal gap between object surfaces (0 = touching). Use the measured outcomes of earlier plans
to calibrate speeds: if a plan moved too little, raise the speed by the square-root rule, not a few percent.
{history}
Reply with JSON only:
{{"evaluation": [{{"plan": <index of a plan from the previous round>, "score": 0-10, "why": "short"}}],
  "goal_met_by": <index of a previous plan that fully meets the goal, or null>,
  "plans": [{{"idea": "short", "actions": [{{"name": "...", "args": {{...}}, "wait": 0.6}}]}}]}}
Give {n} plans that differ in strategy (order, speeds, directions, heights). Use object ids as strings.
Leave "evaluation" empty in the first round. Score only from the measured outcomes, not from the plan text."""


def _tools_text():
    return json.dumps([{"name": t["function"]["name"], "description": t["function"]["description"],
                        "args": t["function"]["parameters"]["properties"]}
                       for t in TOOLS if t["function"]["name"] in ACTIONS])


def _parse(text):
    try:
        return json.loads(text[text.index("{"): text.rindex("}") + 1])
    except ValueError:
        return None


def _chat(msgs, provider, max_tokens):
    if provider == "baseten":
        from backends import baseten
        r = baseten.chat(msgs, max_tokens=max_tokens)
    else:
        from backends.meta import client
        r = client.spark_chat(msgs, max_tokens=max_tokens)
    if r.get("_status") != 200:
        raise RuntimeError(f"Muse error: {str(r)[:300]}")
    c = r["choices"][0]
    if not (c["message"].get("content") or "").strip() and c.get("finish_reason") == "length" and max_tokens < 16000:
        return _chat(msgs, provider, max_tokens * 2)    # all tokens went to reasoning: once more with more room
    return c["message"].get("content") or ""


# ---- measuring ------------------------------------------------------------------
def state(ctrl):
    """Measured state of every object: position, top/bottom height, tilt, table, what it rests on."""
    ids = [o["id"] for o in ctrl.sc.objects if not o.get("removed")]
    hulls = {k: ctrl._world_hull(k) for k in ids}
    out = {}
    for k in ids:
        p = ctrl.w.pose(k)[0]
        h = hulls[k]
        on = None
        for b in ids:                                   # resting on b: low end near b's top, centre over b
            if b == k:
                continue
            lo, hi = hulls[b][:, :2].min(0), hulls[b][:, :2].max(0)
            if (abs(h[:, 2].min() - hulls[b][:, 2].max()) < 0.03 * ctrl.sc.radius
                    and np.all(p[:2] > lo) and np.all(p[:2] < hi)):
                on = b
        o = ctrl.obj(k)
        out[k] = {"name": ctrl.label(k), "friction": round(float(o["friction"]), 2), "mass_kg": round(float(o["mass"]), 3),
                  "footprint": round(float(np.ptp(h[:, :2], 0).max()), 3), "xy": [round(float(p[0]), 3), round(float(p[1]), 3)],
                  "bottom": round(float(h[:, 2].min()), 3), "top": round(float(h[:, 2].max()), 3),
                  "tilt_deg": round(ctrl._tilt(k), 1), "on_table": bool(ctrl._on_table(k)),
                  "resting_on": str(on) if on is not None else ("table" if ctrl._on_table(k) else "nothing (fell)")}
    keys = sorted(out)
    gaps = {f"{a}-{b}": round(float(np.linalg.norm(ctrl.w.pose(a)[0][:2] - ctrl.w.pose(b)[0][:2])), 3)
            for i, a in enumerate(keys) for b in keys[i + 1:]}
    surf = {}                                   # closest hull points, horizontal (0 = touching)
    for i, a in enumerate(keys):
        for b in keys[i + 1:]:
            A, B = hulls[a][:, :2], hulls[b][:, :2]
            surf[f"{a}-{b}"] = round(float(np.sqrt(((A[:, None] - B[None]) ** 2).sum(-1)).min()), 3)
    return {"objects": {str(k): v for k, v in out.items()}, "centre_distances": gaps, "surface_gaps": surf,
            "stack_height": round(max((v["top"] for v in out.values()), default=0.0), 3)}


def simulate(ctrl, actions):
    """Run one plan on a copy of the physics state. Returns (outcome dict, action log). Live state untouched."""
    live = ctrl.w.d
    d = mujoco.MjData(ctrl.w.m)
    mujoco.mj_copyData(d, ctrl.w.m, live)
    ctrl.w.d = d
    slides = {k: list(v) for k, v in ctrl.w.slides.items()}
    try:
        log = []
        for a in actions:
            call = {"name": a.get("name"), "args": a.get("args") or {}}
            if call["name"] not in ACTIONS:
                log.append(f"error: {call['name']} is not an action")
                continue
            res = ctrl.run(call)
            log.append(res)
            if not res.startswith("error:"):
                ctrl.w.step(float(np.clip(a.get("wait", DEFAULT_WAIT), 0.05, 5.0)))
        ctrl.w.step(SETTLE)
        return state(ctrl), log
    finally:
        ctrl.w.d = live
        ctrl.w.slides = slides


def _clean(plans, n):
    out = []
    for p in (plans or [])[:n]:
        acts = []
        for a in (p.get("actions") or [])[:8] if isinstance(p, dict) else []:
            if not isinstance(a, dict):
                continue
            args = dict(a.get("args") or {})
            for key in ("obj", "base", "target"):       # models may send ids as numbers
                if isinstance(args.get(key), (int, float)) and not isinstance(args[key], bool):
                    args[key] = str(int(args[key]))
            call = {"name": a.get("name"), "args": args}
            if a.get("name") in ACTIONS and not validate(call):
                acts.append({**call, "wait": a.get("wait", DEFAULT_WAIT)})
        if acts:
            out.append({"idea": str(p.get("idea", ""))[:200], "actions": acts})
    return out


# ---- the loop -------------------------------------------------------------------
def direct(goal, ctrl, provider="baseten", n=6, rounds=2, max_tokens=8000, progress=print, log_dir=None, lock=None,
           good=8):
    """Returns {"best": plan, "score", "met", "rounds": [...], "calls"}. Does not touch the live scene.
    lock: the viewer's physics lock (simulate swaps the world's data, so the live loop must not step meanwhile)."""
    import contextlib
    lock = lock or contextlib.nullcontext()
    with lock:
        start = state(ctrl)
    tried, history, calls = [], "", 0
    best = {"plan": None, "score": -1, "why": "", "outcome": None}
    met = False
    for rnd in range(rounds + 1):                       # the extra call only scores the last round
        last = rnd == rounds
        if last and not tried:
            break
        progress("Checking which idea worked best…" if last or rnd else "Thinking of ways to do it…")
        msg = PROMPT.format(goal=goal, objects=scene_summary(ctrl), radius=ctrl.sc.radius, g=ctrl.w.gravity,
                            state=json.dumps(start), tools=_tools_text(), wait=DEFAULT_WAIT, settle=SETTLE,
                            history=history, n=0 if last else n)
        reply = _parse(_chat([{"role": "user", "content": msg}], provider, max_tokens))
        calls += 1
        if reply is None:
            progress("Muse reply was not JSON; retrying the round")
            continue
        prev = tried[-1] if tried else []
        for e in reply.get("evaluation") or []:
            try:
                i, s = int(e["plan"]), float(e["score"])
            except (KeyError, TypeError, ValueError):
                continue
            if 0 <= i < len(prev):
                prev[i]["score"], prev[i]["why"] = s, str(e.get("why", ""))[:200]
                if s > best["score"]:
                    best = {"plan": prev[i]["plan"], "score": s, "why": prev[i]["why"], "outcome": prev[i]["outcome"]}
        g = reply.get("goal_met_by")
        if isinstance(g, (int, float)) and 0 <= int(g) < len(prev):
            p = prev[int(g)]
            best = {"plan": p["plan"], "score": max(p.get("score", 10), best["score"]), "why": p.get("why", ""),
                    "outcome": p["outcome"]}
            met = True
            progress(f"Goal met by plan {int(g) + 1}: {p['plan']['idea']}")
            break
        if last or best["score"] >= good:                # good enough: do not spend another round
            break
        plans = _clean(reply.get("plans"), n)
        if not plans:
            progress("Muse gave no valid plans")
            break
        progress(f"Trying {len(plans)} ideas in the simulator…")
        batch = []
        for p in plans:
            with lock:
                outcome, alog = simulate(ctrl, p["actions"])
            batch.append({"plan": p, "outcome": outcome, "log": alog})
        tried.append(batch)
        history = "\nPREVIOUS ROUND (plan index: plan -> measured outcome):\n" + "\n".join(
            f"{i}: {json.dumps(b['plan'])} -> {json.dumps(b['outcome'])} log: {b['log']}" for i, b in enumerate(batch)) + \
            "\nEvaluate these plans, then propose improved plans (keep the best idea, fix what failed)."
    result = {"goal": goal, "provider": provider, "met": met, "best": best, "calls": calls,
              "plans_simulated": sum(len(b) for b in tried),
              "rounds": [[{k: b[k] for k in ("plan", "outcome", "log")} | {"score": b.get("score"), "why": b.get("why")}
                          for b in batch] for batch in tried]}
    if log_dir:
        d = Path(log_dir)
        d.mkdir(parents=True, exist_ok=True)
        (d / f"{time.strftime('%Y%m%d-%H%M%S')}.json").write_text(json.dumps(result, indent=1, default=str))
    return result


def play(plan, ctrl, lock, before=None):
    """Perform the plan live: each action under the lock, then let the viewer's physics run its wait time."""
    out = []
    for a in plan["actions"]:
        with lock:
            if before:
                before()
            out.append(ctrl.run({"name": a["name"], "args": a["args"]}))
        time.sleep(float(np.clip(a.get("wait", DEFAULT_WAIT), 0.05, 5.0)))
    return out


def summary(result):
    b = result["best"]
    if b["plan"] is None:
        return f"Director: no working plan after {result['plans_simulated']} simulated plans."
    head = "Done" if result["met"] or b["score"] >= 8 else f"Closest I got ({b['score']:.0f}/10)"
    return f"{head}: tried {result['plans_simulated']} ideas in the simulator, playing the best one."


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("goal")
    ap.add_argument("--src", default="thrift_sfm")
    ap.add_argument("--provider", default="baseten", choices=["baseten", "meta"])
    ap.add_argument("--plans", type=int, default=6)
    ap.add_argument("--rounds", type=int, default=2)
    a = ap.parse_args()
    from agent.controller import Controller
    from physics.world import World
    from viewer.app import Scene
    sc = Scene(a.run_dir, a.src)
    w = World(sc.objects, sc.polygon, floor_z=sc.floor_z)
    ctrl = Controller(sc, w, a.run_dir)
    r = direct(a.goal, ctrl, a.provider, a.plans, a.rounds, log_dir=Path(a.run_dir) / "director")
    print(summary(r))
    print(json.dumps(r["best"], indent=1, default=str))


if __name__ == "__main__":
    main()
