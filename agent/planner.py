"""Command text -> tool calls -> results.

backend "local": agent.parser (rules, free).
backend "meta" : Muse Spark tool calling with the TOOLS schema (cached + budget-guarded).
"""
import json

from agent import parser
from agent.schema import TOOLS

SYSTEM = ("You control a physics sandbox of real objects on a table. Use the tools to do what the user asks; "
          "chain several calls if needed. Directions are relative to the user's view. If the request "
          "cannot be done with the tools, call no tool and say briefly what is possible instead. "
          "For questions like 'what if' or 'will it fall', use simulate. Objects: {objects}. "
          "Gravity: {g:.2f} scene units/s2. Scene scale is uncalibrated; do not claim precise metric dimensions. "
          "Mass and friction are estimates unless the user supplied them.")


def scene_summary(ctrl):
    return "; ".join(f"id {o['id']} = {', '.join(ctrl.names[o['id']][:2])}"
                     + (" (removed)" if o.get("removed") else f" ({o['mass']:.2f} kg)")
                     for o in ctrl.sc.objects)


def run_meta(text, ctrl, max_rounds=3, default=None, cache_only=False, provider="meta"):
    """Tool-use loop: Spark calls tools, sees the results, and may continue (e.g. list, then stack).
    Returns (list of (call, result), final text). NOT yet tested with paid calls (multi-round)."""
    from backends.meta import client
    msgs = [{"role": "system", "content": SYSTEM.format(objects=scene_summary(ctrl), g=ctrl.w.gravity)},
            {"role": "user", "content": text}]
    if default is not None and default in ctrl.names:
        msgs[0]["content"] += f" Selected object: id {default} ({ctrl.label(default)}); 'it' refers to this selection."
    done, note = [], ""
    for _ in range(max_rounds):
        try:
            if provider == "baseten":                         # Muse Spark Open, open weights, Baseten credit
                from backends import baseten
                r = baseten.chat(msgs, tools=TOOLS, max_tokens=3000, cache_only=cache_only)
            else:
                r = client.spark_chat(msgs, tools=TOOLS, max_tokens=2000, cache_only=cache_only)
        except client.PermissionRequired as e:
            return done, str(e)
        if r.get("_status") != 200:
            return done, f"Spark error: {str(r)[:200]}"
        msg = r["choices"][0]["message"]
        note = (msg.get("content") or "").strip()
        tcs = msg.get("tool_calls") or []
        if not tcs:
            break
        msgs.append({"role": "assistant", "content": msg.get("content"), "tool_calls": tcs})
        for tc in tcs:
            try:
                call = {"name": tc["function"]["name"], "args": json.loads(tc["function"]["arguments"] or "{}")}
            except ValueError:
                call = {"name": tc["function"]["name"], "args": {"_bad_json": True}}
            for key in ("obj", "base", "target"):               # models may send object ids as numbers
                if isinstance(call["args"].get(key), (int, float)) and not isinstance(call["args"][key], bool):
                    call["args"][key] = str(int(call["args"][key]))
            res = ctrl.run(call)
            done.append((call, res))
            msgs.append({"role": "tool", "tool_call_id": tc.get("id", ""), "content": res})
        if all(c["name"] in ("list_objects", "measure") for c, _ in done[-len(tcs):]):
            continue                                     # it only looked: let it act on what it saw
        break
    return done, note


def handle(text, ctrl, backend="local", default=None):
    """Returns (list of (call, result) pairs, note)."""
    if backend == "baseten":
        return run_meta(text, ctrl, default=default, provider="baseten")
    if backend in ("meta", "cached"):
        return run_meta(text, ctrl, default=default, cache_only=backend == "cached")
    else:
        calls, unknown = parser.parse(text, ctrl.names, default)
        note = ("not understood: " + "; ".join(f"'{u}'" for u in unknown)) if unknown else ""
        if not calls and not unknown:
            note = "nothing to do"
    return [(c, ctrl.run(c)) for c in calls], note
