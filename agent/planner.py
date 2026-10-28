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
          "Gravity: {g:.2f} m/s2.")


def scene_summary(ctrl):
    return "; ".join(f"id {o['id']} = {', '.join(ctrl.names[o['id']][:2])}"
                     + (" (removed)" if o.get("removed") else f" ({o['mass']:.2f} kg)")
                     for o in ctrl.sc.objects)


def run_meta(text, ctrl, max_rounds=3):
    """Tool-use loop: Spark calls tools, sees the results, and may continue (e.g. list, then stack).
    Returns (list of (call, result), final text). NOT yet tested with paid calls (multi-round)."""
    from backends.meta import client
    msgs = [{"role": "system", "content": SYSTEM.format(objects=scene_summary(ctrl), g=ctrl.w.gravity)},
            {"role": "user", "content": text}]
    done, note = [], ""
    for _ in range(max_rounds):
        r = client.spark_chat(msgs, tools=TOOLS, max_tokens=2000)
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
            res = ctrl.run(call)
            done.append((call, res))
            msgs.append({"role": "tool", "tool_call_id": tc.get("id", ""), "content": res})
        if all(c["name"] in ("list_objects", "measure") for c, _ in done[-len(tcs):]):
            continue                                     # it only looked: let it act on what it saw
        break
    return done, note


def handle(text, ctrl, backend="local", default=None):
    """Returns (list of (call, result) pairs, note)."""
    if backend == "meta":
        return run_meta(text, ctrl)
    else:
        calls, unknown = parser.parse(text, ctrl.names, default)
        note = ("not understood: " + "; ".join(f"'{u}'" for u in unknown)) if unknown else ""
        if not calls and not unknown:
            note = "nothing to do"
    return [(c, ctrl.run(c)) for c in calls], note
