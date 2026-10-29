"""Agent tool definitions (OpenAI function format, used as-is for Muse Spark) and validation.

Every tool call is {"name": str, "args": dict}. validate() returns an error string or None.
Object arguments are object names or ids; the controller resolves them.
"""
import math

COLORS = {
    "red": (0.85, 0.12, 0.10), "green": (0.15, 0.70, 0.20), "blue": (0.12, 0.35, 0.90),
    "yellow": (0.95, 0.85, 0.10), "orange": (0.98, 0.50, 0.08), "purple": (0.55, 0.20, 0.75),
    "pink": (0.98, 0.45, 0.70), "white": (0.95, 0.95, 0.95), "black": (0.06, 0.06, 0.06),
    "gray": (0.50, 0.50, 0.50), "brown": (0.45, 0.28, 0.12), "gold": (0.90, 0.72, 0.20),
    "cyan": (0.10, 0.80, 0.85),
}
DIRECTIONS = ["left", "right", "forward", "back", "away", "toward"]

_OBJ = {"type": "string", "description": "object name or id, e.g. 'bottle' or '2'"}


def _fn(name, desc, props=None, req=None):
    return {"type": "function", "function": {"name": name, "description": desc, "parameters": {
        "type": "object", "properties": props or {}, "required": req or []}}}


TOOLS = [
    _fn("list_objects", "List the objects on the table with names, sizes and properties."),
    _fn("measure", "Size (width, depth, height), volume and mass of one object.", {"obj": _OBJ}, ["obj"]),
    _fn("push", "Slide an object across the table. Direction is relative to the viewer's camera.",
        {"obj": _OBJ, "direction": {"type": "string", "enum": DIRECTIONS},
         "speed": {"type": "number", "description": "m/s, 0.2 gentle .. 3 hard; default 0.6"},
         "target": {"type": "string", "description": "object to push toward/away from (optional)"}},
        ["obj", "direction"]),
    _fn("drop", "Lift an object and let it fall.", {"obj": _OBJ, "height": {"type": "number", "description": "m above its rest pose, default 0.3"}}, ["obj"]),
    _fn("throw", "Throw an object upward with spin.", {"obj": _OBJ, "speed": {"type": "number", "description": "m/s, default 2.5"}}, ["obj"]),
    _fn("stack", "Place obj on top of base and let physics settle it.", {"obj": _OBJ, "base": _OBJ}, ["obj", "base"]),
    _fn("set_mass", "Set an object's mass.", {"obj": _OBJ, "kg": {"type": "number"}}, ["obj", "kg"]),
    _fn("set_friction", "Set friction (0 ice .. 1.5 rubber).", {"obj": _OBJ, "mu": {"type": "number"}}, ["obj", "mu"]),
    _fn("set_bounce", "Set bounciness (0 dead .. 0.95 superball).", {"obj": _OBJ, "e": {"type": "number"}}, ["obj", "e"]),
    _fn("set_gravity", "Set gravity for the whole scene (Earth 9.81, Moon 1.62, Mars 3.71, Jupiter 24.8).",
        {"g": {"type": "number"}}, ["g"]),
    _fn("set_color", "Recolour an object, keeping its shading.", {"obj": _OBJ, "color": {"type": "string", "enum": list(COLORS)}}, ["obj", "color"]),
    _fn("scale", "Scale an object's size (mass scales with volume).", {"obj": _OBJ, "factor": {"type": "number", "description": "0.3 .. 3"}}, ["obj", "factor"]),
    _fn("remove", "Remove an object from the scene.", {"obj": _OBJ}, ["obj"]),
    _fn("restore", "Bring back a removed object and undo its colour/scale changes.", {"obj": _OBJ}, ["obj"]),
    _fn("reset", "Reset positions, colours, sizes, removed objects, properties and gravity to the capture."),
    _fn("simulate", "Predict without showing: apply actions, run physics for some seconds, report what happens.",
        {"actions": {"type": "array", "items": {"type": "object"}, "description": "list of {name, args} action calls"},
         "seconds": {"type": "number", "description": "default 2"}}, ["actions"]),
]
TOOL_NAMES = {t["function"]["name"] for t in TOOLS}
ACTIONS = {"push", "drop", "throw", "stack"}            # allowed inside simulate
LIMITS = {"speed": (0.05, 5.0), "height": (0.0, 2.0), "kg": (0.001, 50.0), "mu": (0.0, 2.0),
          "e": (0.0, 0.95), "g": (0.0, 50.0), "factor": (0.2, 4.0), "seconds": (0.1, 10.0)}


def validate(call):
    if not isinstance(call, dict):
        return "tool call must be an object"
    name, args = call.get("name"), call.get("args", {})
    if not isinstance(name, str) or name not in TOOL_NAMES:
        return f"unknown tool '{name}'"
    if not isinstance(args, dict):
        return f"{name}: args must be an object"
    spec = next(t["function"]["parameters"] for t in TOOLS if t["function"]["name"] == name)
    for r in spec["required"]:
        if r not in args:
            return f"{name}: missing '{r}'"
    for k, v in args.items():
        if k not in spec["properties"]:
            return f"{name}: unknown argument '{k}'"
        p = spec["properties"][k]
        if p.get("type") == "number":
            if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v):
                return f"{name}: '{k}' must be a finite number"
            lo, hi = LIMITS.get(k, (-1e9, 1e9))
            if not lo <= v <= hi:
                return f"{name}: '{k}'={v} outside [{lo}, {hi}]"
        if "enum" in p and v not in p["enum"]:
            return f"{name}: '{k}' must be one of {p['enum']}"
        if p.get("type") == "string" and (not isinstance(v, str) or not v.strip()):
            return f"{name}: '{k}' must be non-empty text"
        if p.get("type") == "array" and not isinstance(v, list):
            return f"{name}: '{k}' must be a list"
    if name == "simulate":
        if not 1 <= len(args["actions"]) <= 32:
            return "simulate: provide 1 to 32 actions"
        for c in args["actions"]:
            if not isinstance(c, dict) or not isinstance(c.get("name"), str) or c["name"] not in ACTIONS:
                return f"simulate: only {sorted(ACTIONS)} allowed inside"
            err = validate(c)
            if err:
                return f"simulate: {err}"
    return None
