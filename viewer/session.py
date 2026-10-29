"""Compact editor history and saved layouts. Geometry stays in the original scan.

Snapshots store poses, velocities, names and edits, never millions of Gaussian arrays.
Saved layouts are tied to a geometry fingerprint and loaded only on user request.
"""
import csv
import hashlib
import io
import json
from pathlib import Path

import numpy as np

from agent.schema import COLORS


class EditorSession:
    def __init__(self, controller, run_dir, source="thrift_sfm", limit=30):
        self.ctrl = controller
        self.path = Path(run_dir) / "editor_state.json"
        self.source = source
        self.limit = limit
        self.undo_stack, self.redo_stack = [], []
        self.baselines = {}
        self._register()

    def _register(self):
        for o in self.ctrl.sc.objects:
            if o["id"] not in self.baselines:
                self.baselines[o["id"]] = {
                    "hull": o["hull"].copy(),
                    "capture_means": o.get("capture_means", o["splat"]["means"]).copy(),
                    "parts": {n: {k: o[n][k].copy() for k in ("means", "scales")}
                              for n in ("splat",) if o.get(n) is not None},
                }

    def geometry_changed(self):
        """Object additions change topology; history from before that boundary is invalid."""
        self._register()
        self.undo_stack.clear()
        self.redo_stack.clear()

    def fingerprint(self):
        h = hashlib.sha256(self.source.encode())
        for k, base in sorted(self.baselines.items()):
            h.update(str(k).encode())
            h.update(np.asarray(base["hull"], dtype="<f4").tobytes())
            h.update(np.asarray(base["capture_means"], dtype="<f4").tobytes())
        return h.hexdigest()

    def snapshot(self):
        c = self.ctrl
        rows = []
        for o in c.sc.objects:
            k = o["id"]
            pose, vel = None, None
            if c.w.present(k):
                q, v = c.w._jnt(k)
                pose = c.w.d.qpos[q:q + 7].tolist()
                vel = c.w.d.qvel[v:v + 6].tolist()
            rows.append({"id": k, "names": list(c.names[k]), "pose": pose, "velocity": vel,
                         "scale": o["scale"], "color": o.get("color"), "removed": bool(o.get("removed")),
                         "mass": o["mass"], "friction": o["friction"], "bounce": o["bounce"],
                         "property_source": o.get("property_source", "scan estimates/defaults")})
        return {"format": "livesplat-editor/1", "source": self.source, "gravity": c.w.gravity,
                "time": float(c.w.d.time), "objects": rows}

    def checkpoint(self, label):
        self.record(label, self.snapshot())

    def record(self, label, state):
        self.undo_stack.append((label, state))
        self.undo_stack = self.undo_stack[-self.limit:]
        self.redo_stack.clear()

    def _validate(self, state):
        if not isinstance(state, dict) or state.get("format") != "livesplat-editor/1":
            raise ValueError("Unsupported saved layout")
        if state.get("source") != self.source:
            raise ValueError("This layout belongs to a different reconstruction")
        rows = state.get("objects")
        if not isinstance(rows, list) or any(not isinstance(o, dict) for o in rows):
            raise ValueError("Invalid objects in layout")
        ids = [o.get("id") for o in rows]
        if len(ids) != len(self.baselines) or set(ids) != set(self.baselines):
            raise ValueError("The scene's objects have changed; load a matching layout")

        def number(v, lo, hi):
            return type(v) in (int, float) and np.isfinite(v) and lo <= v <= hi

        if not number(state.get("gravity"), 0, 50) or not number(state.get("time"), 0, 1e12):
            raise ValueError("Invalid simulation settings")
        for row in rows:
            if not isinstance(row.get("names"), list) or not row["names"] or any(
                    not isinstance(n, str) or not n.strip() or len(n) > 120 for n in row["names"]):
                raise ValueError("Invalid object names")
            if row.get("color") is not None and row["color"] not in COLORS:
                raise ValueError("Invalid colour")
            if not isinstance(row.get("removed"), bool):
                raise ValueError("Invalid visibility")
            for field, lo, hi in (("scale", 0.05, 20), ("mass", 0.001, 50), ("friction", 0, 2), ("bounce", 0, 0.95)):
                if not number(row.get(field), lo, hi):
                    raise ValueError(f"Invalid {field}")
            if not row["removed"]:
                for field, n in (("pose", 7), ("velocity", 6)):
                    v = row.get(field)
                    if not isinstance(v, list) or len(v) != n or not all(number(x, -1e6, 1e6) for x in v):
                        raise ValueError(f"Invalid {field}")
                if abs(np.linalg.norm(row["pose"][3:]) - 1) > 0.01:
                    raise ValueError("Invalid rotation")

    def restore(self, state):
        self._validate(state)  # validate everything before modifying anything
        c = self.ctrl
        appearance_changed = []
        for row in state["objects"]:
            k = row["id"]
            o, base = c.obj(k), self.baselines[k]
            f = row["scale"]
            resized = f != o["scale"]
            recolored = row["color"] != o.get("color")
            if resized:
                o["hull"] = base["hull"] * f
                for n, arrays in base["parts"].items():
                    for key, values in arrays.items():
                        o[n][key] = values * f
            if resized or recolored:
                appearance_changed.append(k)
            for key in ("scale", "color", "removed", "mass", "friction", "bounce"):
                o[key] = row[key]
            o["property_source"] = row.get("property_source", "scan estimates/defaults")
            c.names[k] = list(row["names"])
            if recolored:
                o["splat"]["rgb"] = o["rgb0"].copy()
            if recolored and o["color"] is not None:
                # Avoid rebuilding Viser nodes until the full state is ready.
                callback, c.on_appearance = c.on_appearance, lambda k: None
                try:
                    c.t_set_color(str(k), o["color"])
                finally:
                    c.on_appearance = callback
        c.w.gravity = state["gravity"]
        c.w._build(keep=False)
        for row in state["objects"]:
            if not row["removed"]:
                q, v = c.w._jnt(row["id"])
                c.w.d.qpos[q:q + 7] = row["pose"]
                c.w.d.qvel[v:v + 6] = row["velocity"]
        c.w.d.time = state["time"]
        c.w.forward()
        for k in appearance_changed:
            c.on_appearance(k)

    def undo(self):
        if not self.undo_stack:
            return "Nothing to undo"
        label, state = self.undo_stack[-1]
        current = self.snapshot()
        self.restore(state)
        self.undo_stack.pop()
        self.redo_stack.append((label, current))
        return f"Undid {label}"

    def redo(self):
        if not self.redo_stack:
            return "Nothing to redo"
        label, state = self.redo_stack[-1]
        current = self.snapshot()
        self.restore(state)
        self.redo_stack.pop()
        self.undo_stack.append((label, current))
        return f"Redid {label}"

    def save(self):
        state = self.snapshot()
        state["geometry"] = self.fingerprint()
        text = json.dumps(state, indent=2, allow_nan=False)
        tmp = self.path.with_suffix(".json.tmp")
        tmp.write_text(text)
        tmp.replace(self.path)
        return self.path

    def load(self):
        state = json.loads(self.path.read_text())
        self._validate(state)
        if state.get("geometry") != self.fingerprint():
            raise ValueError("The reconstruction changed; this saved layout no longer matches")
        self.checkpoint("load layout")
        self.restore(state)

    def rename(self, k, name):
        name = name.strip().lower()
        if not name or len(name) > 80:
            raise ValueError("Use a name between 1 and 80 characters")
        if any(name in ns for i, ns in self.ctrl.names.items() if i != k):
            raise ValueError("Another object already uses that name")
        self.ctrl.names[k] = list(dict.fromkeys([name, f"object {k}", str(k), *self.ctrl.names[k]]))

    def inventory_csv(self):
        """Geometry measurements stay explicitly in scene units until physical calibration exists."""
        out = io.StringIO()
        writer = csv.writer(out)
        writer.writerow(["id", "name", "removed", "width_scene_units", "depth_scene_units", "height_scene_units",
                         "mass_kg", "friction", "property_source", "x_scene_units", "y_scene_units", "z_scene_units"])
        for o in self.ctrl.sc.objects:
            k = o["id"]
            pos = self.ctrl.w.pose(k)[0] if self.ctrl.w.present(k) else [None] * 3
            name = self.ctrl.label(k)
            if name.startswith(("=", "+", "-", "@")):
                name = "'" + name
            writer.writerow([k, name, bool(o.get("removed")), *np.ptp(o["hull"], axis=0).tolist(),
                             o["mass"], o["friction"], o.get("property_source", "scan estimates/defaults"), *pos])
        return out.getvalue()
