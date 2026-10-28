"""Executes validated tool calls on the physics World and the viewer Scene.

Directions are relative to the viewer camera (projected on the table plane).
Lengths are scene units (~metres from MapAnything; not calibrated without a reference object).
"""
import json
from pathlib import Path

import numpy as np

from agent.schema import COLORS, validate

NAMED_G = {"earth": 9.81, "moon": 1.62, "mars": 3.71, "jupiter": 24.8, "zero": 0.0}


def color_name(rgb):
    """Nearest named colour to a median RGB (for automatic object names)."""
    rgb = np.asarray(rgb)
    return min(COLORS, key=lambda c: np.sum((np.array(COLORS[c]) - rgb) ** 2))


class Controller:
    def __init__(self, scene, world, run_dir, on_appearance=None, camera=None):
        self.sc, self.w = scene, world
        self.on_appearance = on_appearance or (lambda k: None)
        self.camera = camera or (lambda: (np.array([1.0, -1.0, 1.0]), np.zeros(3)))
        self.names = self._names(Path(run_dir))
        for o in self.sc.objects:
            o.setdefault("rgb0", o["splat"]["rgb"].copy())
            o.setdefault("scale", 1.0)
            o.setdefault("defaults", {"mass": o["mass"], "friction": o["friction"], "bounce": o["bounce"]})

    # ---- names -------------------------------------------------------------
    def _names(self, run):
        """id -> list of names. names.json (from Spark/SAM or a fixture) first, then automatic ones."""
        given = {}
        p = run / "names.json"
        if p.exists():
            given = {int(k): v if isinstance(v, list) else [v] for k, v in json.loads(p.read_text()).items()}
        heights = {o["id"]: o["hull"][:, 2].max() for o in self.sc.objects}
        tallest = max(heights, key=heights.get)
        out = {}
        for o in self.sc.objects:
            c = color_name(np.median(o["splat"]["rgb"], 0))
            auto = [f"object {o['id']}", str(o["id"]), f"{c} object", c]
            if o["id"] == tallest:
                auto.append("tallest")
            out[o["id"]] = [n.lower() for n in given.get(o["id"], [])] + auto
        return out

    def label(self, k):
        return self.names[k][0]

    def resolve(self, ref):
        ref = str(ref).lower().strip().removeprefix("the ").strip()
        for k, ns in self.names.items():
            if ref in ns:
                return k
        for k, ns in self.names.items():                  # partial: "bottle" matches "lotion bottle"
            if any(ref in n or n in ref for n in ns if len(n) > 2):
                return k
        raise KeyError(f"no object called '{ref}'. Objects: " + ", ".join(self.label(k) for k in self.names))

    def obj(self, k):
        return next(o for o in self.sc.objects if o["id"] == k)

    # ---- geometry helpers --------------------------------------------------
    def _dir(self, direction, k, target=None):
        pos, look = self.camera()
        fwd = np.asarray(look, float) - np.asarray(pos, float)
        fwd[2] = 0
        fwd /= max(np.linalg.norm(fwd), 1e-9)
        right = np.array([fwd[1], -fwd[0], 0.0])
        if target is not None:
            d = self.w.pose(self.resolve(target))[0] - self.w.pose(k)[0]
            d[2] = 0
            d /= max(np.linalg.norm(d), 1e-9)
            return d if direction != "away" else -d
        return {"forward": fwd, "away": fwd, "back": -fwd, "toward": -fwd,
                "right": right, "left": -right}[direction]

    def _world_hull(self, k):
        from scipy.spatial.transform import Rotation
        p, q = self.w.pose(k)
        R = Rotation.from_quat([q[1], q[2], q[3], q[0]]).as_matrix()
        return self.obj(k)["hull"] @ R.T + p

    def _tilt(self, k):
        q = self.w.pose(k)[1]
        return float(np.degrees(2 * np.arccos(min(1.0, abs(q[0])))))

    def _on_table(self, k):
        from segment.cluster import in_polygon
        p = self.w.pose(k)[0]
        return bool(in_polygon(p[None, :2], self.sc.polygon)[0]) and p[2] > -0.05 * self.sc.radius

    # ---- dispatch ----------------------------------------------------------
    def run(self, call):
        """Validate + execute one call. Returns a short human-readable result string."""
        err = validate(call)
        if err:
            return f"error: {err}"
        try:
            return getattr(self, "t_" + call["name"])(**call.get("args", {}))
        except KeyError as e:
            return f"error: {e.args[0]}"

    # ---- tools -------------------------------------------------------------
    def t_list_objects(self):
        rows = []
        for o in self.sc.objects:
            st = "removed" if o.get("removed") else f"{o['mass']:.2f} kg, mu {o['friction']:.2f}"
            rows.append(f"{o['id']}: {self.label(o['id'])} ({st})")
        return "; ".join(rows)

    def t_measure(self, obj):
        k = self.resolve(obj)
        o = self.obj(k)
        ext = o["hull"].max(0) - o["hull"].min(0)
        return (f"{self.label(k)}: {ext[0] * 100:.1f} x {ext[1] * 100:.1f} x {ext[2] * 100:.1f} cm (W x D x H), "
                f"volume {o['volume'] * o['scale'] ** 3 * 1e6:.0f} cm3, mass {o['mass']:.2f} kg "
                f"(uncalibrated scale: needs a reference object)")

    def t_push(self, obj, direction, speed=0.6, target=None):
        k = self.resolve(obj)
        d = self._dir(direction, k, target)
        self.w.push(k, d[:2], speed)
        return f"pushed {self.label(k)} {direction}{' of ' + target if target else ''} at {speed:.1f} m/s"

    def t_drop(self, obj, height=0.3):
        k = self.resolve(obj)
        self.w.drop(k, height)
        return f"dropped {self.label(k)} from {height:.2f} m"

    def t_throw(self, obj, speed=2.5):
        k = self.resolve(obj)
        self.w.throw(k, (0.1 * speed, 0.0, speed), spin=(3.0, 1.0, 0.0))
        return f"threw {self.label(k)} up at {speed:.1f} m/s"

    def t_stack(self, obj, base):
        k, b = self.resolve(obj), self.resolve(base)
        if k == b:
            return "error: cannot stack an object on itself"
        top = self._world_hull(b)[:, 2].max()
        xy = self.w.pose(b)[0][:2]
        self.w.place(k, (xy[0], xy[1], top + 0.01), (1.0, 0.0, 0.0, 0.0))
        return f"placed {self.label(k)} on {self.label(b)}; physics decides if it stays"

    def t_set_mass(self, obj, kg):
        k = self.resolve(obj)
        self.w.set_mass(k, kg)
        return f"{self.label(k)} mass = {kg:.3g} kg"

    def t_set_friction(self, obj, mu):
        k = self.resolve(obj)
        self.w.set_friction(k, mu)
        return f"{self.label(k)} friction = {mu:.2f}"

    def t_set_bounce(self, obj, e):
        k = self.resolve(obj)
        self.w.set_bounce(k, e)
        return f"{self.label(k)} bounce = {e:.2f}"

    def t_set_gravity(self, g):
        self.w.set_gravity(g)
        return f"gravity = {g:.2f} m/s2"

    def t_set_color(self, obj, color):
        k = self.resolve(obj)
        o = self.obj(k)
        rgb0 = o["rgb0"]
        lum = rgb0 @ np.array([0.299, 0.587, 0.114])
        shade = lum / max(np.median(lum), 1e-3)                  # keep highlights and shadows
        o["splat"]["rgb"] = np.clip(np.array(COLORS[color])[None] * shade[:, None], 0, 1).astype(np.float32)
        self.on_appearance(k)
        return f"{self.label(k)} is now {color}"

    def t_scale(self, obj, factor):
        k = self.resolve(obj)
        o = self.obj(k)
        f = factor
        o["splat"]["means"] = o["splat"]["means"] * f
        o["splat"]["scales"] = o["splat"]["scales"] * f
        if o.get("refl") is not None:
            o["refl"]["means"] = o["refl"]["means"] * f
            o["refl"]["scales"] = o["refl"]["scales"] * f
        o["hull"] = o["hull"] * f
        o["scale"] *= f
        self.w.rescale(k, o["hull"], o["mass"] * f ** 3)
        self.on_appearance(k)
        return f"{self.label(k)} scaled x{f:.2g} (now x{o['scale']:.2g}, {o['mass']:.2f} kg)"

    def t_remove(self, obj):
        k = self.resolve(obj)
        self.w.set_removed(k, True)
        self.on_appearance(k)
        return f"removed {self.label(k)}"

    def t_restore(self, obj):
        k = self.resolve(obj)
        o = self.obj(k)
        if o["scale"] != 1.0:
            self.t_scale(str(k), 1.0 / o["scale"])
        o["splat"]["rgb"] = o["rgb0"].copy()
        self.w.set_removed(k, False)
        self.on_appearance(k)
        return f"restored {self.label(k)}"

    def t_reset(self):
        """Everything back to the capture: poses, colours, sizes, removed objects, properties, gravity."""
        for o in self.sc.objects:
            k = o["id"]
            changed = o["scale"] != 1.0 or not np.array_equal(o["splat"]["rgb"], o["rgb0"])
            if o["scale"] != 1.0:
                f = 1.0 / o["scale"]
                for part in (o["splat"], o.get("refl")):
                    if part is not None:
                        part["means"] = part["means"] * f
                        part["scales"] = part["scales"] * f
                o["hull"] = o["hull"] * f
                o["scale"] = 1.0
            o["splat"]["rgb"] = o["rgb0"].copy()
            o.update(o["defaults"])
            o["removed"] = False
            if changed:
                self.on_appearance(k)
        self.w.gravity = 9.81
        self.w._build(keep=False)                          # fresh model: rest poses, zero velocity
        return "reset everything: positions, colours, sizes, removed objects, properties, gravity"

    def t_simulate(self, actions, seconds=2.0):
        """Run actions on a saved state, report, then restore. Deterministic."""
        saved = (self.w.d.qpos.copy(), self.w.d.qvel.copy())
        ids = [o["id"] for o in self.sc.objects if not o.get("removed")]
        p0 = {k: self.w.pose(k)[0] for k in ids}
        log = [self.run(c) for c in actions]
        self.w.step(seconds)
        rows = []
        for k in ids:
            moved = np.linalg.norm(self.w.pose(k)[0] - p0[k]) * 100
            tilt = self._tilt(k)
            state = ("fell off the table" if not self._on_table(k) else
                     "tipped over" if tilt > 45 else "stayed upright")
            if moved > 0.5 or tilt > 10:
                rows.append(f"{self.label(k)}: moved {moved:.0f} cm, {state}")
        self.w.d.qpos[:], self.w.d.qvel[:] = saved
        self.w.forward()
        return "prediction (" + "; ".join(log) + f") after {seconds:.0f}s: " + ("; ".join(rows) or "nothing moves")
