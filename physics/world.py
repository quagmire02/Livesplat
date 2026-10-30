"""MuJoCo world built from segmented objects. Table frame: Z up, table top at z = 0.

Each object = free body with a convex-hull mesh of its Gaussian centres (local frame:
origin = object base centre, so body pose == viewer node pose). Table = static prism
from the table polygon. Floor = plane at floor_z.
"""
import mujoco
import numpy as np
from scipy.spatial import ConvexHull

DT = 0.002                       # physics timestep (s)


def hull_vertices(pts, max_v=200):
    """Convex hull vertices (MuJoCo limits mesh complexity; it hulls them again itself)."""
    v = pts[ConvexHull(pts).vertices]
    if len(v) > max_v:
        v = v[np.random.default_rng(0).choice(len(v), max_v, replace=False)]
    return v


def _vec(a):
    return " ".join(f"{x:.6g}" for x in np.ravel(a))


def build_xml(objects, polygon, table_thick, floor_z, gravity=9.81):
    """objects: list of dicts {id, base (3,), hull (V,3) local, mass, friction, bounce}."""
    prism = np.concatenate([np.c_[polygon, np.zeros(len(polygon))],
                            np.c_[polygon, np.full(len(polygon), -table_thick)]])
    assets = [f'<mesh name="table" vertex="{_vec(prism)}"/>']
    bodies = []
    for o in objects:
        if o.get("removed"):
            continue
        assets.append(f'<mesh name="m{o["id"]}" vertex="{_vec(o["hull"])}"/>')
        # solref: (timeconst, dampratio). Lower dampratio = bouncier.
        damp = max(0.05, 1.0 - o.get("bounce", 0.1))
        bodies.append(
            f'<body name="o{o["id"]}" pos="{_vec(o["base"])}"><freejoint/>'
            f'<geom type="mesh" mesh="m{o["id"]}" mass="{o["mass"]:.6g}" '
            f'friction="{o["friction"]:.4g} 0.005 0.0001" solref="0.004 {damp:.3g}" condim="4"/></body>')
    return f"""<mujoco>
  <option timestep="{DT}" gravity="0 0 {-gravity}" integrator="implicitfast"/>
  <asset>{''.join(assets)}</asset>
  <worldbody>
    <!-- MuJoCo uses the MAX friction of the two geoms: keep these low so each object's own friction applies -->
    <geom name="floor" type="plane" pos="0 0 {floor_z:.6g}" size="0 0 0.05" friction="0.02 0.005 0.0001"/>
    <geom name="table" type="mesh" mesh="table" friction="0.02 0.005 0.0001"/>
    {''.join(bodies)}
  </worldbody>
</mujoco>"""


def stands_alone(hull, mass=0.3, seconds=1.0, max_tilt_deg=10.0):
    """Does this collision hull stand on a flat plane without tipping? (Objects stood still in the scan.)"""
    xml = (f'<mujoco><option timestep="{DT}"/><asset><mesh name="h" vertex="{_vec(hull)}"/></asset>'
           f'<worldbody><geom type="plane" size="0 0 0.05"/><body pos="0 0 0.001"><freejoint/>'
           f'<geom type="mesh" mesh="h" mass="{mass}"/></body></worldbody></mujoco>')
    m = mujoco.MjModel.from_xml_string(xml)
    d = mujoco.MjData(m)
    for _ in range(int(seconds / DT)):
        mujoco.mj_step(m, d)
    tilt = np.degrees(2 * np.arccos(min(1.0, abs(d.qpos[3]))))
    return tilt <= max_tilt_deg


class World:
    def __init__(self, objects, polygon, table_thick=0.03, floor_z=-0.75):
        self.objects = objects
        self.polygon, self.table_thick, self.floor_z = polygon, table_thick, floor_z
        self.gravity = 9.81
        self.slides = {}                     # id -> [unit xy direction, speed, distance left]: hand-guided pushes
        self._build()

    def _build(self, keep=True):
        """(Re)compile the model. Per-object state is saved and restored by object id, because
        adding or removing a body changes the qpos layout."""
        saved = self._save() if keep and hasattr(self, "d") else {}
        self.xml = build_xml(self.objects, self.polygon, self.table_thick, self.floor_z, self.gravity)
        self.m = mujoco.MjModel.from_xml_string(self.xml)
        self.d = mujoco.MjData(self.m)
        self.bid = {o["id"]: self.m.body(f"o{o['id']}").id for o in self.objects if not o.get("removed")}
        self.rest = {k: self.d.qpos[self._jnt(k)[0]:self._jnt(k)[0] + 7].copy() for k in self.bid}
        for k, (qp, qv) in saved.items():
            if k in self.bid:
                q, dof = self._jnt(k)
                self.d.qpos[q:q + 7], self.d.qvel[dof:dof + 6] = qp, qv
        mujoco.mj_forward(self.m, self.d)

    def _save(self):
        out = {}
        for k in self.bid:
            q, dof = self._jnt(k)
            out[k] = (self.d.qpos[q:q + 7].copy(), self.d.qvel[dof:dof + 6].copy())
        return out

    def _jnt(self, k):
        j = self.m.body_jntadr[self.bid[k]]
        return self.m.jnt_qposadr[j], self.m.jnt_dofadr[j]

    # ---- state -----------------------------------------------------------
    def step(self, seconds):
        for _ in range(max(1, int(round(seconds / DT)))):
            if self.slides:
                self._guide()
            mujoco.mj_step(self.m, self.d)
        self.d.xfrc_applied[:] = 0

    def _guide(self):
        """Hand-guided slide: steady horizontal speed, no tipping rotation, released after the distance."""
        for k in list(self.slides):
            if k not in self.bid:
                del self.slides[k]
                continue
            u, v, left = self.slides[k]
            _, dof = self._jnt(k)
            if left <= 0:
                self.d.qvel[dof:dof + 2] = 0
                self.d.qvel[dof + 3:dof + 6] = 0
                del self.slides[k]
                continue
            self.d.qvel[dof:dof + 2] = u * v
            self.d.qvel[dof + 3:dof + 6] = 0
            self.slides[k][2] = left - v * DT

    def forward(self):
        mujoco.mj_forward(self.m, self.d)

    def present(self, k):
        return k in self.bid

    def pose(self, k):
        b = self.bid[k]
        return self.d.xpos[b].copy(), self.d.xquat[b].copy()           # wxyz

    def speed(self, k):
        _, dof = self._jnt(k)
        return float(np.linalg.norm(self.d.qvel[dof:dof + 3]))

    def at_rest_pose(self, k, tol=0.01):
        if k not in self.bid:
            return False
        q, _ = self._jnt(k)
        return float(np.linalg.norm(self.d.qpos[q:q + 3] - self.rest[k][:3])) < tol

    def reset(self):
        for k in self.bid:
            q, dof = self._jnt(k)
            self.d.qpos[q:q + 7] = self.rest[k]
            self.d.qvel[dof:dof + 6] = 0
        mujoco.mj_forward(self.m, self.d)

    def place(self, k, pos, wxyz):
        q, dof = self._jnt(k)
        self.d.qpos[q:q + 3], self.d.qpos[q + 3:q + 7] = pos, wxyz
        self.d.qvel[dof:dof + 6] = 0
        mujoco.mj_forward(self.m, self.d)

    def set_removed(self, k, removed):
        for o in self.objects:
            if o["id"] == k:
                o["removed"] = removed
        self._build()

    def rescale(self, k, hull, mass):
        """New collision hull + mass. Hulls scale about the base origin (z = 0), so the base stays on the table."""
        for o in self.objects:
            if o["id"] == k:
                o["hull"], o["mass"] = hull, max(1e-3, mass)
        self._build()

    # ---- actions (all units SI, table frame) -----------------------------
    def push(self, k, direction, speed=0.6):
        """Give object k a horizontal velocity (m/s) in `direction` (2- or 3-vector)."""
        _, dof = self._jnt(k)
        v = np.zeros(3)
        v[:len(direction)] = direction
        v /= max(np.linalg.norm(v), 1e-9)
        self.d.qvel[dof:dof + 3] += speed * v

    def slide(self, k, direction, distance, speed=0.3):
        """Guide object k across the table for `distance` at a steady `speed` (like a hand), then let go."""
        u = np.asarray(direction, float)[:2]
        u = u / max(np.linalg.norm(u), 1e-9)
        self.slides[k] = [u, float(speed), float(distance)]

    def drop(self, k, height):
        q, dof = self._jnt(k)
        self.d.qpos[q + 2] = self.rest[k][2] + height
        self.d.qvel[dof:dof + 6] = 0
        mujoco.mj_forward(self.m, self.d)

    def throw(self, k, velocity, spin=(0, 0, 0)):
        _, dof = self._jnt(k)
        self.d.qvel[dof:dof + 3] = velocity
        self.d.qvel[dof + 3:dof + 6] = spin

    def set_mass(self, k, kg):
        self._set_obj(k, mass=kg)

    def set_friction(self, k, mu):
        self._set_obj(k, friction=mu)

    def set_bounce(self, k, e):
        self._set_obj(k, bounce=e)

    def set_gravity(self, g):
        self.gravity = g
        self.m.opt.gravity[:] = (0, 0, -g)

    def _set_obj(self, k, **kw):
        for o in self.objects:
            if o["id"] == k:
                o.update(kw)
        self._build()
