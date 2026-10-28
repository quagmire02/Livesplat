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
    <geom name="floor" type="plane" pos="0 0 {floor_z:.6g}" size="0 0 0.05" friction="0.8 0.005 0.0001"/>
    <geom name="table" type="mesh" mesh="table" friction="0.6 0.005 0.0001"/>
    {''.join(bodies)}
  </worldbody>
</mujoco>"""


class World:
    def __init__(self, objects, polygon, table_thick=0.03, floor_z=-0.75):
        self.objects = objects
        self.polygon, self.table_thick, self.floor_z = polygon, table_thick, floor_z
        self.gravity = 9.81
        self._build()

    def _build(self, keep_state=None):
        self.xml = build_xml(self.objects, self.polygon, self.table_thick, self.floor_z, self.gravity)
        self.m = mujoco.MjModel.from_xml_string(self.xml)
        self.d = mujoco.MjData(self.m)
        self.qpos0 = self.d.qpos.copy()
        if keep_state is not None:
            self.d.qpos[:], self.d.qvel[:] = keep_state
        mujoco.mj_forward(self.m, self.d)
        self.bid = {o["id"]: self.m.body(f"o{o['id']}").id for o in self.objects}

    def _jnt(self, k):
        j = self.m.body_jntadr[self.bid[k]]
        return self.m.jnt_qposadr[j], self.m.jnt_dofadr[j]

    # ---- state -----------------------------------------------------------
    def step(self, seconds):
        for _ in range(max(1, int(round(seconds / DT)))):
            mujoco.mj_step(self.m, self.d)
        self.d.xfrc_applied[:] = 0

    def pose(self, k):
        b = self.bid[k]
        return self.d.xpos[b].copy(), self.d.xquat[b].copy()           # wxyz

    def speed(self, k):
        _, dof = self._jnt(k)
        return float(np.linalg.norm(self.d.qvel[dof:dof + 3]))

    def at_rest_pose(self, k, tol=0.01):
        q, _ = self._jnt(k)
        return float(np.linalg.norm(self.d.qpos[q:q + 3] - self.qpos0[q:q + 3])) < tol

    def reset(self):
        self.d.qpos[:] = self.qpos0
        self.d.qvel[:] = 0
        mujoco.mj_forward(self.m, self.d)

    # ---- actions (all units SI, table frame) -----------------------------
    def push(self, k, direction, speed=0.6):
        """Give object k a horizontal velocity (m/s) in `direction` (2- or 3-vector)."""
        _, dof = self._jnt(k)
        v = np.zeros(3)
        v[:len(direction)] = direction
        v /= max(np.linalg.norm(v), 1e-9)
        self.d.qvel[dof:dof + 3] += speed * v

    def drop(self, k, height):
        q, dof = self._jnt(k)
        self.d.qpos[q + 2] = self.qpos0[q + 2] + height
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
        self._build(keep_state=(self.d.qpos.copy(), self.d.qvel.copy()))
