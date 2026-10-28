"""LiveSplat viewer: background + one Viser splat node per object, driven by MuJoCo. Table frame, Z up.

Run (WSL): python -m viewer.app <run_dir> [--src thrift_sfm] [--port 8080]
Object node origin = object base centre (xy median, z = 0 on the table) = MuJoCo body frame.
Label -k = object k's reflection in a glossy table: shown only while k is at rest.
"""
import argparse
import json
import threading
import time
from pathlib import Path

import numpy as np
import viser
from scipy.spatial import ConvexHull

from physics.world import World, hull_vertices
from scene_io import splat

DEFAULT_DENSITY = 300.0          # kg/m^3: light household items; Spark properties replace this
DEFAULT_FRICTION = 0.5


class Scene:
    def __init__(self, run_dir, src="thrift_sfm", max_bg=1_200_000, crop=1.6):
        run = Path(run_dir)
        al = json.loads((run / "align.json").read_text())
        self.T = np.array(al["T_world_to_table"])
        self.radius = r = al["radius"]
        self.polygon = np.array(al["polygon"])
        # A refined splat carries its own labels (densification changed the Gaussian count).
        lab_path = next(p for p in (run / src / "labels.npy", run / "labels.npy", run / "labels_geo.npy")
                        if p.exists())
        labels = np.load(lab_path)
        raw = splat.transform(splat.load_raw(run / src / "splat.ply"), self.T)
        m = raw["means"]
        radial = np.linalg.norm(m[:, :2], axis=1)
        near = (radial < crop * r) & (m[:, 2] > -2.5 * r) & (m[:, 2] < 2.0 * r)
        bg_idx = np.flatnonzero((labels == 0) & near)
        if len(bg_idx) > max_bg:                         # keep the most opaque background splats
            bg_idx = bg_idx[np.argsort(-raw["opacity"][bg_idx])[:max_bg]]
        self.bg = splat.activate(splat.subset(raw, bg_idx))
        self.floor_z = self._floor(m, raw["opacity"], labels, radial)

        self.objects = []
        for k in range(1, labels.max() + 1):
            a = splat.activate(splat.subset(raw, labels == k))
            base = np.array([*np.median(a["means"][:, :2], 0), 0.0])
            a["means"] = (a["means"] - base).astype(np.float32)
            solid = a["means"][(a["opacity"] > 0.3) & (a["means"][:, 2] >= 0)]
            hull = hull_vertices(solid)
            vol = ConvexHull(solid).volume
            refl = None
            if (labels == -k).any():
                refl = splat.activate(splat.subset(raw, labels == -k))
                refl["means"] = (refl["means"] - base).astype(np.float32)
            self.objects.append({"id": k, "base": base, "splat": a, "refl": refl, "hull": hull,
                                 "volume": vol, "mass": max(0.01, DEFAULT_DENSITY * vol),
                                 "friction": DEFAULT_FRICTION, "bounce": 0.1})

    def _floor(self, m, opacity, labels, radial):
        """Floor height = densest level well below the table top."""
        r = self.radius
        z = m[(labels == 0) & (opacity > 0.5) & (radial < 1.5 * r) & (m[:, 2] < -0.6 * r), 2]
        if len(z) < 100:
            return -1.6 * r
        h, e = np.histogram(z, bins=60)
        return float(e[np.argmax(h)] + (e[1] - e[0]) / 2)


def add_splat(server, name, a):
    return server.scene.add_gaussian_splats(name, centers=a["means"], covariances=splat.covariances(a),
                                            rgbs=a["rgb"], opacities=a["opacity"][:, None])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("--src", default="thrift_sfm")
    ap.add_argument("--port", type=int, default=8080)
    a = ap.parse_args()

    sc = Scene(a.run_dir, a.src)
    world = World(sc.objects, sc.polygon, floor_z=sc.floor_z)
    lock = threading.Lock()
    r = sc.radius

    server = viser.ViserServer(port=a.port)
    server.scene.set_up_direction("+z")
    add_splat(server, "/bg", sc.bg)
    nodes, refl_nodes = {}, {}
    for o in sc.objects:
        nodes[o["id"]] = add_splat(server, f"/obj{o['id']}", o["splat"])
        nodes[o["id"]].position = tuple(o["base"])
        if o["refl"] is not None:
            refl_nodes[o["id"]] = add_splat(server, f"/refl{o['id']}", o["refl"])
            refl_nodes[o["id"]].position = tuple(o["base"])
    print(f"{len(sc.objects)} objects, table radius {r:.3f}, floor z {sc.floor_z:.3f}")
    for o in sc.objects:
        print(f"  obj {o['id']}: volume {o['volume'] * 1e6:.0f} cm3, mass {o['mass']:.3f} kg, "
              f"hull {len(o['hull'])} verts")

    @server.on_client_connect
    def _(client):
        client.camera.position = (1.3 * r, -1.3 * r, 1.1 * r)
        client.camera.look_at = (0.0, 0.0, 0.1 * r)

    # ---- GUI -------------------------------------------------------------
    ids = [str(o["id"]) for o in sc.objects]
    pick = server.gui.add_dropdown("object", ids)
    with server.gui.add_folder("actions"):
        b_push = {d: server.gui.add_button(f"push {d}") for d in ("+x", "-x", "+y", "-y")}
        b_drop = server.gui.add_button("drop from 0.3 m")
        b_throw = server.gui.add_button("throw up")
        b_reset = server.gui.add_button("reset scene")
    with server.gui.add_folder("properties"):
        s_mass = server.gui.add_slider("mass kg", 0.01, 5.0, 0.01, sc.objects[0]["mass"])
        s_fric = server.gui.add_slider("friction", 0.0, 1.5, 0.01, DEFAULT_FRICTION)
        s_bnc = server.gui.add_slider("bounce", 0.0, 0.95, 0.05, 0.1)
        s_grav = server.gui.add_slider("gravity m/s2", 0.0, 25.0, 0.1, 9.81)
    with server.gui.add_folder("speed"):
        s_push = server.gui.add_slider("push m/s", 0.1, 3.0, 0.1, 0.6)
        paused = server.gui.add_checkbox("pause", False)

    def k():
        return int(pick.value)

    def obj(kk):
        return next(o for o in sc.objects if o["id"] == kk)

    @pick.on_update
    def _(_):
        o = obj(k())
        s_mass.value, s_fric.value, s_bnc.value = o["mass"], o["friction"], o["bounce"]

    for dname, btn in b_push.items():
        vec = {"+x": (1, 0), "-x": (-1, 0), "+y": (0, 1), "-y": (0, -1)}[dname]
        btn.on_click(lambda _, v=vec: (lock.acquire(), world.push(k(), v, s_push.value), lock.release()))

    @b_drop.on_click
    def _(_):
        with lock:
            world.drop(k(), 0.3)

    @b_throw.on_click
    def _(_):
        with lock:
            world.throw(k(), (0.3, 0.0, 2.5), spin=(3.0, 1.0, 0.0))

    @b_reset.on_click
    def _(_):
        with lock:
            world.reset()

    def prop(name, slider):
        @slider.on_update
        def _(_):
            if getattr(obj(k()), "get")(name) == slider.value:
                return
            with lock:
                getattr(world, f"set_{name}")(k(), slider.value)
    prop("mass", s_mass)
    prop("friction", s_fric)
    prop("bounce", s_bnc)

    @s_grav.on_update
    def _(_):
        with lock:
            world.set_gravity(s_grav.value)

    # ---- loop ------------------------------------------------------------
    frame = 1 / 60
    while True:
        t0 = time.time()
        if not paused.value:
            with lock:
                world.step(frame)
                poses = {o["id"]: world.pose(o["id"]) for o in sc.objects}
                rest = {o["id"]: world.at_rest_pose(o["id"]) for o in sc.objects}
            with server.atomic():
                for kk, (p, q) in poses.items():
                    nodes[kk].position = tuple(p)
                    nodes[kk].wxyz = tuple(q)
                    if kk in refl_nodes:
                        refl_nodes[kk].visible = rest[kk]
        time.sleep(max(0.0, frame - (time.time() - t0)))


if __name__ == "__main__":
    main()
