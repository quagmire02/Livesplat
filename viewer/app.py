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
from viewer.holefill import fill_splat

DEFAULT_DENSITY = 300.0          # kg/m^3: light household items; Spark properties replace this
DEFAULT_FRICTION = 0.5


def solid_points(a, min_opacity=0.5, k=12, keep=2.0):
    """Points for the collision hull: opaque, above the table, statistical outliers removed
    (floaters attached by region growing would otherwise inflate the hull)."""
    from scipy.spatial import cKDTree
    P = a["means"][(a["opacity"] > min_opacity) & (a["means"][:, 2] >= 0)]
    if len(P) < 50:
        P = a["means"][a["means"][:, 2] >= 0]
    d = cKDTree(P).query(P, k=k + 1)[0][:, 1:].mean(1)
    P = P[d < np.median(d) * keep]
    lo, hi = np.percentile(P[:, :2], 1, 0), np.percentile(P[:, :2], 99, 0)
    P = P[((P[:, :2] >= lo) & (P[:, :2] <= hi)).all(1)]
    # Flat footprint: the underside is never seen, so project the lowest band onto the table.
    # Use ALL low Gaussians (any opacity): thin or shiny objects have few opaque ones, and a too
    # narrow footprint makes an object that stood still in the video tip over.
    A = a["means"][(a["means"][:, 2] >= 0) & (a["opacity"] > 0.05)]
    band = A[A[:, 2] < 0.1 * np.ptp(P[:, 2])]
    if len(band) >= 20:
        lo, hi = np.percentile(band[:, :2], 5, 0), np.percentile(band[:, :2], 95, 0)
        band = band[((band[:, :2] >= lo) & (band[:, :2] <= hi)).all(1)]
    return np.concatenate([P, np.c_[band[:, :2], np.zeros(len(band))]])


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
            solid = solid_points(a)
            hull = hull_vertices(solid)
            vol = ConvexHull(solid).volume
            refl = None
            if (labels == -k).any():
                refl = splat.activate(splat.subset(raw, labels == -k))
                refl["means"] = (refl["means"] - base).astype(np.float32)
            self.objects.append({"id": k, "base": base, "splat": a, "refl": refl, "hull": hull,
                                 "volume": vol, "mass": max(0.01, DEFAULT_DENSITY * vol),
                                 "friction": DEFAULT_FRICTION, "bounce": 0.1,
                                 "fill": fill_splat(self.bg, base, hull, r)})

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
    import os

    from agent.controller import Controller
    from agent.planner import handle

    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("--src", default="thrift_sfm")
    ap.add_argument("--port", type=int, default=8080)
    a = ap.parse_args()

    sc = Scene(a.run_dir, a.src)
    world = World(sc.objects, sc.polygon, floor_z=sc.floor_z)
    lock = threading.RLock()
    r = sc.radius
    server = viser.ViserServer(port=a.port)
    server.scene.set_up_direction("+z")
    add_splat(server, "/bg", sc.bg)
    nodes, refl_nodes, fill_nodes = {}, {}, {}
    cam = {"pos": np.array([1.3 * r, -1.3 * r, 1.1 * r]), "look": np.array([0.0, 0.0, 0.1 * r])}

    def build_nodes(k):
        o = next(o for o in sc.objects if o["id"] == k)
        nodes[k] = add_splat(server, f"/obj{k}", o["splat"])
        if o["refl"] is not None:
            refl_nodes[k] = add_splat(server, f"/refl{k}", o["refl"])
            refl_nodes[k].position = tuple(o["base"])
        if o["fill"] is not None:
            fill_nodes[k] = add_splat(server, f"/fill{k}", o["fill"])
            fill_nodes[k].position = tuple(o["base"])
            fill_nodes[k].visible = False

    for o in sc.objects:
        build_nodes(o["id"])
    ctrl = Controller(sc, world, a.run_dir, on_appearance=build_nodes, camera=lambda: (cam["pos"], cam["look"]))
    print(f"{len(sc.objects)} objects, table radius {r:.3f}, floor z {sc.floor_z:.3f}")
    for o in sc.objects:
        print(f"  {o['id']} {ctrl.names[o['id']][:3]}: volume {o['volume'] * 1e6:.0f} cm3, mass {o['mass']:.3f} kg, "
              f"fill {'yes' if o['fill'] is not None else 'no'}")

    @server.on_client_connect
    def _(client):
        client.camera.position = tuple(cam["pos"])
        client.camera.look_at = tuple(cam["look"])

        @client.camera.on_update
        def _(c):
            cam["pos"], cam["look"] = np.array(c.position), np.array(c.look_at)

    # ---- command box -------------------------------------------------------
    allow_meta = os.environ.get("LIVESPLAT_ALLOW_META") == "1"
    with server.gui.add_folder("command"):
        backend = server.gui.add_dropdown("planner", ["local", "meta"] if allow_meta else ["local"])
        box = server.gui.add_text("say", "push the tallest object left")
        go = server.gui.add_button("run")
        out = server.gui.add_markdown("")
    history = []

    def run_command(text):
        with lock:
            results, note = handle(text, ctrl, backend.value, default=int(pick.value))
        lines = [f"**> {text}**"] + [f"- `{c['name']}` → {res}" for c, res in results]
        if note:
            lines.append(f"- _{note}_")
        history[:0] = ["\n".join(lines)]
        out.content = "\n\n".join(history[:6])
        print("\n".join(lines), flush=True)

    @go.on_click
    def _(_):
        run_command(box.value)

    # ---- direct controls -----------------------------------------------------
    ids = [str(o["id"]) for o in sc.objects]
    pick = server.gui.add_dropdown("object", ids)
    with server.gui.add_folder("actions", expand_by_default=False):
        for d in ("left", "right", "away", "toward"):
            server.gui.add_button(f"push {d}").on_click(
                lambda _, d=d: run_command(f"push object {pick.value} {d}"))
        server.gui.add_button("drop").on_click(lambda _: run_command(f"drop object {pick.value}"))
        server.gui.add_button("throw").on_click(lambda _: run_command(f"throw object {pick.value}"))
        server.gui.add_button("reset").on_click(lambda _: run_command("reset"))
    with server.gui.add_folder("world", expand_by_default=False):
        s_grav = server.gui.add_slider("gravity m/s2", 0.0, 25.0, 0.1, 9.81)
        paused = server.gui.add_checkbox("pause", False)

    @s_grav.on_update
    def _(_):
        with lock:
            world.set_gravity(s_grav.value)

    # ---- loop ----------------------------------------------------------------
    frame = 1 / 60
    while True:
        t0 = time.time()
        if not paused.value:
            with lock:
                world.step(frame)
                state = {}
                for o in sc.objects:
                    k = o["id"]
                    present = world.present(k)
                    pose = world.pose(k) if present else None
                    # Upright on the table: the reflection (part of a glossy object's look) follows it.
                    upright = present and abs(pose[0][2] - world.rest[k][2]) < 0.01 and abs(pose[1][0]) > 0.996
                    state[k] = (present, pose, world.at_rest_pose(k), upright)
            with server.atomic():
                for k, (present, pose, rest, upright) in state.items():
                    nodes[k].visible = present
                    if present:
                        nodes[k].position, nodes[k].wxyz = tuple(pose[0]), tuple(pose[1])
                    if k in refl_nodes:
                        refl_nodes[k].visible = upright
                        if upright:
                            w_, z_ = pose[1][0], pose[1][3]           # yaw-only part of the rotation
                            n_ = max(np.hypot(w_, z_), 1e-9)
                            refl_nodes[k].position = (pose[0][0], pose[0][1], 0.0)
                            refl_nodes[k].wxyz = (w_ / n_, 0.0, 0.0, z_ / n_)
                    if k in fill_nodes:
                        fill_nodes[k].visible = not rest
        time.sleep(max(0.0, frame - (time.time() - t0)))


if __name__ == "__main__":
    main()
