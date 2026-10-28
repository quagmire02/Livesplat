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

        pp = run / "props.json"                         # Spark estimates (agent/properties.py)
        if pp.exists():
            props = json.loads(pp.read_text())
            for o in self.objects:
                p = props.get(str(o["id"]), {})
                o["mass"] = float(p.get("mass_kg", o["mass"]))
                o["friction"] = float(p.get("friction", o["friction"]))
                o["bounce"] = float(p.get("bounce", o["bounce"]))

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
    from agent.schema import COLORS
    from physics.pick import pick

    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("--src", default="thrift_sfm")
    ap.add_argument("--port", type=int, default=8080)
    ap.add_argument("--sim", help="sim package for Simulation mode (default: <run>/sim_m or <run>/sim)")
    a = ap.parse_args()

    sc = Scene(a.run_dir, a.src)
    world = World(sc.objects, sc.polygon, floor_z=sc.floor_z)
    lock = threading.RLock()
    r = sc.radius
    server = viser.ViserServer(port=a.port)
    server.scene.set_up_direction("+z")
    server.gui.configure_theme(control_layout="floating", control_width="medium", show_logo=False)
    bg_node = add_splat(server, "/bg", sc.bg)
    nodes, refl_nodes, fill_nodes = {}, {}, {}
    cam = {"pos": np.array([1.3 * r, -1.3 * r, 1.1 * r]), "look": np.array([0.0, 0.0, 0.1 * r])}
    ui = {"sel": None, "held": False, "target": None, "trail": [], "paused": False, "sim": False}

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

    @server.on_client_connect
    def _(client):
        client.camera.position = tuple(cam["pos"])
        client.camera.look_at = tuple(cam["look"])

        @client.camera.on_update
        def _(c):
            cam["pos"], cam["look"] = np.array(c.position), np.array(c.look_at)

    # ---- panel (compact) -----------------------------------------------------
    allow_meta = os.environ.get("LIVESPLAT_ALLOW_META") == "1"
    server.gui.add_markdown("**Click** an object to select it. **Drag** the arrows to move it; "
                            "let go while moving to throw it.")
    box = server.gui.add_text("command", "push the bottle left, then make it red")
    planner = server.gui.add_dropdown("planner", ["local", "meta"]) if allow_meta else None
    go = server.gui.add_button("run command")
    log = server.gui.add_markdown("")
    sel_md = server.gui.add_markdown("_nothing selected_")
    obj_btn = server.gui.add_button_group("selected", ["push", "drop", "throw", "remove"])
    color = server.gui.add_dropdown("colour", ["(original)"] + list(COLORS))
    sim_dir = a.sim or next((str(Path(a.run_dir) / n) for n in ("sim_m", "sim")
                             if (Path(a.run_dir) / n / "robot_scene.xml").exists()), None)
    scene_btn = server.gui.add_button_group("scene", ["reset", "pause", "moon g", "earth g"]
                                            + (["simulation"] if sim_dir else []))
    sim_mode = {"obj": None}
    history = []

    def run_command(text):
        with lock:
            results, note = handle(text, ctrl, planner.value if planner else "local", default=ui["sel"])
        lines = [f"**> {text}**"] + [f"- `{c['name']}` → {res}" for c, res in results]
        if note:
            lines.append(f"- _{note}_")
        history[:0] = ["\n".join(lines)]
        log.content = "\n\n".join(history[:3])
        print("\n".join(lines), flush=True)
        show_selection()

    go.on_click(lambda _: run_command(box.value))

    # ---- selection + gizmo -----------------------------------------------------
    gizmo = server.scene.add_transform_controls("/gizmo", scale=max(0.45 * r, 0.1))
    gizmo.visible = False                       # (disable_sliders/depth_test options gave NaN lines)

    def mid_offset(k, q):
        """Gizmo sits at the object's mid-height, not buried at its base."""
        from scipy.spatial.transform import Rotation
        h = ctrl.obj(k)["hull"][:, 2].max()
        return Rotation.from_quat([q[1], q[2], q[3], q[0]]).as_matrix() @ np.array([0.0, 0.0, 0.5 * h])

    def show_selection():
        k = ui["sel"]
        if k is None or not world.present(k):
            ui["sel"] = None
            gizmo.visible = False
            sel_md.content = "_nothing selected_"
            return
        o = ctrl.obj(k)
        sel_md.content = (f"**{ctrl.label(k)}** · {o['mass']:.2f} kg · friction {o['friction']:.2f} "
                          f"· size x{o['scale']:.2g}")
        gizmo.visible = True

    @server.scene.on_click()
    def _(ev):
        with lock:
            k = pick(ev.ray_origin, ev.ray_direction, sc.objects, world)
        ui["sel"] = k
        if k is not None:
            p, q = world.pose(k)
            gizmo.position, gizmo.wxyz = tuple(p + mid_offset(k, q)), tuple(q)
        show_selection()

    @gizmo.on_drag_start
    def _(_):
        if ui["sel"] is not None:
            ui["held"], ui["trail"] = True, []

    @gizmo.on_update
    def _(_):
        if ui["held"]:
            q = np.array(gizmo.wxyz)
            ui["target"] = (np.array(gizmo.position) - mid_offset(ui["sel"], q), q)
            ui["trail"] = (ui["trail"] + [(time.time(), np.array(gizmo.position))])[-6:]

    @gizmo.on_drag_end
    def _(_):
        ui["held"] = False
        k, tr = ui["sel"], ui["trail"]
        if k is None or len(tr) < 2:
            return
        (t0, p0), (t1, p1) = tr[0], tr[-1]
        if time.time() - t1 < 0.15 and t1 > t0:              # still moving at release: throw
            v = 0.6 * (p1 - p0) / (t1 - t0)
            s = np.linalg.norm(v)
            if s > 0.5:                                    # a slow release just puts it down
                v = v * min(1.0, 2.5 / s)
                with lock:
                    world.throw(k, tuple(v))

    @obj_btn.on_click
    def _(_):
        if ui["sel"] is None:
            log.content = "_select an object first (click it)_"
            return
        verb = {"push": "push", "drop": "drop", "throw": "throw", "remove": "remove"}[obj_btn.value]
        run_command(f"{verb} object {ui['sel']}" + (" away" if verb == "push" else ""))

    @color.on_update
    def _(_):
        if ui["sel"] is None:
            return
        k = ui["sel"]
        if color.value == "(original)":
            o = ctrl.obj(k)
            o["splat"]["rgb"] = o["rgb0"].copy()
            build_nodes(k)
        else:
            run_command(f"make object {k} {color.value}")

    @scene_btn.on_click
    def _(_):
        v = scene_btn.value
        if v == "simulation":                    # optional mode: robot + sim package, core scene hidden
            if sim_mode["obj"] is None:
                from sim.view import SimMode
                log.content = "_loading simulation mode..._"
                sim_mode["obj"] = SimMode(server, sim_dir)
            ui["sim"] = not ui["sim"]
            sim_mode["obj"].set_visible(ui["sim"])
            for n in [bg_node, *nodes.values(), *refl_nodes.values(), *fill_nodes.values()]:
                n.visible = not ui["sim"]
            gizmo.visible = False
            log.content = "_simulation mode (press simulation again to go back)_" if ui["sim"] else "_tabletop mode_"
            return
        if v == "reset":
            run_command("reset")
        elif v == "pause":
            ui["paused"] = not ui["paused"]
            log.content = "_paused_" if ui["paused"] else "_running_"
        else:
            run_command("moon gravity" if v == "moon g" else "earth gravity")

    # ---- loop ----------------------------------------------------------------
    frame = 1 / 60
    while True:
        t0 = time.time()
        if not ui["paused"] and not ui["sim"]:
            with lock:
                k = ui["sel"]
                if ui["held"] and k is not None and world.present(k) and ui["target"] is not None:
                    world.place(k, *ui["target"])                 # held: follow the gizmo
                world.step(frame)
                state = {}
                for o in sc.objects:
                    kk = o["id"]
                    present = world.present(kk)
                    pose = world.pose(kk) if present else None
                    upright = present and abs(pose[0][2] - world.rest[kk][2]) < 0.01 and abs(pose[1][0]) > 0.996
                    state[kk] = (present, pose, world.at_rest_pose(kk), upright)
            with server.atomic():
                for kk, (present, pose, rest, upright) in state.items():
                    nodes[kk].visible = present
                    if present:
                        nodes[kk].position, nodes[kk].wxyz = tuple(pose[0]), tuple(pose[1])
                    if kk in refl_nodes:
                        refl_nodes[kk].visible = upright
                        if upright:
                            w_, z_ = pose[1][0], pose[1][3]           # yaw-only part of the rotation
                            n_ = max(np.hypot(w_, z_), 1e-9)
                            refl_nodes[kk].position = (pose[0][0], pose[0][1], 0.0)
                            refl_nodes[kk].wxyz = (w_ / n_, 0.0, 0.0, z_ / n_)
                    if kk in fill_nodes:
                        fill_nodes[kk].visible = not rest
                k = ui["sel"]
                if k is not None and not ui["held"] and state.get(k, (False,))[0]:
                    gp, gq = state[k][1]
                    gizmo.position, gizmo.wxyz = tuple(gp + mid_offset(k, gq)), tuple(gq)
        time.sleep(max(0.0, frame - (time.time() - t0)))


if __name__ == "__main__":
    main()
