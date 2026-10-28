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

from physics.world import World, hull_vertices, stands_alone
from scene_io import splat
from segment.cluster import in_polygon
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
        rm = run / "bg_removed.npy"                         # user erasures (viewer clean-up tools)
        if rm.exists():
            bg_idx = bg_idx[~np.isin(bg_idx, np.load(rm))]
        self.bg_idx = bg_idx
        self.bg = splat.activate(splat.subset(raw, bg_idx))
        self.floor_z = self._floor(m, raw["opacity"], labels, radial)

        self.objects = []
        for k in range(1, labels.max() + 1):
            a = splat.activate(splat.subset(raw, labels == k))
            base = np.array([*np.median(a["means"][:, :2], 0), 0.0])
            a["means"] = (a["means"] - base).astype(np.float32)
            solid = solid_points(a)
            gen = run / "gen" / f"obj_{k}.obj"                # TripoSR mesh: collision shape only (visual = splat)
            if gen.exists():
                import trimesh
                V = np.asarray(trimesh.load(gen, process=False).vertices, np.float32)
                band = V[V[:, 2] < V[:, 2].min() + 0.1 * np.ptp(V[:, 2])]          # flat base on the table
                V = np.concatenate([V, np.c_[band[:, :2], np.zeros(len(band))]])
                if stands_alone(hull_vertices(V)):           # unstable meshes fall back to the scan hull
                    solid = V
                else:
                    print(f"  obj {k}: generated collision mesh tips over -> scan hull")
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
            fr = run / "gen" / f"fill_rgb_{k}.npy"              # Muse Image table texture (holefill_ai)
            f_ = self.objects[-1]["fill"]
            if f_ is not None and fr.exists() and len(np.load(fr)) == len(f_["means"]):
                f_["rgb"] = np.load(fr)

        pp = run / "props.json"                         # Spark estimates (agent/properties.py)
        if pp.exists():
            props = json.loads(pp.read_text())
            for o in self.objects:
                p = props.get(str(o["id"]), {})
                o["mass"] = float(p.get("mass_kg", o["mass"]))
                o["friction"] = float(p.get("friction", o["friction"]))
                o["bounce"] = float(p.get("bounce", o["bounce"]))

    def make_object(self, k, a):
        """New object from an activated splat subset in the table frame (click-to-add)."""
        base = np.array([*np.median(a["means"][:, :2], 0), 0.0])
        a = dict(a)
        a["means"] = (a["means"] - base).astype(np.float32)
        solid = solid_points(a)
        hull = hull_vertices(solid)
        vol = ConvexHull(solid).volume
        return {"id": k, "base": base, "splat": a, "refl": None, "hull": hull, "volume": vol,
                "mass": max(0.01, DEFAULT_DENSITY * vol), "friction": DEFAULT_FRICTION, "bounce": 0.1,
                "fill": fill_splat(self.bg, base, hull, self.radius)}

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
    bg_holder = {"node": add_splat(server, "/bg", sc.bg)}
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
    sim_dir = a.sim or next((str(Path(a.run_dir) / n) for n in ("sim_m", "sim")
                             if (Path(a.run_dir) / n / "robot_scene.xml").exists()), None)
    mode_btn = server.gui.add_button("switch to simulation mode", visible=bool(sim_dir))
    intro = server.gui.add_markdown("**Click** an object to select it. **Drag** the arrows to move it; "
                                    "let go while moving to throw it.")
    box = server.gui.add_text("command", "push the bottle left, then make it red")
    planner = server.gui.add_dropdown("planner", ["local", "meta"]) if allow_meta else None
    go = server.gui.add_button("run command")
    log = server.gui.add_markdown("")
    sel_md = server.gui.add_markdown("_nothing selected_")
    obj_btn = server.gui.add_button_group("selected", ["push", "drop", "throw", "remove"])
    color = server.gui.add_dropdown("colour", ["(original)"] + list(COLORS))
    scene_btn = server.gui.add_button_group("scene", ["reset", "pause", "moon g", "earth g"])
    sim_mode = {"obj": None}
    history = []

    from sim.view import SimMode

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
        if ui["sim"]:                                     # simulation mode: tabletop objects are hidden
            return
        if ui.get("adding"):
            ui["adding"] = False
            add_btn.label = "add missed object (then click it)"
            add_clicked(ev.ray_origin, ev.ray_direction)
            return
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

    # ---- background clean-up (image-editor style) ------------------------------------
    keep = np.ones(len(sc.bg["means"]), bool)
    undo_stack = []
    rm_path = Path(a.run_dir) / "bg_removed.npy"
    prev_removed = np.load(rm_path) if rm_path.exists() else np.zeros(0, np.int64)   # earlier sessions
    on_table = (np.abs(sc.bg["means"][:, 2]) < 0.03 * r) & in_polygon(sc.bg["means"][:, :2], sc.polygon * 1.03)
    add_btn = server.gui.add_button("add missed object (then click it)")

    @add_btn.on_click
    def _(_):
        ui["adding"] = True
        add_btn.label = "click the missed object in the scene..."

    def add_clicked(ray_o, ray_d):
        from viewer.add_object import ray_hit, segment_click
        from scene_io.thrift import load_cameras
        p = ray_hit(sc, keep, ray_o, ray_d)
        if p is None:
            log.content = "_nothing there to add_"
            return
        log.content = "_segmenting the clicked object with SAM 3.1..._"
        sel, n = segment_click(sc, load_cameras(Path(a.run_dir) / a.src), keep, p)
        if sel is None or sel.sum() < 200:
            log.content = f"_could not separate an object there ({0 if sel is None else int(sel.sum())} splats)_"
            return
        k = max(o["id"] for o in sc.objects) + 1
        with lock:
            o = sc.make_object(k, {key: v[sel] for key, v in sc.bg.items()})
            o.update(rgb0=o["splat"]["rgb"].copy(), scale=1.0,
                     defaults={"mass": o["mass"], "friction": o["friction"], "bounce": o["bounce"]})
            sc.objects.append(o)                               # World shares this list
            world._build()
            ctrl.names[k] = [f"object {k}", str(k), "new object"]
            moved_out[sel] = True
            keep[sel] = False
            refresh_bg()
            build_nodes(k)
            lab_path = Path(a.run_dir) / "labels.npy"          # persist: these splats are object k now
            labels = np.load(lab_path) if lab_path.exists() else np.load(Path(a.run_dir) / "labels_geo.npy")
            labels[sc.bg_idx[sel]] = k
            np.save(lab_path, labels)
        ui["sel"] = k
        log.content = f"**added object {k}** ({int(sel.sum()):,} splats, {n} views) · try: push object {k}"
        show_selection()

    clean = server.gui.add_folder("clean up background", expand_by_default=False)
    with clean:
        server.gui.add_markdown("**Shift + drag** a box to erase background splats.")
        protect = server.gui.add_checkbox("protect table top", True)
        radius = server.gui.add_slider("keep within (x table)", 1.0, 4.0, 0.1, 2.5)
        min_op = server.gui.add_slider("min opacity", 0.0, 0.5, 0.01, 0.1)
        auto = server.gui.add_button("auto clean")
        undo_btn = server.gui.add_button("undo erase")
        clean_info = server.gui.add_markdown("")

    def apply_bg(sel):
        if not sel.any():
            return
        undo_stack.append(sel.copy())
        keep[sel] = False
        refresh_bg()

    moved_out = np.zeros(len(keep), bool)                      # background splats that became objects

    def refresh_bg():
        bg_holder["node"] = add_splat(server, "/bg", {k: v[keep] for k, v in sc.bg.items()})
        np.save(rm_path, np.concatenate([prev_removed, sc.bg_idx[~keep & ~moved_out]]).astype(np.int64))
        clean_info.content = f"{(~keep).sum():,} splats erased this session · {keep.sum():,} kept"

    @server.scene.on_rect_select(modifier="shift")
    def _(ev):
        if ui["sim"]:
            return
        from scipy.spatial.transform import Rotation
        cam = ev.client.camera
        q = cam.wxyz
        R = Rotation.from_quat([q[1], q[2], q[3], q[0]]).as_matrix()      # camera-to-world, OpenCV axes
        pc = (sc.bg["means"] - np.array(cam.position)) @ R
        z = pc[:, 2]
        tv = np.tan(cam.fov / 2)
        with np.errstate(divide="ignore", invalid="ignore"):
            u = 0.5 + pc[:, 0] / (z * 2 * tv * cam.aspect)
            v = 0.5 + pc[:, 1] / (z * 2 * tv)
        (x0, y0), (x1, y1) = ev.screen_min, ev.screen_max
        sel = keep & (z > 0) & (u >= x0) & (u <= x1) & (v >= y0) & (v <= y1)
        if protect.value:
            sel &= ~on_table
        apply_bg(sel)

    @auto.on_click
    def _(_):
        m = sc.bg["means"]
        far = np.linalg.norm(m[:, :2], axis=1) > radius.value * r
        faint = sc.bg["opacity"] < min_op.value
        sel = keep & (far | faint)
        if protect.value:
            sel &= ~on_table
        apply_bg(sel)

    @undo_btn.on_click
    def _(_):
        if undo_stack:
            keep[undo_stack.pop()] = True
            refresh_bg()

    tabletop_gui = [intro, box, go, log, sel_md, obj_btn, color, scene_btn, clean, add_btn] + ([planner] if planner else [])

    @mode_btn.on_click
    def _(_):
        """Optional Simulation mode: robot + sim package; the tabletop scene and its controls are hidden."""
        if sim_mode["obj"] is None:
            mode_btn.label = "loading simulation mode..."
            sim_mode["obj"] = SimMode(server, sim_dir)
        ui["sim"] = not ui["sim"]
        ui["sel"], ui["held"] = None, False
        sim_mode["obj"].set_visible(ui["sim"])
        for n in [bg_holder["node"], *nodes.values(), *refl_nodes.values(), *fill_nodes.values()]:
            n.visible = not ui["sim"]
        for g in tabletop_gui:
            g.visible = not ui["sim"]
        gizmo.visible = False
        mode_btn.label = "back to tabletop mode" if ui["sim"] else "switch to simulation mode"

    @scene_btn.on_click
    def _(_):
        v = scene_btn.value
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
