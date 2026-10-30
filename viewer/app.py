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
from viewer.session import EditorSession
from viewer.editor import EditorPanel
from viewer.contact import at_capture, split_contact, table_patch_polygon
from scene_io.appearance import view_rgb

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
    def __init__(self, run_dir, src="thrift_sfm", max_bg=1_200_000, crop=1.6, separate_contact=True):
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
            poly = table_patch_polygon(a, refl, r) if separate_contact else None
            body, contact = split_contact(a, r) if separate_contact else (np.ones(len(a["means"]), bool), None)
            contact_part = {key: v[contact] for key, v in a.items()} if contact is not None and contact.any() else None
            appearance = {key: raw[key][labels == k][body] for key in ("sh0", "shN")}
            moving = {key: v[body] for key, v in a.items()}
            self.objects.append({"id": k, "base": base, "splat": a, "refl": refl, "hull": hull,
                                 "volume": vol, "mass": max(0.01, DEFAULT_DENSITY * vol),
                                 "friction": DEFAULT_FRICTION, "bounce": 0.1,
                                 "fill": fill_splat(self.bg, base, hull, r, polygon=poly),
                                 "contact": contact_part, "appearance": appearance,
                                 "capture_means": a["means"]})
            self.objects[-1]["splat"] = moving
            if separate_contact:
                print(f"  obj {k}: {int(body.sum()):,} movable splats; {int(contact.sum()):,} anchored contact splats")
            fr = run / "gen" / f"fill_rgb_{k}.npy"              # Muse Image table texture (holefill_ai)
            f_ = self.objects[-1]["fill"]
            if not separate_contact and f_ is not None and fr.exists() and len(np.load(fr)) == len(f_["means"]):
                f_["rgb"] = np.load(fr)

        pp = run / "props.json"                         # Spark estimates (agent/properties.py)
        if pp.exists():
            props = json.loads(pp.read_text())
            for o in self.objects:
                p = props.get(str(o["id"]), {})
                o["mass"] = float(p.get("mass_kg", o["mass"]))
                o["friction"] = float(p.get("friction", o["friction"]))
                o["bounce"] = float(p.get("bounce", o["bounce"]))
                o["property_source"] = "AI estimate" if p else "default density / friction"

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
    ap.add_argument("--load-layout", action="store_true", help="restore editor_state.json at startup (physics stays paused)")
    ap.add_argument("--legacy-contact", action="store_true", help="comparison only: keep the old table/object base assignment")
    a = ap.parse_args()

    sc = Scene(a.run_dir, a.src, separate_contact=not a.legacy_contact)
    world = World(sc.objects, sc.polygon, floor_z=sc.floor_z)
    lock = threading.RLock()
    r = sc.radius
    server = viser.ViserServer(port=a.port)
    server.scene.set_up_direction("+z")
    server.gui.configure_theme(control_layout="floating", control_width="medium", show_logo=False)
    bg_holder = {"node": add_splat(server, "/bg", sc.bg)}
    nodes, refl_nodes, fill_nodes, contact_nodes = {}, {}, {}, {}
    appearance_state = {}
    cam = {"pos": np.array([1.3 * r, -1.3 * r, 1.1 * r]), "look": np.array([0.0, 0.0, 0.1 * r]), "updated": 0.0}
    ui = {"sel": None, "held": False, "target": None, "trail": [], "paused": True, "sim": False}

    def build_nodes(k):
        o = next(o for o in sc.objects if o["id"] == k)
        nodes[k] = add_splat(server, f"/obj{k}", o["splat"])
        appearance_state.pop(k, None)
        if o.get("contact") is not None:
            contact_nodes[k] = add_splat(server, f"/contact{k}", o["contact"])
            contact_nodes[k].position = tuple(o["base"])
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
    session = EditorSession(ctrl, a.run_dir, a.src)
    if a.load_layout:
        session.load()
    editor = None
    print(f"{len(sc.objects)} objects, table radius {r:.3f}, floor z {sc.floor_z:.3f}")

    @server.on_client_connect
    def _(client):
        client.camera.position = tuple(cam["pos"])
        client.camera.look_at = tuple(cam["look"])

        @client.camera.on_update
        def _(c):
            cam["pos"], cam["look"] = np.array(c.position), np.array(c.look_at)
            cam["updated"] = time.monotonic()

    # ---- panel (compact) -----------------------------------------------------
    allow_meta = os.environ.get("LIVESPLAT_ALLOW_META") == "1"
    sim_dir = a.sim or next((str(Path(a.run_dir) / n) for n in ("sim_m", "sim")
                             if (Path(a.run_dir) / n / "robot_scene.xml").exists()), None)
    server.gui.configure_theme(
        titlebar_content={"buttons": ({"text": "LiveSplat  ·  scan it, tag it, play with it", "icon": None,
                                        "href": None},), "image": None},
        control_layout="floating", control_width="medium", dark_mode=True, show_logo=False,
        show_share_button=False, brand_color=(255, 140, 66))
    # Four tabs by task: Play (do things), Object (the selected object), Scene (clean the room), Robots (optional).
    tabs = server.gui.add_tab_group()
    play_tab, obj_tab, clean_tab, sim_tab = (tabs.add_tab("Play"), tabs.add_tab("Object"),
                                             tabs.add_tab("Scene"), tabs.add_tab("Robots"))
    with obj_tab:
        server.gui.add_markdown("Click an object in the scene, then open a section.")
        inspect_tab = server.gui.add_folder("Details & placement")
        references_tab = server.gui.add_folder("Identify with AI (SAM 3.1 + Muse)", expand_by_default=False)
        repair_tab = server.gui.add_folder("Fix the scan", expand_by_default=False)
    head = lambda k: ctrl.label(k).split()[-1]
    ids_ = [o["id"] for o in sc.objects]
    if not ids_:
        raise RuntimeError("No objects were found. Check segmentation before opening the editor.")
    ids_ = (ids_ * 3)[:3]
    examples = {"push": f"push the {head(ids_[0])} left", "recolour": f"make the {head(ids_[1])} red",
                "what if": f"what if I knock the {head(ids_[2])} over", "moon": "moon gravity", "reset": "reset"}
    class Toast:
        """Command feedback as a short notification at the bottom instead of a growing log in the panel."""
        visible = True

        @property
        def content(self):
            return ""

        @content.setter
        def content(self, text):
            import html as _html
            body = str(text).replace("**", "").replace("`", "").strip("_ ").replace("\n- ", "\n")
            lines = [_html.escape(x) for x in body.split("\n") if x.strip()][:5]
            toast_state["n"] += 1
            n = toast_state["n"]
            toast_html.content = (
                '<div style="position:fixed;left:50%;bottom:24px;transform:translateX(-50%);z-index:30;'
                'max-width:min(560px,90vw);background:rgba(30,30,30,.92);color:#f3f3f3;border:1px solid #ff8c42;'
                'border-radius:10px;padding:10px 16px;font-size:13px;line-height:1.45;pointer-events:none;'
                'box-shadow:0 4px 18px rgba(0,0,0,.4)">' + "<br>".join(lines) + "</div>")

            def clear():
                if toast_state["n"] == n:                     # a newer message replaced this one
                    toast_html.content = ""
            threading.Timer(4.0, clear).start()

    toast_state = {"n": 0}
    toast_html = server.gui.add_html("")

    with play_tab:
        intro = server.gui.add_markdown("**Click** an object to select it, **drag** its arrows to move it, "
                                        "or **tell it** what to do.")
        box = server.gui.add_text("command", examples["push"] + ", then make it red",
                                  hint="A command (run command) or a goal for Director, e.g. 'slide the bottle next to the can'")
        go = server.gui.add_button("run command")
        direct_btn = server.gui.add_button("plan it · Director", hint="Give a goal ('stack everything into a tower'). "
                                           "Muse proposes plans, physics tests each, Muse revises, the best one plays.")
        direct_md = server.gui.add_markdown("", visible=False)
        try_btn = server.gui.add_button_group("try", list(examples))
        sel_md = server.gui.add_markdown("_nothing selected_")
        obj_btn = server.gui.add_button_group("selected", ["push", "drop", "throw", "remove"])
        color = server.gui.add_dropdown("colour", ["(original)"] + list(COLORS))
        scene_btn = server.gui.add_button_group("scene", ["reset", "pause", "moon g", "earth g"])
        settings = server.gui.add_folder("settings", expand_by_default=False)
    with settings:
        interaction = server.gui.add_dropdown("interaction", ["Arrange", "Experiment"],
                                              hint="Arrange = physics paused while you place things; Experiment = physics runs.")
        throwing = server.gui.add_checkbox("throw on drag release", False)
        detail = server.gui.add_checkbox("view-dependent object detail", True,
                                         hint="Uses colour detail already learned by the scan. Refreshes when camera/object motion settles.")
        mode_info = server.gui.add_markdown("**Arrange** · physics paused. Switch to Experiment to let objects move.")
        allow_baseten = os.environ.get("LIVESPLAT_ALLOW_BASETEN") == "1" and bool(os.environ.get("BASETEN_API_KEY"))
        planner = server.gui.add_dropdown("planner", ["local", "cached"] + (["meta"] if allow_meta else [])
                                          + (["baseten"] if allow_baseten else []),
                                          initial_value="meta" if allow_meta else "baseten" if allow_baseten else "local",
                                          hint="local = offline rules; cached = saved Muse responses only; meta = approved "
                                               "live round; baseten = Muse Spark Open (open weights) on Baseten credit")
    log = Toast()
    with sim_tab:
        server.gui.add_markdown("Test **robot arms** on this table: pick an arm, place it, try grasps.")
        mode_btn = server.gui.add_button("start simulation mode", visible=bool(sim_dir))
        if not sim_dir:
            server.gui.add_markdown("_No simulation package yet. Run `python -m sim.export <run> --scale ...` "
                                    "and `python -m sim.robot <run>/sim_m`._")
    sim_mode = {"obj": None}
    history = []

    from sim.view import SimMode

    def run_command(text):
        with lock:
            before = session.snapshot()
            try:
                results, note = handle(text, ctrl, planner.value if planner else "local", default=ui["sel"])
            except Exception as e:
                session.restore(before)
                log.content = f"Command failed; the scene was restored: {e}"
                return
            changes = [c for c, res in results if c["name"] not in ("measure", "list_objects", "simulate")
                       and not res.startswith("error:")]
            if changes:
                session.record(text, before)
            if any(c["name"] in ("push", "slide", "drop", "throw", "stack") for c in changes):
                ui["paused"] = False
            sync_mode()
        lines = [f"{text}"] + [f"• {res}" for c, res in results]
        if note:
            lines.append(f"• {note}")
        history[:0] = ["\n".join(lines)]
        log.content = "\n".join(lines)
        print("\n".join(lines), flush=True)
        show_selection()

    go.on_click(lambda _: run_command(box.value))
    director_busy = threading.Lock()

    @direct_btn.on_click
    def _(_):
        from agent import director
        provider = planner.value if planner else "local"
        if provider not in ("meta", "baseten"):
            log.content = "Director needs Muse: set planner to meta or baseten (Play > settings)."
            return
        if not director_busy.acquire(blocking=False):
            log.content = "Director is already working on a goal."
            return
        goal = box.value
        direct_btn.disabled = True

        def work():
            try:
                res = director.direct(goal, ctrl, provider, lock=lock, log_dir=Path(a.run_dir) / "director",
                                      progress=lambda s: setattr(log, "content", s))
                rows = [f"**Director** · {goal}"]
                for i, batch in enumerate(res["rounds"]):
                    rows += [f"- r{i + 1}: {b['plan']['idea']} → "
                             + (f"{b['score']:.0f}/10" if b["score"] is not None else "not scored") for b in batch]
                direct_md.content, direct_md.visible = "\n".join(rows[:25]), True
                if res["best"]["plan"] is None:
                    log.content = director.summary(res)
                    return
                with lock:
                    before = session.snapshot()
                    ui["paused"] = False
                    sync_mode()
                log.content = director.summary(res) + "\nplaying it now…"
                director.play(res["best"]["plan"], ctrl, lock)
                session.record(f"director: {goal}", before)
                history[:0] = [director.summary(res)]
                log.content = director.summary(res)
                print(director.summary(res), flush=True)
            except Exception as e:
                log.content = f"Director failed: {e}"
            finally:
                direct_btn.disabled = False
                director_busy.release()
        threading.Thread(target=work, daemon=True).start()

    @try_btn.on_click
    def _(_):
        box.value = examples[try_btn.value]
        run_command(box.value)

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
        cards.select(k if k in ctrl.names else None)
        if k is None or k not in ctrl.names:
            ui["sel"] = None
            gizmo.visible = False
            sel_md.content = "_nothing selected_"
            if editor is not None:
                editor.refresh()
            return
        o = ctrl.obj(k)
        sel_md.content = (f"**{ctrl.label(k)}** · {o['mass']:.2f} kg · friction {o['friction']:.2f} "
                          f"· size x{o['scale']:.2g}")
        gizmo.visible = world.present(k) and not ui["sim"]
        if editor is not None:
            editor.refresh()

    def select_object(k):
        ui["sel"], ui["held"], ui["target"] = k, False, None
        if k is not None and world.present(k):
            p, q = world.pose(k)
            gizmo.position, gizmo.wxyz = tuple(p + mid_offset(k, q)), tuple(q)
        show_selection()
        sync_mode()

    def sync_mode():
        interaction.value = "Arrange" if ui["paused"] else "Experiment"
        mode_info.content = ("**Arrange** · physics paused." if ui["paused"] else
                             "**Experiment** · physics running. Mass and friction are estimates.")

    @interaction.on_update
    def _(ev):
        with lock:
            ui["paused"] = interaction.value == "Arrange"
            sync_mode()

    with inspect_tab:
        editor_root = server.gui.add_folder("Object inspector")
        with editor_root:
            editor = EditorPanel(server, ctrl, session, lock, ui, select_object, a.run_dir)

    from viewer.repair import RepairPanel
    with repair_tab:
        repair = RepairPanel(server, sc, world, ui, lock, nodes, contact_nodes, add_splat, a.run_dir)
        if repair.path.exists():                            # saved repairs (e.g. viewer.apply_fixes) load at startup
            repair.disk("load repairs")
    from viewer.object_card import ObjectCards
    reference_sidebar = server.gui.add_html('')
    def confirm_reference_name(k, name):
        with lock:
            session.checkpoint("confirm reference name")
            ctrl.names[k] = [name.strip()[:120]] + ctrl.names[k]
            show_selection()
    with references_tab:
        cards = ObjectCards(server, sc, ctrl, a.run_dir, a.src, allow_meta, reference_sidebar, confirm_reference_name)
        cards.repair = repair                               # SAM-suggested fixes go through the repair workspace

    @server.scene.on_click()
    def _(ev):
        if ui["sim"]:                                     # simulation mode: tabletop objects are hidden
            return
        if repair.active.value:
            repair.click(ev)
            sync_mode()
            return
        if ui.get("adding"):
            ui["adding"] = False
            add_btn.label = "add missed object (then click it)"
            add_clicked(ev.ray_origin, ev.ray_direction)
            return
        with lock:
            k = pick(ev.ray_origin, ev.ray_direction, sc.objects, world)
            select_object(k)

    @gizmo.on_drag_start
    def _(_):
        if ui["sel"] is not None:
            with lock:
                session.checkpoint("drag object")
                ui["held"], ui["trail"], ui["target"] = True, [], None

    @gizmo.on_update
    def _(_):
        if ui["held"]:
            q = np.array(gizmo.wxyz)
            ui["target"] = (np.array(gizmo.position) - mid_offset(ui["sel"], q), q)
            ui["trail"] = (ui["trail"] + [(time.time(), np.array(gizmo.position))])[-6:]

    @gizmo.on_drag_end
    def _(_):
        with lock:
            if ui["sel"] is not None and ui["target"] is not None and world.present(ui["sel"]):
                world.place(ui["sel"], *ui["target"])
            ui["held"] = False
        k, tr = ui["sel"], ui["trail"]
        if k is None or len(tr) < 2:
            show_selection()
            return
        (t0, p0), (t1, p1) = tr[0], tr[-1]
        if throwing.value and time.time() - t1 < 0.15 and t1 > t0:
            v = 0.6 * (p1 - p0) / (t1 - t0)
            s = np.linalg.norm(v)
            if s > 0.5:                                    # a slow release just puts it down
                v = v * min(1.0, 2.5 / s)
                with lock:
                    world.throw(k, tuple(v))
                    ui["paused"] = False
                    sync_mode()
        show_selection()

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
            with lock:
                session.checkpoint("restore colour")
                o = ctrl.obj(k)
                o["splat"]["rgb"] = o["rgb0"].copy()
                o["color"] = None
                build_nodes(k)
        else:
            run_command(f"make object {k} {color.value}")

    # ---- background clean-up (image-editor style) ------------------------------------
    keep = np.ones(len(sc.bg["means"]), bool)
    undo_stack = []
    rm_path = Path(a.run_dir) / "bg_removed.npy"
    prev_removed = np.load(rm_path) if rm_path.exists() else np.zeros(0, np.int64)   # earlier sessions
    # The whole table column: a glossy top is made of faint splats and reflections below the surface.
    on_table = (sc.bg["means"][:, 2] < 0.03 * r) & in_polygon(sc.bg["means"][:, :2], sc.polygon * 1.05)
    with play_tab:
        add_btn = server.gui.add_button("add missed object (then click it)", disabled=not allow_meta,
                                        hint="Uses SAM. Enable only during an explicitly approved API round.")

    @add_btn.on_click
    def _(_):
        ui["adding"] = True
        add_btn.label = "click the missed object in the scene..."

    def add_clicked(ray_o, ray_d):
        from backends.meta.client import require_permission
        require_permission()
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
            lab_path = Path(a.run_dir) / a.src / "labels.npy"
            if not lab_path.exists():
                lab_path = Path(a.run_dir) / "labels.npy"
            labels = np.load(lab_path) if lab_path.exists() else np.load(Path(a.run_dir) / "labels_geo.npy")
            labels[sc.bg_idx[sel]] = k
            np.save(lab_path, labels)
            session.geometry_changed()
        ui["sel"] = k
        log.content = f"**added object {k}** ({int(sel.sum()):,} splats, {n} views) · try: push object {k}"
        show_selection()

    with clean_tab:
        server.gui.add_markdown("**Shift + drag** a box to erase background splats.")
        protect = server.gui.add_checkbox("protect table", True)
        radius = server.gui.add_slider("keep within (x table)", 1.0, 4.0, 0.1, 2.5)
        min_op = server.gui.add_slider("min opacity", 0.0, 0.5, 0.01, 0.0)
        auto = server.gui.add_button("auto clean")
        undo_btn = server.gui.add_button("undo erase")
        clean_widgets = [protect, radius, min_op, auto, undo_btn]
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
        if ui["sim"] or repair.active.value:
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

    tabletop_gui = [intro, interaction, throwing, detail, mode_info, box, try_btn, go, log, sel_md, obj_btn, color,
                    scene_btn, add_btn, *clean_widgets] + ([planner] if planner else [])

    @mode_btn.on_click
    def _(_):
        """Optional Simulation mode: robot + sim package; the tabletop scene and its controls are hidden."""
        if sim_mode["obj"] is None:
            mode_btn.label = "loading simulation mode..."
            with sim_tab:
                sim_mode["obj"] = SimMode(server, sim_dir)
        ui["sim"] = not ui["sim"]
        reference_sidebar.visible = not ui["sim"]
        repair.active.value = False
        repair.isolate.value = False
        repair.clear()
        repair.tick()
        ui["sel"], ui["held"] = None, False
        editor_root.visible = not ui["sim"]
        sim_mode["obj"].set_visible(ui["sim"])
        for n in [bg_holder["node"], *nodes.values(), *refl_nodes.values(), *fill_nodes.values(), *contact_nodes.values()]:
            n.visible = not ui["sim"]
        for g in tabletop_gui:
            g.visible = not ui["sim"]
        gizmo.visible = False
        mode_btn.label = "back to tabletop mode" if ui["sim"] else "start simulation mode"

    @scene_btn.on_click
    def _(_):
        v = scene_btn.value
        if v == "reset":
            run_command("reset")
        elif v == "pause":
            ui["paused"] = not ui["paused"]
            sync_mode()
            log.content = "_paused_" if ui["paused"] else "_running_"
        else:
            run_command("moon gravity" if v == "moon g" else "earth gravity")

    # ---- loop ----------------------------------------------------------------
    frame = 1 / 60
    last_detail = 0.0
    while True:
        t0 = time.time()
        if not ui["sim"]:
            with lock:
                k = ui["sel"]
                if ui["held"] and k is not None and world.present(k) and ui["target"] is not None:
                    world.place(k, *ui["target"])                 # held: follow the gizmo
                if not ui["paused"]:
                    world.step(frame)
                state = {}
                for o in sc.objects:
                    kk = o["id"]
                    present = world.present(kk)
                    pose = world.pose(kk) if present else None
                    upright = present and abs(pose[0][2] - world.rest[kk][2]) < 0.01 and abs(pose[1][0]) > 0.996
                    captured = at_capture(pose, o["base"], r, o["scale"])
                    state[kk] = (present, pose, captured, upright)
            with server.atomic():
                for kk, (present, pose, rest, upright) in state.items():
                    nodes[kk].visible = present
                    if present:
                        nodes[kk].position, nodes[kk].wxyz = tuple(pose[0]), tuple(pose[1])
                    if kk in refl_nodes:
                        # Captured reflections include table pixels: never translate them with an object.
                        refl_nodes[kk].visible = rest
                    if kk in contact_nodes:
                        contact_nodes[kk].visible = rest
                    if kk in fill_nodes:
                        fill_nodes[kk].visible = not rest
                isolated = int(repair.pick.value) if repair.isolate.value else None
                bg_holder["node"].visible = isolated is None
                if isolated is not None:
                    for group in (nodes, contact_nodes, refl_nodes, fill_nodes):
                        for kk, node in group.items():
                            node.visible = node.visible and kk == isolated
                    for group in (refl_nodes, fill_nodes):
                        for node in group.values():
                            node.visible = False
                if repair.active.value:
                    gizmo.visible = False
                with lock:
                    repair.tick()
                k = ui["sel"]
                if k is not None and not ui["held"] and state.get(k, (False,))[0]:
                    gp, gq = state[k][1]
                    gizmo.position, gizmo.wxyz = tuple(gp + mid_offset(k, gq)), tuple(gq)
            now = time.monotonic()
            if now - last_detail > 0.4 and now - cam["updated"] > 0.2 and not ui["held"]:
                last_detail = now
                with lock:
                    for o in sc.objects:
                        k = o["id"]
                        if not world.present(k) or o.get("appearance") is None or o.get("color") is not None:
                            continue
                        if not ui["paused"] and world.speed(k) > 0.02:
                            continue
                        p, q = world.pose(k)
                        key = np.r_[cam["pos"], p, q, o["scale"], float(detail.value)]
                        if k in appearance_state and np.allclose(key, appearance_state[k], atol=1e-4, rtol=0):
                            continue
                        if detail.value:
                            R = splat.quat_wxyz_to_mat(q[None])[0]
                            local_camera = (cam["pos"] - p) @ R
                            dirs = (o["splat"]["means"] - local_camera) @ sc.T[:3, :3]
                            rgb = view_rgb(o["appearance"]["sh0"], o["appearance"]["shN"], dirs)
                        else:
                            rgb = o["splat"]["rgb"]
                        nodes[k].rgbs = rgb
                        appearance_state[k] = key
        time.sleep(max(0.0, frame - (time.time() - t0)))


if __name__ == "__main__":
    main()
