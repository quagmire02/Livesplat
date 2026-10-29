"""Object inspector, layout history, camera presets and inventory downloads for Viser."""
from pathlib import Path

import numpy as np


class EditorPanel:
    def __init__(self, server, ctrl, session, lock, ui, select, run_dir):
        self.server, self.ctrl, self.session = server, ctrl, session
        self.lock, self.ui, self.select = lock, ui, select
        self.run_dir = Path(run_dir)
        self.syncing = False
        gui = server.gui
        self.status = gui.add_markdown("Select an object from the list or click it in the scene.")
        self.pick = gui.add_dropdown("object", self.choices())
        self.details = gui.add_markdown("")
        with gui.add_folder("Name and properties", expand_by_default=False):
            self.name = gui.add_text("name", "")
            self.mass = gui.add_number("mass (kg)", 0.3, min=0.001, max=50.0, step=0.01)
            self.friction = gui.add_number("friction", 0.5, min=0.0, max=2.0, step=0.05)
            self.bounce = gui.add_number("bounce control", 0.1, min=0.0, max=0.95, step=0.05,
                                         hint="Simulation damping control, not a measured coefficient of restitution.")
            apply = gui.add_button("apply name / properties")
        with gui.add_folder("Precise placement", expand_by_default=False):
            gui.add_markdown("Coordinates and dimensions are in **scene units**, not calibrated metres. "
                             "Z = 0 is the tabletop. Arrange mode keeps objects where you put them.")
            self.position = gui.add_vector3("position (scene units)", (0., 0., 0.), step=0.01)
            self.yaw = gui.add_number("upright angle (degrees)", 0., min=-180., max=180., step=5.)
            move = gui.add_button("place upright")
            self.size = gui.add_number("size multiplier", 1., min=0.05, max=20., step=0.1)
            resize = gui.add_button("apply size")
            actions = gui.add_button_group("object actions", ["restore", "remove", "original position"])
            self.actions = actions
        with gui.add_folder("Compare objects", expand_by_default=False):
            self.other = gui.add_dropdown("compare with", self.choices())
            compare = gui.add_button("measure separation")
            self.comparison = gui.add_markdown("Distances use object base centres, not surface clearance.")
        with gui.add_folder("Layout", expand_by_default=False):
            history = gui.add_button_group("history", ["undo", "redo"])
            disk = gui.add_button_group("saved layout", ["save", "load"])
            download = gui.add_button("download object inventory (CSV)")
            gui.add_markdown("Save stores object poses, names, colours, sizes and physics settings. "
                             "Background cleanup is saved separately. Loading pauses physics.")
        with gui.add_folder("View", expand_by_default=False):
            view = gui.add_button_group("camera", ["home", "top", "front", "focus selected"])
        self.edit_widgets = [self.name, self.mass, self.friction, self.bounce, apply, self.position,
                             self.yaw, move, self.size, resize]

        @self.pick.on_update
        def _(ev):
            if not self.syncing:
                with lock:
                    self.select(self.choice_id(self.pick.value))

        @apply.on_click
        def _(ev):
            def edit(k):
                self.session.rename(k, self.name.value)
                # All fields checked before a model rebuild.
                values = [(self.mass.value, 0.001, 50), (self.friction.value, 0, 2), (self.bounce.value, 0, 0.95)]
                if any(not np.isfinite(v) or not lo <= v <= hi for v, lo, hi in values):
                    raise ValueError("Invalid physical property")
                o = ctrl.obj(k)
                o.update(mass=self.mass.value, friction=self.friction.value, bounce=self.bounce.value,
                         property_source="user-entered")
                ctrl.w._build()
            self.edit("edit properties", edit)

        @move.on_click
        def _(ev):
            def edit(k):
                p = np.asarray(self.position.value)
                if not np.isfinite(p).all() or np.abs(p).max() > 1000 or not np.isfinite(self.yaw.value):
                    raise ValueError("Enter finite coordinates within 1000 scene units")
                a = np.radians(self.yaw.value) / 2
                ctrl.w.place(k, p, (np.cos(a), 0, 0, np.sin(a)))
            self.edit("place object", edit)

        @resize.on_click
        def _(ev):
            def edit(k):
                size = self.size.value
                if not np.isfinite(size) or not 0.05 <= size <= 20:
                    raise ValueError("Size must be between 0.05 and 20")
                result = ctrl.t_scale(str(k), size / ctrl.obj(k)["scale"])
                if result.startswith("error:"):
                    raise ValueError(result[7:])
            self.edit("resize object", edit)

        @actions.on_click
        def _(ev):
            def edit(k):
                if actions.value == "original position":
                    if not ctrl.w.present(k):
                        ctrl.t_restore(str(k))
                    ctrl.w.place(k, ctrl.obj(k)["base"], (1, 0, 0, 0))
                else:
                    getattr(ctrl, "t_" + actions.value)(str(k))
            self.edit(actions.value, edit)

        @compare.on_click
        def _(ev):
            with lock:
                k, other = ui["sel"], self.choice_id(self.other.value)
                if k is None or other is None or not ctrl.w.present(k) or not ctrl.w.present(other):
                    self.comparison.content = "Choose two objects that are present in the scene."
                    return
                delta = ctrl.w.pose(other)[0] - ctrl.w.pose(k)[0]
                self.comparison.content = (f"**Base-centre distance:** {np.linalg.norm(delta):.4f} scene units\n\n"
                                           f"On-table distance: {np.linalg.norm(delta[:2]):.4f} · "
                                           f"height difference: {abs(delta[2]):.4f}\n\n"
                                           "This is not a surface-to-surface clearance or fit guarantee.")

        @history.on_click
        def _(ev):
            with lock:
                ui["paused"] = True
                self.status.content = getattr(session, history.value)()
                self.select(ui["sel"])

        @disk.on_click
        def _(ev):
            with lock:
                try:
                    if disk.value == "save":
                        session.save()
                        self.status.content = "**Layout saved.**"
                    else:
                        session.load()
                        ui["paused"] = True
                        self.status.content = "**Layout loaded.** Physics paused; undo returns to your previous layout."
                        self.select(ui["sel"])
                except (OSError, ValueError, KeyError, TypeError) as e:
                    self.status.content = f"Could not {disk.value} layout: {e}"

        @download.on_click
        def _(ev):
            if ev.client is not None:
                with lock:
                    data = session.inventory_csv().encode("utf-8")
                ev.client.send_file_download("livesplat-inventory.csv", data, save_immediately=True)

        @view.on_click
        def _(ev):
            if ev.client is None:
                return
            with lock:
                r = ctrl.sc.radius
                target = np.array([0., 0., 0.1 * r])
                offset = np.array([1.3 * r, -1.3 * r, 1.1 * r])
                if view.value == "top":
                    offset = np.array([0., -0.001 * r, 2.8 * r])
                elif view.value == "front":
                    offset = np.array([0., -2.5 * r, 0.45 * r])
                elif view.value == "focus selected":
                    k = ui["sel"]
                    if k is None or not ctrl.w.present(k):
                        self.status.content = "Select a visible object first."
                        return
                    target = ctrl.w.pose(k)[0] + [0, 0, np.ptp(ctrl.obj(k)["hull"][:, 2]) / 2]
                    d = max(np.linalg.norm(np.ptp(ctrl.obj(k)["hull"], axis=0)), r * 0.2)
                    offset = np.array([1.3, -1.3, 0.8]) * d
                ev.client.camera.position = tuple(target + offset)
                ev.client.camera.look_at = tuple(target)

        self.refresh()

    def choices(self):
        return ["Select an object"] + [f"{o['id']}: {self.ctrl.label(o['id'])}" +
                                       (" (removed)" if o.get("removed") else "") for o in self.ctrl.sc.objects]

    @staticmethod
    def choice_id(value):
        return int(value.split(":", 1)[0]) if ":" in value else None

    def edit(self, label, fn):
        with self.lock:
            k = self.ui["sel"]
            if k is None:
                self.status.content = "Select an object first."
                return
            before = self.session.snapshot()
            try:
                fn(k)
            except (ValueError, KeyError, TypeError) as e:
                self.session.restore(before)
                self.status.content = str(e)
                return
            self.session.record(label, before)
            self.status.content = f"**{label.capitalize()}** · undo is available."
            self.select(k)

    def refresh(self):
        self.syncing = True
        try:
            choices = self.choices()
            self.pick.options = choices
            other_id = self.choice_id(self.other.value)
            self.other.options = choices
            self.other.value = next((s for s in choices if self.choice_id(s) == other_id), choices[0])
            k = self.ui["sel"]
            self.pick.value = next((s for s in choices if self.choice_id(s) == k), choices[0])
            for widget in self.edit_widgets:
                widget.disabled = k is None
            self.actions.visible = k is not None
            if k is None:
                self.details.content = f"**{len(self.ctrl.sc.objects)} objects** · scan dimensions are uncalibrated."
                return
            o = self.ctrl.obj(k)
            self.name.value = self.ctrl.label(k)
            self.mass.value, self.friction.value, self.bounce.value = o["mass"], o["friction"], o["bounce"]
            self.size.value = o["scale"]
            ext = np.ptp(o["hull"], axis=0)
            self.details.content = (f"**{self.ctrl.label(k)}** · object {k}" + (" · removed" if o.get("removed") else "") +
                                    f"\n\nW × D × H: **{ext[0]:.3f} × {ext[1]:.3f} × {ext[2]:.3f} scene units**" +
                                    f"\n\nProperties: {o.get('property_source', 'scan estimates/defaults')}. "
                                    "Dimensions follow the collision hull; unseen surfaces are approximate.")
            if self.ctrl.w.present(k):
                p, q = self.ctrl.w.pose(k)
                self.position.value = tuple(p)
                self.yaw.value = float(np.degrees(np.arctan2(2 * (q[0] * q[3] + q[1] * q[2]),
                                                               1 - 2 * (q[2] ** 2 + q[3] ** 2))))
        finally:
            self.syncing = False
