"""T9: view a sim package in the browser: photoreal splats + Franka Panda meshes, run grasps live.

Run (WSL): python -m sim.view <sim_dir> [--port 8080]   then open http://localhost:8080
Needs robot_scene.xml + checks/robot_report.json (python -m sim.robot <sim_dir>).
"""
import argparse
import json
import threading
import time
from pathlib import Path

import mujoco
import numpy as np
import viser

from scene_io import splat
from sim.grasp import load, run_grasp


def add_splat(server, name, a):
    return server.scene.add_gaussian_splats(name, centers=a["means"], covariances=splat.covariances(a),
                                            rgbs=a["rgb"], opacities=a["opacity"][:, None])


def robot_meshes(m, P):
    """Visual meshes of the Panda, grouped per body, vertices in the body frame."""
    out = {}
    for g in range(m.ngeom):
        b = m.geom_bodyid[g]
        if not m.body(b).name.startswith(P) or m.geom_type[g] != mujoco.mjtGeom.mjGEOM_MESH:
            continue
        if m.geom_contype[g] or m.geom_conaffinity[g]:          # collision geoms: skip, keep visual ones
            continue
        mid = m.geom_dataid[g]
        v = m.mesh_vert[m.mesh_vertadr[mid]:m.mesh_vertadr[mid] + m.mesh_vertnum[mid]]
        f = m.mesh_face[m.mesh_faceadr[mid]:m.mesh_faceadr[mid] + m.mesh_facenum[mid]]
        R = np.zeros(9)
        mujoco.mju_quat2Mat(R, m.geom_quat[g])
        v = v @ R.reshape(3, 3).T + m.geom_pos[g]
        out.setdefault(b, []).append((v.astype(np.float32), f.astype(np.int32), m.geom_rgba[g][:3]))
    return out


class SimMode:
    """Simulation mode inside any Viser server: package splats + a robot under /sim, GUI in one folder.
    The user picks the arm (sim/robots/*.yaml or their own yaml) and where it sits (x, y, yaw or auto)."""

    def __init__(self, server, sim_dir):
        from sim.robots_list import robot_choices
        self.sim = sim = Path(sim_dir)
        self.server = server
        self.meta, self.rep, self.arm = load(sim)
        meta = self.meta
        self.root = server.scene.add_frame("/sim", show_axes=False)
        add_splat(server, "/sim/bg", splat.activate(splat.load_raw(sim / meta["splats"]["background"])))
        self.obj_nodes, self.fill_nodes, self.rest = {}, {}, {}
        for o in meta["objects"]:
            b = o["body"]
            self.obj_nodes[b] = add_splat(server, f"/sim/{b}", splat.activate(splat.load_raw(sim / o["files"]["splat"])))
            fp = sim / o["files"]["splat"].replace(".ply", "_fill.ply")
            if fp.exists():
                self.fill_nodes[b] = add_splat(server, f"/sim/{b}_fill", splat.activate(splat.load_raw(fp)))
                self.fill_nodes[b].visible = False
            self.rest[b] = np.array(o["pose"]["pos"])
        self.link_nodes, self.robot_root = {}, None
        self._load_robot()

        r = meta["table"]["radius"]
        bx, by, _ = self.rep["base"]["pos"]
        self.folder = server.gui.add_folder("simulation")
        busy = threading.Lock()
        with self.folder:
            self.title = server.gui.add_markdown("")
            choices = robot_choices()
            robot = server.gui.add_dropdown("robot arm", list(choices), initial_value=self._robot_label(choices))
            own = server.gui.add_text("or your robot.yaml", "")
            sx = server.gui.add_slider("base x", -2.5 * r, 2.5 * r, 0.005, float(bx))
            sy = server.gui.add_slider("base y", -2.5 * r, 2.5 * r, 0.005, float(by))
            syaw = server.gui.add_slider("base angle deg", -180, 180, 1, float(((self.rep["base"]["yaw_deg"] + 180) % 360) - 180))
            place = server.gui.add_button_group("placement", ["apply", "auto place"])
            pick = server.gui.add_dropdown("object ", [row["object"] for row in self.rep["objects"]])
            go = server.gui.add_button("grasp + lift")
            info = server.gui.add_markdown("")
        self._update_title()

        def replan(auto):
            if not busy.acquire(blocking=False):
                return
            try:
                from sim.robot import build, choose_base, reach_report
                from sim.robot_spec import RobotSpec
                import trimesh
                spec_ref = own.value.strip() or choices[robot.value]
                spec = RobotSpec.load(spec_ref)
                info.content = f"_planning {spec.name}{' (searching base positions)' if auto else ''}..._"
                hulls = {o["body"]: np.asarray(trimesh.load(sim / o["files"]["collision"]).vertices) for o in meta["objects"]}
                if auto:
                    _, base, yaw, rows = choose_base(sim, spec, meta, hulls, 1.0)
                else:
                    base, yaw = (sx.value, sy.value, 0.0), np.radians(syaw.value)
                    rows = reach_report(build(sim, spec, base, yaw), spec, meta, hulls, 1.0)
                build(sim, spec, base, yaw)
                rep = {"robot": spec.name, "robot_yaml": str(spec_ref), "units_per_metre": 1.0, "calibrated": False,
                       "base": {"pos": list(base), "yaw_deg": round(float(np.degrees(yaw)), 1)}, "objects": rows}
                (sim / "checks" / "robot_report.json").write_text(json.dumps(rep, indent=1))
                self.meta, self.rep, self.arm = load(sim)
                self._load_robot()
                sx.value, sy.value = float(base[0]), float(base[1])
                syaw.value = float(((np.degrees(yaw) + 180) % 360) - 180)
                ok = sum((r_["reachable_top_down"] and r_["fits_gripper"]) or r_["side"]["feasible"] for r_ in rows)
                info.content = f"**{spec.name}** placed · can grasp {ok}/{len(rows)} objects"
                self._update_title()
            except Exception as e:                                     # a lab's yaml may be wrong: say why
                info.content = f"**could not load that robot:** {str(e)[:200]}"
            finally:
                busy.release()

        @place.on_click
        def _(_):
            threading.Thread(target=replan, args=(place.value == "auto place",), daemon=True).start()

        @go.on_click
        def _(_):
            if not busy.acquire(blocking=False):
                return
            try:
                row = next(r_ for r_ in self.rep["objects"] if r_["object"] == pick.value)
                o = next(x for x in self.meta["objects"] if x["body"] == row["body"])
                frames = []
                res = run_grasp(self.arm, row, o, self.rep["units_per_metre"], "auto", frames)
                if "lifted_m" not in res:
                    info.content = f"**{pick.value}**: no grasp ({res.get('reason', '')})"
                    return
                info.content = (f"**{pick.value}**: {res['mode']} grasp "
                                f"{'SUCCESS' if res['success'] else 'failed'} · lifted {res['lifted_m'] * 100:.1f} cm")
                for q in frames:
                    self.show(q)
                    time.sleep(0.05)
                time.sleep(1.0)
                self.show(self.home)
            finally:
                busy.release()
        self.set_visible(False)

    def _robot_label(self, choices):
        return next((k for k, v in choices.items() if Path(str(v)).stem == self.arm.spec.name), next(iter(choices)))

    def _update_title(self):
        s = self.arm.spec
        b = self.rep["base"]
        self.title.content = (f"**{s.name}** · gripper {s.max_opening * 100:.0f} cm · base ({b['pos'][0]:.2f}, "
                              f"{b['pos'][1]:.2f}), {b['yaw_deg']:.0f}° · units: {self.meta.get('units', 'scene units')}")

    def _load_robot(self):
        """(Re)create the robot meshes for the current arm."""
        m, spec = self.arm.m, self.arm.spec
        if self.robot_root is not None:
            self.robot_root.remove()
        self.robot_root = self.server.scene.add_frame("/sim/robot", show_axes=False)
        self.link_nodes = {}
        for bid, parts in robot_meshes(m, spec.prefix).items():
            frame = self.server.scene.add_frame(f"/sim/robot/{m.body(bid).name}", show_axes=False)
            for j, (v, f, rgb) in enumerate(parts):
                self.server.scene.add_mesh_simple(f"/sim/robot/{m.body(bid).name}/g{j}", vertices=v, faces=f,
                                                  color=tuple(int(255 * c) for c in rgb), flat_shading=False)
            self.link_nodes[bid] = frame
        self.d = mujoco.MjData(m)
        mujoco.mj_resetData(m, self.d)
        self.home = self.d.qpos.copy()
        self.home[self.arm.qadr] = spec.home
        self.show(self.home)

    def show(self, qpos):
        m, d = self.arm.m, self.d
        d.qpos[:] = qpos
        mujoco.mj_kinematics(m, d)
        with self.server.atomic():
            for bid, node in self.link_nodes.items():
                node.position, node.wxyz = tuple(d.xpos[bid]), tuple(d.xquat[bid])
            for b, node in self.obj_nodes.items():
                p, q = d.body(b).xpos, d.body(b).xquat
                node.position, node.wxyz = tuple(p), tuple(q)
                if b in self.fill_nodes:
                    self.fill_nodes[b].visible = bool(np.linalg.norm(p[:2] - self.rest[b][:2]) > 0.01)

    def set_visible(self, on):
        self.root.visible = on
        self.folder.visible = on


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("sim_dir")
    ap.add_argument("--port", type=int, default=8080)
    a = ap.parse_args()
    server = viser.ViserServer(port=a.port)
    server.scene.set_up_direction("+z")
    server.gui.configure_theme(control_layout="floating", control_width="medium", show_logo=False)
    SimMode(server, a.sim_dir).set_visible(True)
    while True:
        time.sleep(1.0)


if __name__ == "__main__":
    main()
