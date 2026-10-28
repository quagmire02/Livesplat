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
    """Simulation mode inside any Viser server: package splats + robot under /sim, GUI in one folder."""

    def __init__(self, server, sim_dir):
        sim = Path(sim_dir)
        self.server = server
        self.meta, self.rep, self.arm = load(sim)
        m, spec = self.arm.m, self.arm.spec
        self.d = mujoco.MjData(m)
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
        self.link_nodes = {}
        for bid, parts in robot_meshes(m, spec.prefix).items():
            frame = server.scene.add_frame(f"/sim/robot/{m.body(bid).name}", show_axes=False)
            for j, (v, f, rgb) in enumerate(parts):
                server.scene.add_mesh_simple(f"/sim/robot/{m.body(bid).name}/g{j}", vertices=v, faces=f,
                                             color=tuple(int(255 * c) for c in rgb), flat_shading=False)
            self.link_nodes[bid] = frame
        mujoco.mj_resetData(m, self.d)
        self.home = self.d.qpos.copy()
        self.home[self.arm.qadr] = spec.home
        self.show(self.home)

        rows = {row["object"]: row for row in self.rep["objects"]}
        self.folder = server.gui.add_folder("simulation")
        with self.folder:
            server.gui.add_markdown(f"**{spec.name}** · gripper {spec.max_opening * 100:.0f} cm · "
                                    f"units: {meta.get('units', 'scene units')}")
            pick = server.gui.add_dropdown("object ", list(rows))
            go = server.gui.add_button("grasp + lift")
            info = server.gui.add_markdown("")
        busy = threading.Lock()

        @go.on_click
        def _(_):
            if not busy.acquire(blocking=False):
                return
            try:
                row = rows[pick.value]
                o = next(x for x in meta["objects"] if x["body"] == row["body"])
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
