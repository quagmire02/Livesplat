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
from sim.grasp import try_grasp
from sim.robot import P


def add_splat(server, name, a):
    return server.scene.add_gaussian_splats(name, centers=a["means"], covariances=splat.covariances(a),
                                            rgbs=a["rgb"], opacities=a["opacity"][:, None])


def robot_meshes(m):
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("sim_dir")
    ap.add_argument("--port", type=int, default=8080)
    a = ap.parse_args()
    sim = Path(a.sim_dir)
    meta = json.loads((sim / "metadata.json").read_text())
    rep = json.loads((sim / "checks" / "robot_report.json").read_text())
    m = mujoco.MjModel.from_xml_path(str(sim / "robot_scene.xml"))
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    r = meta["table"]["radius"]

    server = viser.ViserServer(port=a.port)
    server.scene.set_up_direction("+z")
    server.gui.configure_theme(control_layout="floating", control_width="medium", show_logo=False)
    add_splat(server, "/bg", splat.activate(splat.load_raw(sim / meta["splats"]["background"])))
    obj_nodes, fill_nodes, rest = {}, {}, {}
    for o in meta["objects"]:
        b = o["body"]
        obj_nodes[b] = add_splat(server, f"/{b}", splat.activate(splat.load_raw(sim / o["files"]["splat"])))
        fp = sim / o["files"]["splat"].replace(".ply", "_fill.ply")
        if fp.exists():
            fill_nodes[b] = add_splat(server, f"/{b}_fill", splat.activate(splat.load_raw(fp)))
            fill_nodes[b].visible = False
        rest[b] = np.array(o["pose"]["pos"])
    link_nodes = {}
    for bid, parts in robot_meshes(m).items():
        frame = server.scene.add_frame(f"/robot/{m.body(bid).name}", show_axes=False)
        for j, (v, f, rgb) in enumerate(parts):
            server.scene.add_mesh_simple(f"/robot/{m.body(bid).name}/g{j}", vertices=v, faces=f,
                                         color=tuple(int(255 * c) for c in rgb), flat_shading=False)
        link_nodes[bid] = frame

    def show(qpos):
        d.qpos[:] = qpos
        mujoco.mj_kinematics(m, d)
        with server.atomic():
            for bid, node in link_nodes.items():
                node.position, node.wxyz = tuple(d.xpos[bid]), tuple(d.xquat[bid])
            for b, node in obj_nodes.items():
                p, q = d.body(b).xpos, d.body(b).xquat
                node.position, node.wxyz = tuple(p), tuple(q)
                if b in fill_nodes:
                    fill_nodes[b].visible = bool(np.linalg.norm(p[:2] - rest[b][:2]) > 0.01)

    home = np.zeros(m.nq)
    mujoco.mj_resetData(m, d)
    home[:] = d.qpos
    qadr = [m.joint(f"{P}joint{i}").qposadr[0] for i in range(1, 8)]
    home[qadr] = [0, 0, 0, -1.571, 0, 1.571, -0.785]
    show(home)

    @server.on_client_connect
    def _(client):
        bx, by, _ = rep["base"]["pos"]
        client.camera.position = (-1.2 * bx + 0.6 * r, -1.2 * by - 1.2 * r, 1.4 * r + 0.4)
        client.camera.look_at = (0.3 * bx, 0.3 * by, 0.15)

    rows = {row["object"]: row for row in rep["objects"]}
    server.gui.add_markdown(f"**Real-to-sim** · Franka Panda at "
                            f"({rep['base']['pos'][0]:.2f}, {rep['base']['pos'][1]:.2f}) · scale uncalibrated")
    pick = server.gui.add_dropdown("object", list(rows))
    go = server.gui.add_button("grasp + lift")
    info = server.gui.add_markdown("")
    busy = threading.Lock()

    def status(name):
        row = rows[name]
        info.content = (f"reach **{'yes' if row['reachable_top_down'] else 'no'}** · width "
                        f"{row['grasp_width_m'] * 100:.1f} cm (gripper 8 cm) · "
                        f"{'fits' if row['fits_gripper'] else '**too wide**'}")
    status(pick.value)
    pick.on_update(lambda _: status(pick.value))

    @go.on_click
    def _(_):
        if not busy.acquire(blocking=False):
            return
        try:
            row = rows[pick.value]
            if not (row["reachable_top_down"] and row["fits_gripper"]):
                info.content = f"**{pick.value}**: no top-down grasp (" + \
                    ("not reachable" if not row["reachable_top_down"] else "too wide for the gripper") + ")"
                return
            o = next(x for x in meta["objects"] if x["body"] == row["body"])
            frames = []
            res = try_grasp(m, row["body"], row, o, rep["units_per_metre"], frames)
            info.content = (f"**{pick.value}**: {'SUCCESS' if res['success'] else 'failed'} · lifted "
                            f"{res['lifted_m'] * 100:.1f} cm · finger gap {res['finger_gap_m'] * 100:.1f} cm")
            for q in frames:
                show(q)
                time.sleep(0.05)
            time.sleep(1.0)
            show(home)
        finally:
            busy.release()

    while True:
        time.sleep(1.0)


if __name__ == "__main__":
    main()
