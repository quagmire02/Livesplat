"""LiveSplat viewer: background + one Viser splat node per object, in the table frame (Z up).

Run (WSL): python -m viewer.app <run_dir> [--src thrift_sfm] [--port 8080]
Object node origin = object base centre (xy centroid, z = 0 on the table).
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import viser

from scene_io import splat


class Scene:
    def __init__(self, run_dir, src="thrift_sfm", max_bg=1_200_000, crop=1.6):
        run = Path(run_dir)
        al = json.loads((run / "align.json").read_text())
        self.T = np.array(al["T_world_to_table"])
        self.radius = al["radius"]
        lab_path = run / "labels.npy" if (run / "labels.npy").exists() else run / "labels_geo.npy"
        labels = np.load(lab_path)
        raw = splat.transform(splat.load_raw(run / src / "splat.ply"), self.T)
        m = raw["means"]
        near = (np.linalg.norm(m[:, :2], axis=1) < crop * self.radius) & (m[:, 2] > -2.5 * self.radius) \
            & (m[:, 2] < 2.0 * self.radius)
        bg_idx = np.flatnonzero((labels == 0) & near)
        if len(bg_idx) > max_bg:                         # keep the most opaque background splats
            bg_idx = bg_idx[np.argsort(-raw["opacity"][bg_idx])[:max_bg]]
        self.bg = splat.activate(splat.subset(raw, bg_idx))
        self.objects = []
        for k in range(1, labels.max() + 1):
            a = splat.activate(splat.subset(raw, labels == k))
            base = np.array([*np.median(a["means"][:, :2], 0), 0.0], np.float32)
            a["means"] = a["means"] - base
            self.objects.append({"id": k, "base": base, "splat": a})


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
    server = viser.ViserServer(port=a.port)
    server.scene.set_up_direction("+z")
    add_splat(server, "/bg", sc.bg)
    nodes = []
    for o in sc.objects:
        h = add_splat(server, f"/obj{o['id']}", o["splat"])
        h.position = tuple(o["base"])
        nodes.append(h)
    print(f"{len(sc.objects)} objects, table radius {sc.radius:.3f}")

    r = sc.radius

    @server.on_client_connect
    def _(client):
        client.camera.position = (1.6 * r, -1.6 * r, 1.4 * r)
        client.camera.look_at = (0.0, 0.0, 0.0)

    # Manual pose controls (physics replaces these later).
    pick = server.gui.add_dropdown("object", [str(o["id"]) for o in sc.objects])
    sx = server.gui.add_slider("dx", -r, r, 0.005, 0.0)
    sy = server.gui.add_slider("dy", -r, r, 0.005, 0.0)
    sz = server.gui.add_slider("lift", 0.0, r, 0.005, 0.0)
    yaw = server.gui.add_slider("yaw deg", -180, 180, 1, 0)
    hide = server.gui.add_checkbox("hide objects", False)
    reset = server.gui.add_button("reset all")
    offs = {o["id"]: [0.0, 0.0, 0.0, 0.0] for o in sc.objects}

    def apply(_=None):
        k = int(pick.value)
        offs[k] = [sx.value, sy.value, sz.value, yaw.value]
        for o, h in zip(sc.objects, nodes):
            dx, dy, dz, yw = offs[o["id"]]
            h.position = tuple(o["base"] + np.array([dx, dy, dz]))
            t = np.radians(yw) / 2
            h.wxyz = (np.cos(t), 0.0, 0.0, np.sin(t))
            h.visible = not hide.value

    def load_pick(_=None):
        dx, dy, dz, yw = offs[int(pick.value)]
        sx.value, sy.value, sz.value, yaw.value = dx, dy, dz, yw

    for s in (sx, sy, sz, yaw, hide):
        s.on_update(apply)
    pick.on_update(load_pick)

    @reset.on_click
    def _(_):
        for k in offs:
            offs[k] = [0.0, 0.0, 0.0, 0.0]
        load_pick()
        apply()

    while True:
        time.sleep(1.0)


if __name__ == "__main__":
    main()
