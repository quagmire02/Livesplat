"""Spike 2+3: split one splat into 2 Viser nodes and move one node.

Checks: (a) node rotation also rotates Gaussian shapes, (b) update rate,
(c) the "fuzz" left behind when the object node moves away.

Run (WSL): python -m spikes.viser_nodes <splat.ply> [--radius 0.3] [--max 800000]
Open http://localhost:8080
"""
import argparse
import time

import numpy as np
import viser

from scene_io import splat


def add(server, name, a):
    return server.scene.add_gaussian_splats(
        name, centers=a["means"], covariances=splat.covariances(a),
        rgbs=a["rgb"], opacities=a["opacity"][:, None])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ply")
    ap.add_argument("--radius", type=float, default=0.3, help="object = Gaussians within r of the centre")
    ap.add_argument("--max", type=int, default=800_000, help="keep the most opaque N Gaussians")
    a = ap.parse_args()

    s = splat.load_raw(a.ply)
    if len(s["means"]) > a.max:
        keep = np.argsort(-s["opacity"])[:a.max]
        s = splat.subset(s, keep)
    c = np.median(s["means"], 0)
    obj = np.linalg.norm(s["means"] - c, axis=1) < a.radius
    print(f"{len(obj)} Gaussians, object {obj.sum()}, centre {c.round(3)}")

    server = viser.ViserServer(port=8080)
    bg = splat.activate(splat.subset(s, ~obj))
    ob = splat.activate(splat.subset(s, obj))
    ob["means"] = ob["means"] - c          # object local frame: origin at its centre
    add(server, "/bg", bg)
    h = add(server, "/obj", ob)
    h.position = tuple(c)
    ext = float(np.percentile(np.linalg.norm(s["means"] - c, axis=1), 90))

    @server.on_client_connect
    def _(client):
        client.camera.position = tuple(c + np.array([0.0, -1.0, -1.0]) * ext)
        client.camera.look_at = tuple(c)

    t0, n = time.time(), 0
    while True:
        t = time.time() - t0
        ang = 0.8 * t
        h.wxyz = (np.cos(ang / 2), 0.0, 0.0, np.sin(ang / 2))     # spin about world z
        h.position = tuple(c + np.array([0.0, 0.0, 0.15 * (1 + np.sin(t))]))  # lift off the surface
        n += 1
        if n % 300 == 0:
            print(f"update rate {n / (time.time() - t0):.1f} Hz")
        time.sleep(1 / 60)


if __name__ == "__main__":
    main()
