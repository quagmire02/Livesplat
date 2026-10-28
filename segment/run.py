"""Stages 1+2 CLI. Writes align.json, labels_geo.npy and a top-down debug image.

Run (WSL): python -m segment.run <run_dir>     (run_dir = $LIVESPLAT_DATA/runs/<name>)
"""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from scene_io import splat
from scene_io.thrift import load_cameras
from segment.cluster import clean_edges, cluster_objects, grow_labels
from segment.plane import find_table, to_table


def topdown(pts_t, rgb, labels, poly, radius, path, px=800):
    s = px / (2.4 * radius)
    img = np.full((px, px, 3), 255, np.uint8)
    m = np.abs(pts_t[:, :2]).max(1) < 1.2 * radius
    order = np.argsort(pts_t[m, 2])                              # draw low first
    uv = ((pts_t[m][order, :2] * [1, -1]) * s + px / 2).astype(int)
    col = (rgb[m][order][:, ::-1] * 255).astype(np.uint8)
    lab = labels[m][order]
    pal = np.array([[0, 0, 255], [0, 200, 0], [255, 0, 0], [0, 200, 255], [255, 0, 255], [255, 255, 0]])
    col[lab > 0] = pal[(lab[lab > 0] - 1) % len(pal)]
    ok = (uv >= 0).all(1) & (uv < px).all(1)
    img[uv[ok, 1], uv[ok, 0]] = col[ok]
    cv2.polylines(img, [((poly * [1, -1]) * s + px / 2).astype(np.int32)], True, (0, 0, 0), 1)
    cv2.imwrite(str(path), img)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("--src", default="thrift_sfm", help="ThriftSplat output folder inside run_dir")
    a = ap.parse_args()
    run = Path(a.run_dir)
    thrift = run / a.src
    cams = load_cameras(thrift)
    raw = splat.load_raw(thrift / "splat.ply")
    act = splat.activate(raw)

    tb = find_table(act["means"], act["opacity"], [f["c2w"] for f in cams["frames"]])
    pts_t = to_table(tb["T"], act["means"])
    labels = cluster_objects(pts_t, act["opacity"], tb["polygon"], tb["radius"])
    labels = clean_edges(pts_t, act["scales"], act["opacity"], labels)
    labels = grow_labels(pts_t, labels, tb["radius"], tb["polygon"])
    ids = [k for k in range(1, labels.max() + 1) if (labels == k).any()]
    remap = np.zeros(labels.max() + 1, np.int32)
    remap[ids] = np.arange(1, len(ids) + 1)
    labels = remap[labels]

    json.dump({"T_world_to_table": tb["T"].tolist(), "polygon": tb["polygon"].tolist(),
               "radius": tb["radius"], "scene_scale": tb["scene_scale"],
               "plane_thresh": tb["plane_thresh"], "n_table": tb["n_table"]},
              open(run / "align.json", "w"), indent=1)
    np.save(run / "labels_geo.npy", labels)
    topdown(pts_t, act["rgb"], labels, tb["polygon"], tb["radius"], run / "topdown.jpg")
    print(f"table radius {tb['radius']:.3f} (scene units), {tb['n_table']} table pts")
    for k in range(1, labels.max() + 1):
        P = pts_t[labels == k]
        if len(P) == 0:
            continue
        ext = np.percentile(P, 98, 0) - np.percentile(P, 2, 0)
        print(f"  object {k}: {len(P)} Gaussians, extent {ext.round(3)}, centre {P.mean(0)[:2].round(3)}")


if __name__ == "__main__":
    main()
