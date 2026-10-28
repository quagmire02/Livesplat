"""Synthetic sim package (table + boxes) in the same format as sim.export. Labs use it to check a new
robot.yaml before they trust results on a scanned scene; tests use it as a known-answer scene.

    python -m sim.synthetic <out_dir> [--boxes "name:sx,sy,sz@x,y;..."]
Units: metres. Table top at z = 0, round, radius 0.35 m.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import trimesh

DEFAULT = "small cube:0.03,0.03,0.03@0.0,0.0;tall box:0.04,0.04,0.12@0.08,0.06;wide box:0.12,0.06,0.05@-0.09,0.05"


def make(out, boxes=DEFAULT, radius=0.35):
    out = Path(out)
    (out / "meshes").mkdir(parents=True, exist_ok=True)
    ang = np.linspace(0, 2 * np.pi, 32, endpoint=False)
    poly = np.c_[radius * np.cos(ang), radius * np.sin(ang)]
    trimesh.convex.convex_hull(np.concatenate([np.c_[poly, np.zeros(32)], np.c_[poly, np.full(32, -0.03)]])
                               ).export(out / "meshes" / "table.obj")
    objs, bodies = [], []
    for k, spec in enumerate(b for b in boxes.split(";") if b.strip()):
        name, rest = spec.split(":")
        size, pos = rest.split("@")
        sx, sy, sz = (float(v) for v in size.split(","))
        x, y = (float(v) for v in pos.split(","))
        tag = f"obj_{k + 1}"
        box = trimesh.creation.box(extents=(sx, sy, sz))
        box.apply_translation((0, 0, sz / 2))
        box.export(out / "meshes" / f"{tag}_collision.obj")
        bodies.append(f'<body name="{tag}" pos="{x} {y} 0"><freejoint name="{tag}_free"/>'
                      f'<geom name="{tag}_col" type="box" size="{sx / 2} {sy / 2} {sz / 2}" pos="0 0 {sz / 2}" '
                      f'mass="0.1" friction="0.8 0.005 0.0001" condim="4" rgba="0.8 0.3 0.2 1"/></body>')
        objs.append({"id": k + 1, "body": tag, "name": name, "aliases": [], "pose": {"pos": [x, y, 0.0], "quat_wxyz": [1, 0, 0, 0]},
                     "size_xyz": [sx, sy, sz], "files": {"collision": f"meshes/{tag}_collision.obj"},
                     "mass": {"value": 0.1, "source": "synthetic"}})
    (out / "scene.xml").write_text(f"""<mujoco model="synthetic">
  <compiler meshdir="meshes"/>
  <option timestep="0.002" integrator="implicitfast" cone="elliptic" impratio="10"/>
  <asset><mesh name="table" file="table.obj"/></asset>
  <worldbody>
    <light pos="0 0 2" dir="0 0 -1"/>
    <geom name="floor" type="plane" pos="0 0 -0.75" size="0 0 0.05"/>
    <geom name="table" type="mesh" mesh="table" rgba="0.4 0.3 0.2 1" friction="0.6 0.005 0.0001"/>
    {''.join(bodies)}
  </worldbody>
</mujoco>
""")
    (out / "metadata.json").write_text(json.dumps({"format": "livesplat-sim/1", "units": "metres", "objects": objs,
                                                   "table": {"polygon_xy": poly.round(5).tolist(), "radius": radius},
                                                   "cameras": [], "synthetic": True}, indent=1))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--boxes", default=DEFAULT)
    a = ap.parse_args()
    print("synthetic package ->", make(a.out, a.boxes))


if __name__ == "__main__":
    main()
