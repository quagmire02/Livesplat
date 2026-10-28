"""T8: domain-randomized variants of a sim package (pose, yaw, mass, friction, colour tint).

Run (WSL): python -m sim.variants <sim_dir> [--n 8] [--seed 0] [--render]
Writes <sim_dir>/variants/v###/{scene.xml, metadata.json}. Meshes and splats are shared (../../).
Each variant is checked: loads in MuJoCo, objects settle on the table. --render saves a contact sheet.
"""
import argparse
import copy
import json
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

RANGES = {"mass_scale": (0.5, 2.0), "friction": (0.25, 1.0), "tint": (0.75, 1.25)}


def in_poly(p, poly):
    x, y = p
    inside = False
    for (x1, y1), (x2, y2) in zip(poly, np.roll(poly, -1, 0)):
        if (y1 > y) != (y2 > y) and x < (x2 - x1) * (y - y1) / (y2 - y1 + 1e-12) + x1:
            inside = not inside
    return inside


def sample_layout(objs, poly, rng, tries=2000):
    """Non-overlapping footprints (circles), each fully inside the shrunk table polygon."""
    c = poly.mean(0)
    placed = []
    for o in objs:
        rad = 0.5 * np.hypot(*o["size_xyz"][:2])
        shrink = c + (poly - c) * max(0.2, 1 - rad / max(np.linalg.norm(poly - c, axis=1).min(), 1e-6))
        for _ in range(tries):
            p = c + rng.uniform(-1, 1, 2) * np.abs(poly - c).max(0)
            if in_poly(p, shrink) and all(np.linalg.norm(p - q) > rad + rq + 0.005 for q, rq in placed):
                placed.append((p, rad))
                break
        else:
            return None
    return [p for p, _ in placed]


def make_variant(sim, meta, xml_text, rng, i):
    objs = meta["objects"]
    poly = np.array(meta["table"]["polygon_xy"])
    layout = sample_layout(objs, poly, rng)
    if layout is None:
        return None
    root = ET.fromstring(xml_text)
    root.find("compiler").set("meshdir", "../../meshes")
    vm = copy.deepcopy(meta)
    for o, o_v, xy in zip(objs, vm["objects"], layout):
        yaw = rng.uniform(-np.pi, np.pi)
        q = [np.cos(yaw / 2), 0.0, 0.0, np.sin(yaw / 2)]
        pos = [float(xy[0]), float(xy[1]), float(o["pose"]["pos"][2])]
        mass = o["mass"]["value"] * rng.uniform(*RANGES["mass_scale"])
        mu = rng.uniform(*RANGES["friction"])
        tint = rng.uniform(*RANGES["tint"], 3).round(3).tolist()
        body = root.find(f".//body[@name='{o['body']}']")
        body.set("pos", " ".join(f"{v:.6g}" for v in pos))
        body.set("quat", " ".join(f"{v:.6g}" for v in q))
        col = body.find(f"geom[@name='{o['body']}_col']")
        col.set("mass", f"{mass:.6g}")
        col.set("friction", f"{mu:.4g} 0.005 0.0001")
        o_v["pose"] = {"pos": pos, "quat_wxyz": q}
        o_v["mass"] = {"value": mass, "source": "randomized"}
        o_v["friction"] = {"value": mu, "source": "randomized"}
        o_v["tint_rgb"] = tint
        o_v["files"] = {k: "../../" + v for k, v in o["files"].items()}
    vm["splats"]["background"] = "../../" + meta["splats"]["background"]
    vm["variant"] = {"index": i, "ranges": RANGES}
    d = sim / "variants" / f"v{i:03d}"
    d.mkdir(parents=True, exist_ok=True)
    ET.ElementTree(root).write(d / "scene.xml")
    (d / "metadata.json").write_text(json.dumps(vm, indent=1))
    return d


def check(d, seconds=1.5):
    import mujoco
    m = mujoco.MjModel.from_xml_path(str(d / "scene.xml"))
    data = mujoco.MjData(m)
    mujoco.mj_forward(m, data)
    names = [m.body(i).name for i in range(m.nbody) if m.body(i).name.startswith("obj_")]
    p0 = {b: data.body(b).xpos.copy() for b in names}
    for _ in range(int(seconds / m.opt.timestep)):
        mujoco.mj_step(m, data)
    return max(np.linalg.norm(data.body(b).xpos - p0[b]) for b in names), (m, data)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("sim_dir")
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--render", action="store_true")
    a = ap.parse_args()
    sim = Path(a.sim_dir)
    meta = json.loads((sim / "metadata.json").read_text())
    xml_text = (sim / "scene.xml").read_text()
    rng = np.random.default_rng(a.seed)
    tiles = []
    ok = 0
    for i in range(a.n):
        d = make_variant(sim, meta, xml_text, rng, i)
        if d is None:
            print(f"v{i:03d}: no layout found")
            continue
        drift, (m, data) = check(d)
        good = drift < 0.02
        ok += good
        print(f"v{i:03d}: settle drift {drift * 100:.2f} cm {'OK' if good else 'UNSTABLE'}")
        if a.render:
            from sim.render import SplatRenderer
            r = SplatRenderer(d)
            tiles.append(r.render(data, next(iter(r.cams)), width=518))
    print(f"{ok}/{a.n} variants load and settle")
    if tiles:
        from PIL import Image
        cols = 4
        rows = [np.concatenate(tiles[j:j + cols] + [np.zeros_like(tiles[0])] * (cols - len(tiles[j:j + cols])), 1)
                for j in range(0, len(tiles), cols)]
        Image.fromarray(np.concatenate(rows, 0)).save(sim / "variants" / "contact_sheet.jpg", quality=88)
        print("contact sheet ->", sim / "variants" / "contact_sheet.jpg")


if __name__ == "__main__":
    main()
