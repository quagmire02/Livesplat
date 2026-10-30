"""Metric scale of a scan: metres per scene unit, its uncertainty, and where it came from.

Sources, best first: user-entered known size (calibrate), reference object (A4/ArUco, later), LiDAR (later),
Muse typical product sizes (agent.properties, about +-10%). Lab tests vary the scale by +-spread when it is
not calibrated, so a result that depends on size says so.

CLI: python -m measure.scale <run> [<obj id> <height|width|depth> <value> <mm|cm|m|in>]
"""
import json
import re
from pathlib import Path

import numpy as np

UNITS = {"mm": 0.001, "cm": 0.01, "m": 1.0, "in": 0.0254, "inch": 0.0254, "inches": 0.0254}
DIMS = {"tall": "height", "high": "height", "height": "height", "wide": "width", "width": "width",
        "across": "width", "diameter": "width", "long": "depth", "deep": "depth", "depth": "depth"}


def load(run):
    p = Path(run) / "scale.json"
    if not p.exists():
        return {"metres_per_unit": None, "spread": None, "source": "none"}
    s = json.loads(p.read_text())
    s.setdefault("spread", 0.0)
    return s


def extents(points):
    """height = top of the object above the table; width/depth = footprint along its main axes (scene units)."""
    p = np.asarray(points, float)
    h = float(np.percentile(p[:, 2], 98))
    xy = p[:, :2] - p[:, :2].mean(0)
    _, _, vt = np.linalg.svd(xy[::max(1, len(xy) // 20000)], full_matrices=False)
    proj = xy @ vt.T
    w, d = (np.percentile(proj, 99, 0) - np.percentile(proj, 1, 0))
    return {"height": h, "width": float(max(w, d)), "depth": float(min(w, d))}


def calibrate(run, obj_points, dimension, metres, label="object"):
    """User says e.g. 'the can is 15 cm tall' -> metres_per_unit from that object's measured extent."""
    ext = extents(obj_points)[dimension]
    s = {"metres_per_unit": metres / ext, "spread": 0.02,          # ~2%: hand measurement + scan edge
         "source": f"user: {label} {dimension} {metres * 100:.1f} cm", "previous": load(run)}
    s["previous"].pop("previous", None)
    (Path(run) / "scale.json").write_text(json.dumps(s, indent=1))
    return s


def parse_size(text):
    """'the can is 15 cm tall' -> (dimension, metres) or None."""
    m = re.search(r"(\d+(?:\.\d+)?)\s*(mm|cm|m|in|inch|inches)\b\s*(?:in\s+)?(tall|high|height|wide|width|across|"
                  r"diameter|long|deep|depth)", text.lower())
    if not m:
        return None
    return DIMS[m.group(3)], float(m.group(1)) * UNITS[m.group(2)]


def main():
    import sys
    run = sys.argv[1]
    if len(sys.argv) == 2:
        print(json.dumps(load(run), indent=1))
        return
    from viewer.app import Scene
    k, dim, val, unit = int(sys.argv[2]), sys.argv[3], float(sys.argv[4]), sys.argv[5]
    o = next(o for o in Scene(run).objects if o["id"] == k)
    pts = o["capture_means"][o["body_ids"]] if "body_ids" in o else o["capture_means"]
    print(json.dumps(calibrate(run, pts, DIMS.get(dim, dim), val * UNITS[unit], f"object {k}"), indent=1))


if __name__ == "__main__":
    main()
