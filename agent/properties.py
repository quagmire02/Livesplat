"""Object names + physical properties from one image crop per object (Muse Spark vision).

Writes <run>/names.json {id: [name, aliases...]} and <run>/props.json {id: {material, hollow,
mass_kg, friction, bounce}}. The viewer uses props.json for mass/friction/bounce defaults.

Run (WSL, paid, cached):  python -m agent.properties <run_dir> [--src thrift_sfm]
"""
import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

from scene_io.thrift import load_cameras, project

PROMPT = ("This crop shows one object on a table (centre of the image). Reply with JSON only: "
          '{"name": short common name, "aliases": [2-4 other words people might use], '
          '"material": str, "hollow": bool, "mass_kg": float (the real object), '
          '"friction": float 0-1.5 against wood, "bounce": float 0-0.9}')


def best_crop(points_w, cams, pad=0.25):
    """Keyframe where the object's projected box is largest and fully inside the image."""
    best = None
    W, H = cams["width"], cams["height"]
    for f in cams["frames"]:
        uv, z = project(points_w, f["K"], f["c2w"])
        uv = uv[z > 0]
        if len(uv) < 50:
            continue
        x0, y0 = np.percentile(uv, 2, 0)
        x1, y1 = np.percentile(uv, 98, 0)
        if x0 < 0 or y0 < 0 or x1 > W or y1 > H:
            continue
        area = (x1 - x0) * (y1 - y0)
        if best is None or area > best[0]:
            best = (area, f["image"], (x0, y0, x1, y1))
    if best is None:
        return None
    _, path, (x0, y0, x1, y1) = best
    w, h = x1 - x0, y1 - y0
    box = (max(0, x0 - pad * w), max(0, y0 - pad * h), min(W, x1 + pad * w), min(H, y1 + pad * h))
    return Image.open(path).crop(tuple(int(v) for v in box))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("--src", default="thrift_sfm")
    a = ap.parse_args()
    from backends.meta import budget, client
    from scene_io import splat

    run = Path(a.run_dir)
    cams = load_cameras(run / a.src)
    labels = np.load(run / "labels_geo.npy")
    raw = splat.load_raw(run / a.src / "splat.ply")
    names, props = {}, {}
    for k in range(1, labels.max() + 1):
        m = (labels == k) & (1 / (1 + np.exp(-raw["opacity"])) > 0.5)
        crop = best_crop(raw["means"][m], cams)
        if crop is None:
            print(f"obj {k}: no clean view, skipped")
            continue
        crop.save(run / f"crop_obj{k}.jpg", quality=90)
        r = client.spark_chat([{"role": "user", "content": [
            {"type": "text", "text": PROMPT},
            {"type": "image_url", "image_url": {"url": client.image_data_uri(crop)}}]}], max_tokens=2000)
        text = ((r.get("choices") or [{}])[0].get("message") or {}).get("content") or ""
        try:
            j = json.loads(text[text.index("{"): text.rindex("}") + 1])
        except ValueError:
            print(f"obj {k}: could not parse: {text[:200]!r}")
            continue
        names[str(k)] = [j["name"].lower()] + [s.lower() for s in j.get("aliases", [])]
        props[str(k)] = {x: j[x] for x in ("material", "hollow", "mass_kg", "friction", "bounce") if x in j}
        print(f"obj {k}: {names[str(k)]} {props[str(k)]}")
    (run / "names.json").write_text(json.dumps(names, indent=1))
    (run / "props.json").write_text(json.dumps(props, indent=1))
    print(f"spent total ${budget.spent():.4f}")


if __name__ == "__main__":
    main()
