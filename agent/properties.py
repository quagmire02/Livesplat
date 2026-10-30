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

PROMPT = ("These crops show ONE object on a table from different sides (centre of each image). Reply with JSON only: "
          '{"brand": brand name if you can read it on the object, else "", '
          '"name": specific product type, e.g. "aerosol deodorant spray" not "spray can", '
          '"visible_text": main readable text on it or "unknown", '
          '"aliases": [2-4 other words people might use], '
          '"material": str, "hollow": bool, "mass_kg": float (the real object), '
          '"friction": float 0-1.5 against wood, "bounce": float 0-0.9, '
          '"sam_prompt": 1-3 word concrete noun phrase to segment this object (e.g. "spray can"), '
          '"real_size_cm": [width, depth, height] typical real size of this product in cm, '
          '"size_confidence": "high" if you recognise the exact product type, else "low"}')


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
    ap.add_argument("--names-only", action="store_true",
                    help="refresh names.json (+ brand/visible_text in props.json); keep masses, friction and scale")
    a = ap.parse_args()
    from backends.meta import budget, client
    from scene_io import splat

    run = Path(a.run_dir)
    cams = load_cameras(run / a.src)
    labels = np.load(run / "labels_geo.npy")
    raw = splat.load_raw(run / a.src / "splat.ply")
    T = np.array(json.loads((run / "align.json").read_text())["T_world_to_table"])
    names, props = {}, {}
    for k in range(1, labels.max() + 1):
        m = (labels == k) & (1 / (1 + np.exp(-raw["opacity"])) > 0.5)
        crop = best_crop(raw["means"][m], cams)
        if crop is None:
            print(f"obj {k}: no clean view, skipped")
            continue
        crop.save(run / f"crop_obj{k}.jpg", quality=90)
        from viewer.object_card import source_crops
        views = [im for _, im in source_crops(raw["means"][m], cams, count=3)] or [crop]   # 3 sides: brand + type
        r = client.spark_chat([{"role": "user", "content": [{"type": "text", "text": PROMPT}] + [
            {"type": "image_url", "image_url": {"url": client.image_data_uri(im)}} for im in views]}], max_tokens=3000)
        text = ((r.get("choices") or [{}])[0].get("message") or {}).get("content") or ""
        try:
            j = json.loads(text[text.index("{"): text.rindex("}") + 1])
        except ValueError:
            print(f"obj {k}: could not parse: {text[:200]!r}")
            continue
        brand = str(j.get("brand") or "").strip()
        text = str(j.get("visible_text") or "").strip()
        if not brand and text.lower() != "unknown" and 0 < len(text.split()) <= 2:
            brand = text                                       # short readable word on it, e.g. "SHIELD"
        full = f"{brand} {j['name']}" if brand and brand.lower() not in j["name"].lower() else j["name"]
        extra = [text.lower()] if text.lower() not in ("", "unknown") and len(text.split()) <= 3 else []   # "shield"
        names[str(k)] = list(dict.fromkeys([full.lower(), j["name"].lower()] + [s.lower() for s in j.get("aliases", [])]
                                           + extra))
        props[str(k)] = {x: j[x] for x in ("material", "hollow", "mass_kg", "friction", "bounce", "sam_prompt",
                                           "real_size_cm", "size_confidence", "brand", "visible_text") if x in j}
        P = raw["means"][m] @ T[:3, :3].T + T[:3, 3]           # object points in the table frame
        props[str(k)]["measured_height_units"] = float(np.percentile(P[:, 2], 98))
        print(f"obj {k}: {names[str(k)]} {props[str(k)]}")
    (run / "names.json").write_text(json.dumps(names, indent=1))
    if a.names_only:
        old = json.loads((run / "props.json").read_text()) if (run / "props.json").exists() else {}
        for k, p in props.items():
            old.setdefault(k, {}).update({x: p[x] for x in ("brand", "visible_text") if x in p})
        (run / "props.json").write_text(json.dumps(old, indent=1))
        print(f"names only; spent total ${budget.spent():.4f}")
        return
    (run / "props.json").write_text(json.dumps(props, indent=1))
    # Automatic scale: real heights (Spark world knowledge) / measured heights; cross-checked between objects.
    ratios = {k: p["real_size_cm"][2] / 100 / p["measured_height_units"] for k, p in props.items()
              if p.get("real_size_cm") and p.get("measured_height_units", 0) > 0}
    hi = {k: v for k, v in ratios.items() if props[k].get("size_confidence") == "high"} or ratios
    if hi:
        v = np.array(list(hi.values()))
        scale = {"metres_per_unit": float(np.median(v)), "spread": float(np.ptp(v) / np.median(v)),
                 "per_object": ratios, "used": sorted(hi), "source": "Spark typical product sizes"}
        (run / "scale.json").write_text(json.dumps(scale, indent=1))
        print(f"scale: {scale['metres_per_unit']:.3f} m/unit (spread {scale['spread'] * 100:.0f}% over {len(hi)} objects)")
    print(f"spent total ${budget.spent():.4f}")


if __name__ == "__main__":
    main()
