"""Approved round: SAM isolation (cached) + Muse analysis for every object, same requests as the viewer card.

Run (WSL): LIVESPLAT_ALLOW_META=1 python -m spikes.analyse_objects <run_dir> [--src thrift_sfm]
Writes <run>/gen/card_<k>.json. The viewer's "load cached analysis" then shows them offline.
"""
import argparse
import json
from pathlib import Path

from agent.controller import Controller
from backends.meta import budget, client
from physics.world import World
from scene_io.thrift import load_cameras
from viewer.app import Scene
from viewer.object_card import CARD_TOKENS, PROMPT_SAM, parse_card
from viewer.sam_isolate import isolate, metrics_text


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("--src", default="thrift_sfm")
    a = ap.parse_args()
    run = Path(a.run_dir)
    sc = Scene(run, a.src)
    ctrl = Controller(sc, World(sc.objects, sc.polygon, floor_z=sc.floor_z), run)
    props = json.loads((run / "props.json").read_text())
    cams = load_cameras(run / a.src)
    start = budget.spent()
    for o in sc.objects:
        k = o["id"]
        p = o.get("capture_means", o["splat"]["means"]) + o["base"]      # same as ObjectCards._object_world
        world = (p - sc.T[:3, 3]) @ sc.T[:3, :3]
        iso = isolate(world, props.get(str(k), {}).get("sam_prompt") or ctrl.label(k), cams, cache_only=False)
        content = [{"type": "text", "text": PROMPT_SAM.format(metrics=metrics_text(iso))}]
        for x in iso["views"]:
            content += [{"type": "image_url", "image_url": {"url": client.image_data_uri(x[im])}}
                        for im in ("thumb", "overlay")]
        r = client.spark_chat([{"role": "user", "content": content}], max_tokens=CARD_TOKENS)
        u = r.get("usage", {})
        try:
            card = parse_card(r)
        except Exception as e:
            print(f"obj {k}: no card ({e}); finish={r.get('choices', [{}])[0].get('finish_reason')} usage={u}")
            continue
        (run / "gen" / f"card_{k}.json").write_text(json.dumps(card, indent=1))
        print(f"\n== obj {k}: {card['name']} ({card['category']}, {card['confidence']}) "
              f"tokens in {u.get('prompt_tokens')} out {u.get('completion_tokens')}")
        print(f"   visible text: {card['visible_text']}")
        for key in ("observations", "uncertainties", "scan_issues", "capture_next"):
            for s in card[key]:
                print(f"   [{key}] {s}")
        print(f"   repair advice: {card['repair_advice']} · search: {card['search_query']}")
    print(f"\nround cost ${budget.spent() - start:.4f}; total ${budget.spent():.4f}")


if __name__ == "__main__":
    main()
