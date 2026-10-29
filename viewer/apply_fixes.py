"""Apply the SAM-suggested attachment fix to every object (or listed ids) and save visual_repairs.npz.

Run (WSL, cached SAM, $0 on tagged scenes): python -m viewer.apply_fixes <run_dir> [--src thrift_sfm] [--ids 1 2 3]
Same result as "apply fix" in the viewer: splats outside the SAM masks in 2+ views are erased (fuzz in the air)
or kept on the table (only those in the table band). Pinning fuzz to the table left a ghost where objects stood. Existing repairs in the file are kept; the viewer loads the file at startup.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from scene_io.thrift import load_cameras
from viewer.app import Scene
from viewer.sam_isolate import isolate

KEEP_ON_TABLE, ERASE = 2, -1
TABLE_BAND = 0.02          # x table radius: splats this low are table, everything above outside the masks is fuzz


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("--src", default="thrift_sfm")
    ap.add_argument("--ids", type=int, nargs="*")
    a = ap.parse_args()
    run = Path(a.run_dir)
    sc = Scene(run, a.src)
    props = json.loads((run / "props.json").read_text())
    cams = load_cameras(run / a.src)
    path = run / "visual_repairs.npz"
    layers = {}                                            # same layers and fingerprints as viewer.repair
    for o in sc.objects:
        for part in ("splat", "contact"):
            arr = o.get(part)
            if arr is not None:
                layers[f"{o['id']}_{part}"] = (np.zeros(len(arr["means"]), np.int8),
                                               hashlib.sha256(arr["means"].tobytes()).hexdigest())
    if path.exists():
        with np.load(path, allow_pickle=False) as old:
            for name, (state, h) in layers.items():
                if name in old and str(old[name + "_hash"]) == h and old[name].shape == state.shape:
                    state[:] = old[name]
    for o in sc.objects:
        k = o["id"]
        if a.ids and k not in a.ids:
            continue
        p = o.get("capture_means", o["splat"]["means"]) + o["base"]
        world = (p - sc.T[:3, 3]) @ sc.T[:3, :3]
        rep = isolate(world, props.get(str(k), {}).get("sam_prompt", f"object {k}"), cams, cache_only=True)
        state = layers[f"{k}_splat"][0]
        state[state == KEEP_ON_TABLE] = 0                     # undo the old "pin everything" fix
        ids = rep["spill_ids"][rep["spill_ids"] < len(state)]
        low = o["splat"]["means"][ids, 2] < TABLE_BAND * sc.radius
        state[ids[low]] = KEEP_ON_TABLE                        # table bits stay on the table
        state[ids[~low]] = ERASE                               # fuzz around the object is removed
        print(f"obj {k}: {low.sum():,} table splats kept on the table, {(~low).sum():,} fuzz splats erased")
    payload = {}
    for name, (state, h) in layers.items():
        payload[name], payload[name + "_hash"] = state, h
    tmp = path.with_suffix(".tmp.npz")
    np.savez_compressed(tmp, **payload)
    tmp.replace(path)
    print(f"saved {path}")


if __name__ == "__main__":
    main()
