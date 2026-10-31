"""One command from video to LiveSplat scene, with quality presets and automatic fallback.

Run (WSL, LiveSplat venv):
    LIVESPLAT_ALLOW_META=1 python -m recon.build <video> <name> [--quality low|medium|high] [--dry-run]
                                                [--no-tag] [--no-models]

Steps: stage 1 (ThriftSplat cloud) -> global SfM -> splat training -> export -> segment
       -> Muse + SAM tagging (names, properties, scale) -> object refinement at native resolution
          (medium/high, only if the video has more pixels than the base training used)
       -> Hunyuan3D-2mv models per object (also the collision shapes). Close the viewer first (RAM).
Fallback: the requested preset is lowered until its VRAM and RAM estimates fit; if training
still fails with out-of-memory, the next lower preset is tried.
"""
import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
DATA = Path(os.environ.get("LIVESPLAT_DATA", Path.home() / "livesplat_data"))

PRESETS = {   # fps = kept frames per second; cap = max Gaussians; refine = object refinement iters
    "low":    {"fps": 2, "iters": 15000, "width": 1554, "cap": 1_800_000, "refine": 0},
    "medium": {"fps": 3, "iters": 20000, "width": 1554, "cap": 2_000_000, "refine": 3000},
    "high":   {"fps": 4, "iters": 25000, "width": 2072, "cap": 2_200_000, "refine": 6000},
}
ORDER = ["high", "medium", "low"]
MS_PER_ITER_1554 = 216            # measured on RTX 2060, table1
# Peak training VRAM (GB) by Gaussian cap at 1554 px. Measured: 1.47M -> 2.74 GB; ThriftSplat
# notes 2.29M -> 4.35, 2.41M -> 5.34 (superlinear). Interpolated, rounded up.
VRAM_BY_CAP = [(1.5e6, 2.9), (1.8e6, 3.5), (2.0e6, 4.0), (2.2e6, 4.7), (2.4e6, 5.5)]


def sh(cmd, log):
    print("  $", " ".join(map(str, cmd)), flush=True)
    with open(log, "a") as f:
        return subprocess.run(list(map(str, cmd)), stdout=f, stderr=subprocess.STDOUT, cwd=HERE).returncode


def wsl_path(p):
    """Accept WSL paths and (quoted) Windows paths: C:\\x\\v.mov, \\\\wsl.localhost\\Ubuntu-22.04\\home\\u\\v.mov."""
    import re
    if Path(p).exists():
        return str(Path(p))
    m = re.match(r"^[\\/]{2}wsl(?:\.localhost|\$)[\\/][^\\/]+[\\/](.*)$", p, re.I)
    if m:
        return "/" + m.group(1).replace("\\", "/")
    m = re.match(r"^([A-Za-z]):[\\/](.*)$", p)
    if m:
        return f"/mnt/{m.group(1).lower()}/" + m.group(2).replace("\\", "/")
    return p


def probe_video(path):
    out = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
                          "stream=width,height:format=duration", "-of", "json", str(path)],
                         capture_output=True, text=True, check=True).stdout
    j = json.loads(out)
    s = j["streams"][0]
    return int(s["width"]), int(s["height"]), float(j["format"]["duration"])


def free_vram_gb():
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, check=True).stdout
        return float(out.split()[0]) / 1024
    except Exception:
        return 0.0


def free_ram_gb():
    for line in open("/proc/meminfo"):
        if line.startswith("MemAvailable"):
            return int(line.split()[1]) / 1024 ** 2
    return 0.0


def interp(x, table):
    for (x0, y0), (x1, y1) in zip(table, table[1:]):
        if x <= x1:
            return y0 + (y1 - y0) * max(0.0, x - x0) / (x1 - x0)
    return table[-1][1] * x / table[-1][0]


def plan(name, vw, vh, dur, vram, ram):
    p = dict(PRESETS[name])
    native_w = max(518, (vw // 518) * 518)                    # largest multiple of 518 in the video
    p["width"] = min(p["width"], native_w)
    h = int(round(p["width"] * 294 / 518)) // 2 * 2
    p["frames"] = int(dur * p["fps"])
    p["vram_gb"] = round(interp(p["cap"], VRAM_BY_CAP) + 0.8 * max(0.0, (p["width"] / 1554) ** 2 - 1), 2)
    p["ram_gb"] = round(1.5 + 1.3 * p["frames"] * p["width"] * h * 3 / 1e9, 2)
    p["refine_width"] = native_w
    if p["refine"] and native_w < 1.25 * p["width"]:
        p["refine"] = 0                                        # no extra pixels: refinement cannot help
        p["refine_note"] = f"skipped: video {vw}px has no detail beyond training width {p['width']}px"
    p["minutes"] = round(p["iters"] * MS_PER_ITER_1554 * (p["width"] / 1554) ** 2 / 60000
                         + p["frames"] * 3.1 / 60 + p["refine"] * 0.06 / 60 * (native_w / 1554) ** 2, 0)
    p["fits"] = p["vram_gb"] <= vram - 0.6 and p["ram_gb"] <= ram
    return p


def choose(requested, vw, vh, dur, vram, ram):
    for name in ORDER[ORDER.index(requested):]:
        p = plan(name, vw, vh, dur, vram, ram)
        if p["fits"]:
            return name, p
        print(f"  {name}: needs ~{p['vram_gb']} GB VRAM / {p['ram_gb']} GB RAM "
              f"(free {vram:.1f} / {ram:.1f}) -> lower", flush=True)
    return "low", plan("low", vw, vh, dur, vram, ram)


def oom(log):
    t = Path(log).read_text(errors="ignore")[-20000:].lower()
    return "out of memory" in t or "killed" in t or "cuda error" in t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("name")
    ap.add_argument("--quality", choices=ORDER, default="low")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--tag", action="store_true", help="(default now) Muse + SAM tagging; needs LIVESPLAT_ALLOW_META=1")
    ap.add_argument("--no-tag", action="store_true", help="skip tagging")
    ap.add_argument("--no-models", action="store_true", help="skip the Hunyuan 3D models (they also give collision shapes)")
    a = ap.parse_args()

    a.video = wsl_path(a.video)
    if not Path(a.video).exists():
        sys.exit(f"video not found: {a.video}\nIn WSL use a Linux path (e.g. ~/IMG_2985.mov or /mnt/c/Users/you/Videos/x.mov), "
                 "or put a Windows path in quotes.")
    vw, vh, dur = probe_video(a.video)
    vram, ram = free_vram_gb(), free_ram_gb()
    print(f"video {vw}x{vh}, {dur:.1f}s | free VRAM {vram:.1f} GB, RAM {ram:.1f} GB")
    name, p = choose(a.quality, vw, vh, dur, vram, ram)
    print(f"preset: {name} (requested {a.quality}) -> {json.dumps(p)}")
    if a.dry_run:
        return

    run = DATA / "runs" / a.name
    run.mkdir(parents=True, exist_ok=True)
    log = run / "build.log"
    json.dump({"requested": a.quality, "preset": name, "plan": p, "video": [vw, vh, dur]},
              open(run / "build.json", "w"), indent=1)
    t0 = time.time()

    if not (run / "thrift" / "scene" / "scene.json").exists():
        if sh(["bash", "recon/run_thrift.sh", a.video, a.name, "--stage", "cloud", "--fps", p["fps"]], log):
            sys.exit(f"stage 1 failed, see {log}")

    while True:
        rc = sh(["bash", "recon/sfm_and_train.sh", a.name, p["iters"], p["width"], p["cap"]], log)
        if rc == 0:
            break
        if not oom(log) or name == "low":
            sys.exit(f"training failed (rc {rc}), see {log}")
        name = ORDER[ORDER.index(name) + 1]
        print(f"  out of memory -> retry with {name}", flush=True)
        keep_fps = p["fps"]
        p = plan(name, vw, vh, dur, vram, ram)
        p["fps"] = keep_fps                                    # frames already extracted

    py = sys.executable
    if sh([py, "-m", "segment.run", run], log):
        sys.exit(f"segmentation failed, see {log}")
    tagged = False
    if not a.no_tag:                                           # Muse names/properties/scale + SAM labels
        if sh([py, "-m", "agent.tag", run], log):
            print(f"  tagging failed; the scene works without tags. See {log}")
        else:
            tagged = True
    if p["refine"]:
        rc = sh(["bash", "recon/refine.sh", run, "--iters", p["refine"], "--width", p["refine_width"]], log)
        if rc:
            print(f"  refinement failed (rc {rc}); keeping the base splat. See {log}")
    if tagged and not a.no_models:                             # Hunyuan3D-2mv models (+ collision shapes)
        if sh([py, "-m", "viewer.model3d", "build", run], log):
            print(f"  3D models failed; collision uses the scan hull, 3D model button unavailable. See {log}")
    print(f"done: {name} in {(time.time() - t0) / 60:.0f} min -> {run}")


if __name__ == "__main__":
    main()
