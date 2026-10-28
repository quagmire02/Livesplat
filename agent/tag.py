"""Automatic tagging after a scan, one command (paid, cached, budget-guarded):
Spark names + properties + real-world scale, then SAM 3.1 object labels.

Run (WSL): python -m agent.tag <run_dir> [--src thrift_sfm]
Needs segment.run done (align.json, labels_geo.npy). Cost ~ $0.02 (Spark) + ~$0.0025 per SAM image.
"""
import argparse
import subprocess
import sys


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("--src", default="thrift_sfm")
    a = ap.parse_args()
    for mod in ("agent.properties", "segment.sam_vote"):
        print(f"== {mod}", flush=True)
        rc = subprocess.run([sys.executable, "-m", mod, a.run_dir, "--src", a.src]).returncode
        if rc:
            sys.exit(f"{mod} failed (rc {rc})")


if __name__ == "__main__":
    main()
