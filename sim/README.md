# sim/ — real-to-sim (branch real-to-sim)
Purpose: Make a simulation package from a LiveSplat run. Plan: docs/REAL_TO_SIM.md.
Files:
- export.py — `python -m sim.export <run_dir> [--test] [--bg-crop 0]`. Writes <run>/sim: scene.xml (MJCF), urdf/, meshes/ (OBJ), splats/ (body frames, raw PLY; full room by default), metadata.json (sources + uncertainty), render.py (copy), README.md. --test: plain MuJoCo load, 2 s settle, camera convention check.
- render.py — SplatRenderer: splats posed by MuJoCo bodies, rendered with gsplat. Self-contained (copied into packages).
- check_render.py — T4: capture cameras at rest vs real keyframes (PSNR full frame + table region), push sequence images.
- run.sh — Runs a sim module with CUDA_HOME/PATH for the gsplat JIT.
Status: T1-T4 done on table1. Drift 0.00 cm, camera error 7e-7. Render vs real: 25.39 dB full, 25.03 dB table region (training cameras: fit check). Package 347 MB.
Note: A cropped background (old default 1.6x radius) gave 14 dB: the room is necessary in the package.
