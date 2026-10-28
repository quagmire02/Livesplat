# sim/ — real-to-sim (branch real-to-sim)
Purpose: Make a simulation package from a LiveSplat run. Plan: docs/REAL_TO_SIM.md.
Files:
- export.py — `python -m sim.export <run_dir> [--test]`. Writes <run>/sim: scene.xml (MJCF), urdf/, meshes/ (OBJ), splats/ (body frames, raw PLY), metadata.json (sources + uncertainty), README.md. --test: plain MuJoCo load, 2 s settle, camera convention check.
Status: T1 + T2 done. table1: 3 bodies, 10 cameras, drift 0.00 cm, camera error 7e-7. Package 155 MB (background splat).
