# sim/ — real-to-sim (branch real-to-sim)
Purpose: Make a simulation package from a LiveSplat run. Plan: docs/REAL_TO_SIM.md.
Order: export -> robot -> grasp -> view. Each step reads the package from the step before.
Files:
- export.py — `python -m sim.export <run_dir> [--test] [--collision primitives|mesh] [--bg-crop 0]`. Writes <run>/sim: scene.xml, urdf/, meshes/, splats/ (objects, reflections, fills, full-room background), metadata.json, render.py, README.md. Default collision = stack of cylinders/boxes per 4 cm band (convex meshes gave 1 contact point: grasps slipped). Option: elliptic cones, impratio 10.
- render.py — SplatRenderer: splats posed by MuJoCo bodies (gsplat). Self-contained; copied into packages. Colour tint, fills.
- check_render.py — Capture cameras at rest vs real keyframes (PSNR), push sequence.
- variants.py — Randomized layouts/mass/friction/tint; load + settle check; contact sheet.
- robot.py — `python -m sim.robot <sim_dir>`. Copies the Panda (Menagerie, Apache-2.0), attaches it (MjSpec), searches 24 base positions, IK top-down reach + side grasp (approach angles base dir ±60°, height with width ≤ 7.5 cm near 40% of height, palm clearance). Collision check of the arm vs table/floor/other objects at pre-grasp and grasp poses; reports blockers. Writes robot_scene.xml, checks/robot_report.json.
- grasp.py — `python -m sim.grasp <sim_dir> [--mode auto|top|side|both]`. Dynamic grasps (top: above -> down -> close -> lift; side: back off 12 cm -> in -> close -> lift). Success = rises >= 5 cm and stays in the hand.
- view.py — `python -m sim.view <sim_dir>`. Browser view: splats + Panda meshes; "grasp + lift" runs live.
- run.sh — Runs a sim module with CUDA_HOME/PATH for the gsplat JIT.
Results (table1): self-test drift 0.00 cm, camera error 7e-7; render 25.4 dB; 8/8 variants settle; can top-down SUCCESS (16 cm), can side BLOCKED (bottle, roll, table); bottle top FAIL (pump head) and side FAIL (only the neck fits, 0.55 kg slips; body 14.5 units wide in the uncalibrated model); roll too wide.
Limits: scale uncalibrated (units_per_metre = 1.0); Panda gripper force from Menagerie gains; top-down grasps only.
