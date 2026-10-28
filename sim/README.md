# sim/ — Simulation mode (branch real-to-sim)
Scope: OPTIONAL mode. The core product is scan -> tag -> move objects (viewer/, agent/). Plan: docs/REAL_TO_SIM.md.
Order: export -> robot -> grasp -> view. Each step reads the package from the step before.
Files:
- export.py — `python -m sim.export <run_dir> [--scale m_per_unit --scale-source TEXT] [--test] [--collision primitives|mesh]`. Writes <run>/sim (or --out): scene.xml, urdf/, meshes/, splats/, metadata.json (units + scale source), render.py, README.md. --scale rescales geometry, splats and cameras to metres.
- render.py — SplatRenderer (gsplat), self-contained, copied into packages.
- check_render.py, variants.py — render check vs real frames; randomized variants.
- robot_spec.py + robots/*.yaml — Robot description (model, joints, actuators, TCP, approach/close axes, gripper, IK mode full|axis, base gap, side min height, approach tolerance). Labs add their own yaml.
- robot.py — `python -m sim.robot <sim_dir> --robot panda|so100|my.yaml`. Base search, top/side grasp planning, multi-start IK, collision check (mount contacts ignored). Writes robot_scene_<name>.xml, checks/robot_report_<name>.json.
- grasp.py — `python -m sim.grasp <sim_dir> [--mode auto|top|side|both]`. Arm class (actuators from the spec), dynamic grasps.
- synthetic.py — Known-answer package (table + boxes). Use it to check a new robot.yaml.
- view.py — SimMode(server, sim_dir): package splats + robot under /sim, one GUI folder, set_visible(). Used by viewer.app ("simulation" scene button) and standalone (python -m sim.view <sim_dir>).
- run.sh — Runs a module with CUDA_HOME/PATH for the gsplat JIT.
Results:
- table1 metric (scale 0.64 from typical product heights): Panda top-down SUCCESS on can (16.3 cm) and toilet roll (17.0 cm); bottle fails (top: pump head; side: slips). SO-100: objects too tall/far.
- synthetic: Panda cube top, tall box top + side SUCCESS; wide box (6 cm) slips. SO-100 small cube top SUCCESS (13.5 cm); tall box out of reach; wide box too wide; low side grasps hit the table.
Limits: scale needs a reference object (A4) for real results; masses are estimates.
