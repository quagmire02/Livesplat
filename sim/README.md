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
- view.py — SimMode(server, sim_dir): package splats + robot under /sim; panel: robot arm dropdown (sim/robots/*.yaml), "or your robot.yaml" path, base x / y / angle sliders, placement apply / auto place (re-plans in a thread, swaps robot meshes), object + grasp. Standalone: python -m sim.view <sim_dir>.
- robot_probe.py — `python -m sim.robot_probe <model.xml> --name myarm`: drafts a robot yaml (joints, actuators, gripper actuator + open/closed, TCP between the finger bodies, approach/close axes, opening) by closing the gripper in sim. Used for vx300s, wx250s, xarm7 (checked on Panda: TCP 9.6 vs 10.3 cm hand-measured).
- robots/: panda, so100, vx300s (ViperX 300 S, ALOHA), wx250s (WidowX 250 S, Bridge), xarm7 (UFACTORY xArm7 + gripper). Menagerie models in ~/livesplat_data/models/mujoco_menagerie (sparse checkout).
- grip_calibrate.py — `python -m sim.grip_calibrate panda ...`: measures the model's squeeze per finger on a 5 cm block and bisects gripper.force_scale (gain + bias + force range) to match gripper.force_n. Menagerie defaults were 1.3 N (Panda, real 70 N) to 402 N (xArm7). Arm applies force_scale; reset clamps joints into their ranges (Trossen fingers started outside them and jammed).
- robots_list.py — robot_choices(): label -> yaml for every sim/robots/*.yaml.
- run.sh — Runs a module with CUDA_HOME/PATH for the gsplat JIT.
Results:
- table1 metric (scale 0.64 from typical product heights): Panda top-down SUCCESS on can (16.3 cm) and toilet roll (17.0 cm); bottle fails (top: pump head; side: slips). SO-100: objects too tall/far.
- synthetic: Panda cube top, tall box top + side SUCCESS; wide box (6 cm) slips. SO-100 small cube top SUCCESS (13.5 cm); tall box out of reach; wide box too wide; low side grasps hit the table.
Limits: scale needs a reference object (A4) for real results; masses are estimates.
