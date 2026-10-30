# lab/ — Lab bench: tests for robotics labs on their scanned objects

Purpose: answer lab questions with many simulated trials across what we don't know exactly (friction, mass,
scale), plus a suggestion. Not only arms: transport (mobile robots, trays), gripper fit, grasp, camera.

Files:
- objects.py — lab_objects(sc, run): deep copies of the scan's objects; the full 3D model's convex hull (gen/model_<id>/model.glb, flat base, must stand) replaces the scan hull when available.
- transport.py — how hard a mobile robot / tray can accelerate or brake before each object slides or tips. Bisection on a/g per object, 3 directions, friction x0.7/1.0/1.3, surface wood/plastic/rubber. Elliptic friction cone (MuJoCo's default pyramid is 30% weaker diagonally). Scale-free (a ratio). CLI: python -m lab.transport <run> [--surface rubber] -> <run>/lab/transport_<surface>.json.

- gripper.py + grippers/*.yaml — gripper fit without an arm. Parallel jaws: narrowest grip per height slice (3D model shape), fits opening - 5 mm, holds if 2 mu F >= m (g + 0.5 g) 1.5. Suction: patch flat within the cup tolerance, not porous, vacuum * area * efficiency vs the same load. 27 cases (mass, friction, scale if uncalibrated); the suggestion says whether size or grip force fails. CLI: python -m lab.gripper <run> [--gripper so100_jaw panda_hand robotiq_2f85 suction_30mm my.yaml] -> <run>/lab/gripper_<id>.json. Labs add their gripper as a yaml.

Scale: measure/scale.py (load, calibrate from one known size, parse "15 cm tall").
Tests: tests/test_transport.py (slide at mu, tip at half-width/COM height, no diagonal weakness).
Status: transport works on table1 (can tips at 0.22 g sideways; bottle 0.26 g; roll slides 0.41 g on wood).
Next: grasp test across property ranges, gripper fit (jaws, suction), camera test, viewer Lab tab + report export.
