# lab/ — Lab bench: tests for robotics labs on their scanned objects

Purpose: answer lab questions with many simulated trials across what we don't know exactly (friction, mass,
scale), plus a suggestion. Not only arms: transport (mobile robots, trays), gripper fit, grasp, camera.

Files:
- objects.py — lab_objects(sc, run): deep copies of the scan's objects; the full 3D model's convex hull (gen/model_<id>/model.glb, flat base, must stand) replaces the scan hull when available.
- transport.py — how hard a mobile robot / tray can accelerate or brake before each object slides or tips. Bisection on a/g per object, 3 directions, friction x0.7/1.0/1.3, surface wood/plastic/rubber. Elliptic friction cone (MuJoCo's default pyramid is 30% weaker diagonally). Scale-free (a ratio). CLI: python -m lab.transport <run> [--surface rubber] -> <run>/lab/transport_<surface>.json.

Scale: measure/scale.py (load, calibrate from one known size, parse "15 cm tall").
Tests: tests/test_transport.py (slide at mu, tip at half-width/COM height, no diagonal weakness).
Status: transport works on table1 (can tips at 0.22 g sideways; bottle 0.26 g; roll slides 0.41 g on wood).
Next: grasp test across property ranges, gripper fit (jaws, suction), camera test, viewer Lab tab + report export.
