# lab/ — Lab bench: tests for robotics labs on their scanned objects

Purpose: answer lab questions with many simulated trials across what we don't know exactly (friction, mass,
scale), plus a suggestion. Not only arms: transport (mobile robots, trays), gripper fit, grasp, camera.

Files:
- objects.py — lab_objects(sc, run): deep copies of the scan's objects; the full 3D model's convex hull (gen/model_<id>/model.glb, flat base, must stand) replaces the scan hull when available.
- transport.py — how hard a mobile robot / tray can accelerate or brake before each object slides or tips. Bisection on a/g per object, 3 directions, friction x0.7/1.0/1.3, surface wood/plastic/rubber. Elliptic friction cone (MuJoCo's default pyramid is 30% weaker diagonally). Scale-free (a ratio). CLI: python -m lab.transport <run> [--surface rubber] -> <run>/lab/transport_<surface>.json.

- gripper.py + grippers/*.yaml — gripper fit without an arm. Parallel jaws: narrowest grip per height slice (3D model shape), fits opening - 5 mm, holds if 2 mu F >= m (g + 0.5 g) 1.5. Suction: patch flat within the cup tolerance, not porous, vacuum * area * efficiency vs the same load. 27 cases (mass, friction, scale if uncalibrated); the suggestion says whether size or grip force fails. CLI: python -m lab.gripper <run> [--gripper so100_jaw panda_hand robotiq_2f85 suction_30mm my.yaml] -> <run>/lab/gripper_<id>.json. Labs add their gripper as a yaml.

- grasp.py — arm grasp test on sim/ (export -> robot -> grasp): metric package per scale (lab/sim_<mm per unit>), base auto-placed once and fixed in metres, top + side grasps over mass x0.7/1/1.3 x grip friction x0.7/1/1.3 (robot and object geoms set to sqrt(pad * object): MuJoCo takes the max), and scale x(1 -+ spread) unless calibrated. Arms are never scaled. CLI: python -m lab.grasp <run> --robot so100 panda vx300s wx250s xarm7 [--quick] -> <run>/lab/grasp_<robot>.json.
- camera.py + cameras/*.yaml — sensor test: visible share of each object's camera-facing surface (z-buffer vs all objects + room), pixels across, distance in range, depth noise (stereo z^2 sigma_d / (f b); ToF a + b z). Auto placement over 108 poses; synthetic noisy depth + id mask for datasets. Cameras: D435, D455, ZED 2i, Femto Bolt, C920. CLI: python -m lab.camera <run> [--camera ...]. Tests: tests/test_camera.py.
- Viewer: viewer/lab_panel.py (More tools > Lab): scale + calibrate, arm/gripper/surface/camera dropdowns, run tests, result card, HTML report download; camera frustum with the browser-rendered view; "export dataset" = 12 views (RGB render, noisy depth uint16 mm, id mask, K, OpenCV c2w in metres) as a zip.

Scale: measure/scale.py (load, calibrate from one known size, parse "15 cm tall").
Tests: tests/test_transport.py (slide at mu, tip at half-width/COM height, no diagonal weakness).
Status: transport works on table1 (can tips at 0.22 g sideways; bottle 0.26 g; roll slides 0.41 g on wood).
Next: grasp test across property ranges, gripper fit (jaws, suction), camera test, viewer Lab tab + report export.
