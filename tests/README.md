# tests/
- test_cameras.py — Reprojection of ThriftSplat points (LIVESPLAT_TEST_RUN).
- test_physics_settle.py — Settle, pick, full reset (LIVESPLAT_TEST_SCENE).
- test_agent.py — Parser wording, chains, end-to-end commands.
- test_sim.py — Export (scale 0.64), self-test, Panda reach, can grasp (LIVESPLAT_TEST_SCENE).
- test_robots.py — Panda + SO-100 on the synthetic scene (no scan needed).
- fixtures/table1_names.json — Names for table1 without Spark.
Run: `LIVESPLAT_TEST_SCENE=~/livesplat_data/runs/table1 python -m pytest -q tests/`
