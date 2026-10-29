# tests/
- test_cameras.py — Reprojection of ThriftSplat points (LIVESPLAT_TEST_RUN).
- test_physics_settle.py — Settle, pick, full reset (LIVESPLAT_TEST_SCENE).
- test_agent.py — Parser wording, chains, end-to-end commands.
- test_sim.py — Export (scale 0.64), self-test, Panda reach, can grasp (LIVESPLAT_TEST_SCENE).
- test_robots.py — Panda + SO-100 on the synthetic scene (no scan needed).
- fixtures/table1_names.json — Names for table1 without Spark.
- test_editor.py — Tiny synthetic scene: layout/history round trips, prediction isolation, yaw/tilt, inventory, malformed tools and numeric object references. No scan or model needed.
- test_backend_offline.py — Mocked HTTP only: permission gate, offline/cache-only paths, nullable Muse tool messages, selection grounding and retry budget checks.
- test_splat_visuals.py — Contact ownership and captured-pose checks; SH colour calculation compared with gsplat's CPU reference.
- test_repair.py — Brush rejects rear surfaces; repair undo/redo and save/load round trip, mismatched geometry rejection.
- test_object_card.py — Structured model response validation and distinct source-frame crop selection, fully offline.
Run: `LIVESPLAT_TEST_SCENE=~/livesplat_data/runs/table1 python -m pytest -q tests/`

- [CHANGE_NOTES.md](CHANGE_NOTES.md) — Short implementation summary and remaining gaps (Oct 29).
