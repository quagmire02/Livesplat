"""Real-to-sim end to end on a scanned run (LIVESPLAT_TEST_SCENE): export, self-test, Panda reach, can grasp."""
import json
import os

import numpy as np
import pytest

RUN = os.environ.get("LIVESPLAT_TEST_SCENE")


@pytest.mark.skipif(not RUN, reason="LIVESPLAT_TEST_SCENE not set")
def test_export_robot_grasp(tmp_path):
    import trimesh

    from sim.export import export, self_test
    from sim.grasp import Arm, run_grasp
    from sim.robot import build, choose_base
    from sim.robot_spec import RobotSpec
    out = export(RUN, out=tmp_path / "sim", scale=0.64, scale_source="test")
    assert self_test(out)
    meta = json.loads((out / "metadata.json").read_text())
    hulls = {o["body"]: np.asarray(trimesh.load(out / o["files"]["collision"]).vertices) for o in meta["objects"]}
    spec = RobotSpec.load("panda")
    _, base, yaw, rows = choose_base(out, spec, meta, hulls, 1.0)
    assert all(r["reachable_top_down"] for r in rows)
    arm = Arm(build(out, spec, base, yaw), spec)
    can = next(r for r in rows if "can" in r["object"])
    o = next(x for x in meta["objects"] if x["body"] == can["body"])
    assert run_grasp(arm, can, o, 1.0, "top")["success"]
