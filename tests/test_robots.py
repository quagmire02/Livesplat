"""Bring-your-own-robot: both robot yamls on the synthetic known-answer scene (no scan needed)."""
import json

import numpy as np
import pytest
import trimesh


@pytest.mark.parametrize("robot,obj,mode", [("panda", "tall box", "top"), ("panda", "tall box", "side"),
                                            ("so100", "small cube", "top")])
def test_grasp_synthetic(tmp_path, robot, obj, mode):
    from sim.grasp import Arm, run_grasp
    from sim.robot import build, choose_base
    from sim.robot_spec import RobotSpec
    from sim.synthetic import make
    sim = make(tmp_path / "syn")
    spec = RobotSpec.load(robot)
    meta = json.loads((sim / "metadata.json").read_text())
    hulls = {o["body"]: np.asarray(trimesh.load(sim / o["files"]["collision"]).vertices) for o in meta["objects"]}
    _, base, yaw, rows = choose_base(sim, spec, meta, hulls, 1.0)
    arm = Arm(build(sim, spec, base, yaw), spec)
    row = next(r for r in rows if r["object"] == obj)
    o = next(x for x in meta["objects"] if x["body"] == row["body"])
    res = run_grasp(arm, row, o, 1.0, mode)
    assert res["success"], res
