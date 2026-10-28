"""Real-to-sim end to end on a run (LIVESPLAT_TEST_SCENE): export, self-test, robot reach, can grasp."""
import json
import os
from pathlib import Path

import pytest

RUN = os.environ.get("LIVESPLAT_TEST_SCENE")


@pytest.mark.skipif(not RUN, reason="LIVESPLAT_TEST_SCENE not set")
def test_export_robot_grasp(tmp_path):
    import mujoco
    import numpy as np

    from sim.export import export, self_test
    from sim.grasp import try_grasp
    from sim.robot import build, reach_report
    out = export(RUN, out=tmp_path / "sim")
    assert self_test(out)
    meta = json.loads((out / "metadata.json").read_text())
    rep = json.loads((Path(RUN) / "sim" / "checks" / "robot_report.json").read_text())
    import trimesh
    hulls = {o["body"]: np.asarray(trimesh.load(out / o["files"]["collision"]).vertices) for o in meta["objects"]}
    m = build(out, rep["base"]["pos"], np.radians(rep["base"]["yaw_deg"]), 1.0)
    rows = reach_report(m, meta, hulls, 1.0)
    assert all(r["reachable_top_down"] for r in rows)
    can = next(r for r in rows if "can" in r["object"])
    o = next(x for x in meta["objects"] if x["body"] == can["body"])
    res = try_grasp(mujoco.MjModel.from_xml_path(str(out / "robot_scene.xml")), can["body"], can, o, 1.0)
    assert res["success"], res
