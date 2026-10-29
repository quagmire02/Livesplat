"""SAM isolation + scan diagnosis from cached SAM 3.1 responses only (no paid calls). Needs LIVESPLAT_TEST_SCENE."""
import json
import os

import pytest

RUN = os.environ.get("LIVESPLAT_TEST_SCENE")


@pytest.mark.skipif(not RUN, reason="LIVESPLAT_TEST_SCENE not set")
def test_isolate_cached(monkeypatch):
    monkeypatch.delenv("LIVESPLAT_ALLOW_META", raising=False)
    from scene_io.thrift import load_cameras
    from viewer.app import Scene
    from viewer.sam_isolate import isolate
    sc = Scene(RUN)
    props = json.load(open(os.path.join(RUN, "props.json")))
    o = sc.objects[0]
    p = o.get("capture_means", o["splat"]["means"]) + o["base"]
    world = (p - sc.T[:3, 3]) @ sc.T[:3, :3]
    rep = isolate(world, props[str(o["id"])]["sam_prompt"], load_cameras(os.path.join(RUN, "thrift_sfm")), cache_only=True)
    assert len(rep["views"]) >= 2
    assert rep["coverage_min"] > 0.8
    assert len(rep["spill_ids"]) < 0.2 * len(world)          # suggested fix stays a small correction
