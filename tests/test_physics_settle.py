"""Objects placed at their segmented poses must settle without jumping (needs LIVESPLAT_TEST_SCENE run dir)."""
import os

import numpy as np
import pytest

RUN = os.environ.get("LIVESPLAT_TEST_SCENE")


@pytest.mark.skipif(not RUN, reason="LIVESPLAT_TEST_SCENE not set")
def test_settle():
    from physics.world import World
    from viewer.app import Scene
    sc = Scene(RUN)
    w = World(sc.objects, sc.polygon, floor_z=sc.floor_z)
    p0 = {o["id"]: w.pose(o["id"])[0] for o in sc.objects}
    w.step(2.0)
    for o in sc.objects:
        p, q = w.pose(o["id"])
        moved = np.linalg.norm(p - p0[o["id"]])
        tilt = np.degrees(2 * np.arccos(min(1.0, abs(q[0]))))
        print(f"obj {o['id']}: moved {moved * 100:.2f} cm, tilt {tilt:.1f} deg, speed {w.speed(o['id']):.3f}")
        assert moved < 0.02 and tilt < 5


@pytest.mark.skipif(not RUN, reason="LIVESPLAT_TEST_SCENE not set")
def test_pick_and_full_reset():
    from agent.controller import Controller
    from physics.pick import pick
    from physics.world import World
    from viewer.app import Scene
    sc = Scene(RUN)
    w = World(sc.objects, sc.polygon, floor_z=sc.floor_z)
    for o in sc.objects:                      # a ray straight down onto each object picks it
        p = w.pose(o["id"])[0]
        assert pick(p + [0, 0, 2.0], [0, 0, -1], sc.objects, w) == o["id"]
    assert pick([0, 0, 2.0], [1, 0, 0], sc.objects, w) is None
    ctrl = Controller(sc, w, RUN)
    ctrl.t_set_color("1", "red"), ctrl.t_scale("2", 2.0), ctrl.t_remove("3"), ctrl.t_set_gravity(1.62)
    ctrl.t_reset()
    o1, o2 = ctrl.obj(1), ctrl.obj(2)
    assert np.array_equal(o1["splat"]["rgb"], o1["rgb0"]) and o2["scale"] == 1.0
    assert w.present(3) and w.gravity == 9.81
