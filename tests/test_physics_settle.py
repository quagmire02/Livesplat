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
