"""Hand-guided slide: stops just short of the target, stays upright, and simulate does not leak slides."""
from types import SimpleNamespace

import numpy as np

from agent.controller import Controller
from physics.world import World


def _scene(tmp_path):
    hull = np.array([[x, y, z] for x in (-0.03, 0.03) for y in (-0.03, 0.03) for z in (0., 0.12)], np.float32)
    objs = [{"id": k, "base": np.array([0.15 * (k - 1), 0, 0]), "hull": hull.copy(),
             "splat": {"means": hull.copy(), "rgb": np.full((8, 3), 0.5, np.float32)}, "volume": 0.06 * 0.06 * 0.12,
             "mass": 0.1, "friction": 0.8, "bounce": 0.1} for k in (1, 2)]
    sc = SimpleNamespace(objects=objs, polygon=np.array([[-.5, -.5], [.5, -.5], [.5, .5], [-.5, .5]]), radius=.5)
    return Controller(sc, World(objs, sc.polygon), tmp_path)


def test_slide_to_target(tmp_path):
    ctrl = _scene(tmp_path)
    ctrl.w.step(0.5)
    res = ctrl.run({"name": "slide", "args": {"obj": "1", "direction": "toward", "target": "2", "gap": 0.01}})
    assert not res.startswith("error"), res
    ctrl.w.step(2.0)
    gap = ctrl._world_hull(2)[:, 0].min() - ctrl._world_hull(1)[:, 0].max()
    assert 0.0 < gap < 0.02, gap
    assert ctrl._tilt(1) < 5 and ctrl._tilt(2) < 5          # friction 0.8 would tip a flicked box this tall
    assert not ctrl.w.slides


def test_simulate_keeps_live_slides(tmp_path):
    ctrl = _scene(tmp_path)
    p0 = ctrl.w.pose(1)[0]
    out = ctrl.t_simulate([{"name": "slide", "args": {"obj": "1", "direction": "toward", "target": "2"}}], 2.0)
    assert "moved" in out
    assert not ctrl.w.slides and np.allclose(ctrl.w.pose(1)[0], p0)
