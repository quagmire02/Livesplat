"""Small, offline checks of repair depth and undo/file round trips."""
from types import SimpleNamespace
import threading
import numpy as np
from viewer.repair import brush_indices, RepairPanel


def test_brush_excludes_back_surface_and_off_ray():
    p = np.array([[0, 0, 1], [.01, 0, 1.01], [0, 0, 2], [.5, 0, 1], [0, 0, -1]])
    np.testing.assert_array_equal(brush_indices(p, np.zeros(3), [0, 0, 1], .04), [0, 1])


def test_repair_history_and_save(tmp_path):
    panel = RepairPanel.__new__(RepairPanel)
    key = (1, 'splat')
    panel.layers = {key: {'state': np.array([-1, 0], np.int8), 'fingerprint': 'same'}}
    panel.undo = [(key, np.array([0]), np.array([0], np.int8), np.array([-1], np.int8))]
    panel.redo = []
    panel.lock = threading.RLock()
    panel.info = SimpleNamespace(content='')
    panel.clear = lambda: None
    panel.rebuild = lambda key: None
    panel.path = tmp_path / 'visual_repairs.npz'
    panel.history('undo')
    assert panel.layers[key]['state'][0] == 0
    panel.history('redo')
    assert panel.layers[key]['state'][0] == -1
    panel.disk('save repairs')
    panel.layers[key]['state'][:] = 0
    panel.disk('load repairs')
    assert panel.layers[key]['state'][0] == -1
    panel.layers[key]['fingerprint'] = 'different'
    panel.layers[key]['state'][:] = 0
    panel.disk('load repairs')
    assert 'different geometry' in panel.info.content
    assert not panel.layers[key]['state'].any()
