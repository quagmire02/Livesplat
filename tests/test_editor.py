"""Editor round trips and predictions on tiny synthetic geometry; no scan or API needed."""
import copy
import json
from types import SimpleNamespace

import numpy as np
import pytest

from agent.controller import Controller
from agent import parser
from agent.schema import validate
from physics.world import World
from viewer.session import EditorSession


@pytest.fixture
def editor(tmp_path):
    hull = np.array([[x, y, z] for x in (-0.03, 0.03) for y in (-0.03, 0.03) for z in (0., 0.12)], np.float32)
    objs = []
    for k in (1, 2):
        part = {"means": hull.copy(), "scales": np.full((8, 3), 0.005, np.float32),
                "quats": np.tile([1., 0, 0, 0], (8, 1)), "rgb": np.full((8, 3), 0.5, np.float32),
                "opacity": np.ones(8, np.float32)}
        objs.append({"id": k, "base": np.array([0.15 * (k - 1), 0, 0]), "hull": hull.copy(),
                     "splat": part, "refl": None, "volume": 0.06 * 0.06 * 0.12,
                     "mass": 0.1, "friction": 0.5, "bounce": 0.1})
    sc = SimpleNamespace(objects=objs, polygon=np.array([[-.5, -.5], [.5, -.5], [.5, .5], [-.5, .5]]), radius=.5)
    w = World(objs, sc.polygon)
    ctrl = Controller(sc, w, tmp_path)
    return ctrl, EditorSession(ctrl, tmp_path)


def test_layout_roundtrip_and_history(editor):
    ctrl, session = editor
    initial = session.snapshot()
    session.checkpoint("edits")
    session.rename(1, "my bottle")
    ctrl.t_set_color("1", "red")
    ctrl.t_scale("1", 1.5)
    ctrl.t_set_mass("1", 0.7)
    ctrl.w.place(1, [0.1, 0.2, 0.3], [1, 0, 0, 0])
    ctrl.w.throw(1, [0.1, 0.2, 0.3])
    ctrl.t_remove("2")
    ctrl.t_set_gravity(1.62)
    changed = session.snapshot()
    session.save()
    assert session.path.stat().st_size < 5000
    assert session.undo() == "Undid edits"
    assert session.snapshot() == initial
    assert session.redo() == "Redid edits"
    assert session.snapshot() == changed
    ctrl.t_reset()
    session.load()
    assert session.snapshot() == changed
    assert np.allclose(ctrl.obj(1)["hull"], session.baselines[1]["hull"] * 1.5)
    assert ctrl.obj(1)["color"] == "red" and not ctrl.w.present(2)


def test_invalid_layout_does_not_mutate(editor):
    ctrl, session = editor
    before = session.snapshot()
    invalid = copy.deepcopy(before)
    invalid["objects"][0]["mass"] = 1.0
    invalid["objects"][1]["pose"] = [0] * 7
    with pytest.raises(ValueError):
        session.restore(invalid)
    assert session.snapshot() == before
    session.save()
    saved = json.loads(session.path.read_text())
    saved["geometry"] = "different scan"
    session.path.write_text(json.dumps(saved))
    with pytest.raises(ValueError, match="reconstruction changed"):
        session.load()
    assert session.snapshot() == before


def test_prediction_preserves_full_live_data_on_success_and_failure(editor, monkeypatch):
    ctrl, _ = editor
    data = ctrl.w.d
    data.time = 12.3
    data.qacc_warmstart[:] = 0.2
    data.xfrc_applied[:] = 0.3
    before = {k: getattr(data, k).copy() for k in ("qpos", "qvel", "qacc_warmstart", "xfrc_applied")}
    result = ctrl.t_simulate([{"name": "push", "args": {"obj": "1", "direction": "left"}}])
    assert result.startswith("prediction")
    assert ctrl.w.d is data and data.time == 12.3
    for k, v in before.items():
        assert np.array_equal(getattr(data, k), v)
    def fail(seconds):
        raise RuntimeError("simulation failed")
    monkeypatch.setattr(ctrl.w, "step", fail)
    with pytest.raises(RuntimeError):
        ctrl.t_simulate([{"name": "drop", "args": {"obj": "1"}}])
    assert ctrl.w.d is data
    for k, v in before.items():
        assert np.array_equal(getattr(data, k), v)


def test_yaw_is_not_tilt(editor):
    ctrl, _ = editor
    ctrl.w.place(1, [0, 0, 0], [0, 0, 0, 1])
    assert ctrl._tilt(1) == pytest.approx(0)
    ctrl.w.place(1, [0, 0, 0], [np.sqrt(0.5), np.sqrt(0.5), 0, 0])
    assert ctrl._tilt(1) == pytest.approx(90)


def test_inventory_and_ambiguity(editor):
    ctrl, session = editor
    session.rename(1, "=formula")
    assert "'=formula" in session.inventory_csv()
    assert "width_scene_units" in session.inventory_csv()
    with pytest.raises(KeyError, match="several objects"):
        ctrl.resolve("gray")


@pytest.mark.parametrize("bad", [None, [], {"name": []}, {"name": "drop", "args": None},
    {"name": "drop", "args": {"obj": [], "height": 1}},
    {"name": "drop", "args": {"obj": "1", "height": True}},
    {"name": "drop", "args": {"obj": "1", "height": float("nan")}},
    {"name": "simulate", "args": {"actions": "oops"}},
    {"name": "simulate", "args": {"actions": [None]}},
    {"name": "simulate", "args": {"actions": []}}])
def test_malformed_tools_return_errors(bad):
    assert validate(bad)


def test_object_id_is_not_quantity():
    names = {2: ["object 2", "2", "can"]}
    for command, height in [("drop object 2 from 50 cm", 0.5), ("drop object 2", None), ("drop object 2 from 0 cm", 0)]:
        calls, unknown = parser.parse(command, names)
        assert not unknown
        assert calls[0]["args"].get("height") == height
    calls, _ = parser.parse("make the can 0 times smaller", names)
    assert validate(calls[0])
