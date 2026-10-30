"""Lesson plans are validated before drawing: unknown ids/types dropped, points clamped; view-relative axes."""
import numpy as np
import pytest

from viewer.lesson import _axes, _curve, validate


def test_validate_drops_bad_drawings():
    raw = {"title": "Tennis", "steps": [
        {"say": "Hold the racket by the grip.", "focus": [1, 9],
         "draw": [{"type": "highlight", "obj": 1}, {"type": "highlight", "obj": 9},          # 9 does not exist
                  {"type": "script", "code": "alert(1)"},                                   # not in the vocabulary
                  {"type": "move", "obj": 2, "path": [{"obj": 2, "at": "top"}, {"table": [10, 50], "height_cm": 80}],
                   "turn_deg": 9999},
                  {"type": "arrow", "from": {"obj": 1, "at": "nowhere"}, "to": {"table": [0, 0]}}]},
        {"say": "", "draw": []}]}
    v = validate(raw, {1, 2})
    assert len(v["steps"]) == 1
    s = v["steps"][0]
    assert s["focus"] == [1]
    assert [d["type"] for d in s["draw"]] == ["highlight", "move", "arrow"]
    assert s["draw"][1]["turn_deg"] == 360                                       # clamped
    assert s["draw"][2]["from"]["at"] == "center"                                # unknown anchor -> centre


def test_validate_rejects_empty():
    with pytest.raises(RuntimeError):
        validate({"steps": [{"say": "  "}]}, {1})


def test_salvage_keeps_good_steps():
    from viewer.lesson import salvage
    broken = ('{"title": "How to Put Lotion", "steps": [{"say": "Start with the bottle.", "focus": [1], "draw": '
              '[{"type": "highlight", "obj": 1}]}, {"say": "Dab extra.", "draw": [{"type": "label", "obj": 3, '
              '"at": "top"} widespread: null}] , "missing": []}')
    j = salvage(broken)
    assert j["title"] == "How to Put Lotion" and len(j["steps"]) == 1 and j["steps"][0]["say"] == "Start with the bottle."


def test_axes_and_curve():
    r, v = _axes(np.array([0.0, -2.0, 1.0]), np.zeros(3))          # looking along +y
    assert np.allclose(r, [1, 0, 0]) and np.allclose(v, [0, -1, 0])   # right = +x, toward you = -y
    c = _curve(np.zeros(3), np.array([1.0, 0, 0]))
    assert np.allclose(c[0], 0) and np.allclose(c[-1], [1, 0, 0]) and c[:, 2].max() > 0.1   # arcs upward
