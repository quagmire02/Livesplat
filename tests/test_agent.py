"""Agent end to end without the viewer: parser -> controller -> physics (LIVESPLAT_TEST_SCENE run dir)."""
import os

import pytest

from agent import parser

RUN = os.environ.get("LIVESPLAT_TEST_SCENE")
NAMES = {1: ["lotion bottle", "bottle"], 2: ["spray can", "can"], 3: ["toilet roll", "roll"]}


@pytest.mark.parametrize("text,first", [
    ("push the bottle left", "push"),
    ("gently nudge the can away", "push"),
    ("drop the roll from 50 cm", "drop"),
    ("stack the roll on the bottle", "stack"),
    ("make the bottle red", "set_color"),
    ("make the can twice as big", "scale"),
    ("moon gravity", "set_gravity"),
    ("make the roll slippery", "set_friction"),
    ("what if I push the can hard", "simulate"),
    ("remove the roll", "remove"),
    ("reset", "reset"),
])
def test_parse(text, first):
    calls, unknown = parser.parse(text, NAMES)
    assert calls and calls[0]["name"] == first, (calls, unknown)


def test_chain():
    calls, _ = parser.parse("push the can left then drop it", NAMES)
    assert [c["name"] for c in calls] == ["push", "drop"]
    assert calls[1]["args"]["obj"] == "2"


@pytest.mark.skipif(not RUN, reason="LIVESPLAT_TEST_SCENE not set")
def test_run_scene():
    from agent.controller import Controller
    from agent.planner import handle
    from physics.world import World
    from viewer.app import Scene
    sc = Scene(RUN)
    w = World(sc.objects, sc.polygon, floor_z=sc.floor_z)
    ctrl = Controller(sc, w, RUN)
    for text in ["list objects", "measure the bottle", "what if I push the can hard",
                 "stack the roll on the bottle", "make the bottle red", "make the can twice as big",
                 "remove the roll", "restore the roll", "moon gravity", "make it melt"]:
        res, note = handle(text, ctrl)
        print(f"> {text}\n  " + "\n  ".join(f"{c['name']}: {r}" for c, r in res) + (f"\n  ({note})" if note else ""))
        assert all(not r.startswith("error") for _, r in res), res
        w.step(1.0)


def test_it_defaults_to_selection():
    calls, unknown = parser.parse("make it red", NAMES, default=2)
    assert calls == [{"name": "set_color", "args": {"obj": "2", "color": "red"}}], (calls, unknown)


@pytest.mark.parametrize("text,first", [
    ("shove the botle to the left", "push"),          # synonym + typo
    ("knock the can over", "push"),
    ("let the roll fall from 40cm", "drop"),
    ("yeet the can", "throw"),
    ("get rid of the roll", "remove"),
    ("colour the can green", "set_color"),
    ("put the roll on top of the bottle", "stack"),
    ("enlarge the roll", "scale"),
    ("can you gently nudge the bottle", "push"),
    ("start over", "reset"),
    ("would the can fall if I push it hard", "simulate"),
])
def test_wording_variants(text, first):
    calls, unknown = parser.parse(text, NAMES)
    assert calls and calls[0]["name"] == first, (calls, unknown)


def test_all_and_and():
    calls, _ = parser.parse("drop everything", NAMES)
    assert [c["name"] for c in calls] == ["drop"] * 3
    calls, _ = parser.parse("push the can left and drop the roll", NAMES)
    assert [c["name"] for c in calls] == ["push", "drop"]
