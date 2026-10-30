"""Typed questions reach the right modes without a model call; commands fall through to the planner."""
import pytest

from viewer.route import find_object, route

CASES = [
    ("what is this?", True, ["explain"]),
    ("What's this thing", True, ["explain"]),
    ("how do I use this", True, ["explain"]),
    ("how does it work?", True, ["explain"]),
    ("tell me more", True, ["explain"]),
    ("where can I buy this?", True, ["similar"]),
    ("find this online", True, ["similar"]),
    ("show me pictures of it", True, ["similar"]),
    ("anything similar but cheaper?", True, ["similar"]),
    ("other brands?", True, ["similar"]),
    ("what is that lotion bottle and how do i use it and find me similar things", True, ["explain", "similar"]),
    ("how do I make a sandwich", False, ["guide"]),
    ("how do I make a sandwich", True, ["guide"]),
    ("how to get ready after a shower", False, ["guide"]),
    ("walk me through cleaning the table", False, ["guide"]),
    ("what can I make with these", False, ["guide"]),
    ("make a 3D model of this", True, ["model"]),
    ("download the model", True, ["model"]),
    ("can I get a 3d version of the lotion bottle", True, ["model"]),
    ("use the 3d model look", True, ["look"]),
    ("switch to the model instead", True, ["look"]),
    ("back to the scanned look", True, ["look"]),
    ("push the can left", True, []),
    ("slide the bottle next to the can", False, []),
    ("make it red", True, []),
    ("moon gravity", False, []),
    ("reset", False, []),
    ("what is this?", False, []),            # nothing selected or named: let the planner ask which object
]


@pytest.mark.parametrize("text,has_object,want", CASES)
def test_route(text, has_object, want):
    assert route(text, has_object) == want


NAMES = {1: ["cetaphil moisturizing lotion pump bottle", "lotion bottle", "pump bottle"],
         2: ["gillette aerosol deodorant spray", "deodorant", "spray can", "shield"],
         3: ["toilet paper roll", "toilet roll"]}


@pytest.mark.parametrize("text,want", [
    ("what is that lotion bottle and how do i use it", 1),
    ("find me something like the Gillette deodorant", 2),
    ("tell me about the toilet roll", 3),
    ("what is this?", None),
])
def test_find_object(text, want):
    assert find_object(text, NAMES) == want
