"""Typed questions reach the right mode without a model call; commands fall through to the planner."""
import pytest

from viewer.route import route

CASES = [
    ("what is this?", True, "explain"),
    ("What's this thing", True, "explain"),
    ("how do I use this", True, "explain"),
    ("how does it work?", True, "explain"),
    ("tell me more", True, "explain"),
    ("where can I buy this?", True, "similar"),
    ("find this online", True, "similar"),
    ("show me pictures of it", True, "similar"),
    ("anything similar but cheaper?", True, "similar"),
    ("other brands?", True, "similar"),
    ("how do I make a sandwich", False, "guide"),
    ("how do I make a sandwich", True, "guide"),
    ("how to get ready after a shower", False, "guide"),
    ("walk me through cleaning the table", False, "guide"),
    ("what can I make with these", False, "guide"),
    ("push the can left", True, None),
    ("slide the bottle next to the can", False, None),
    ("make it red", True, None),
    ("moon gravity", False, None),
    ("reset", False, None),
    ("what is this?", False, None),            # nothing selected: let the planner ask which object
]


@pytest.mark.parametrize("text,selected,want", CASES)
def test_route(text, selected, want):
    assert route(text, selected) == want
