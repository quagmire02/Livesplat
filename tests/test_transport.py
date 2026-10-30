"""Transport limits match textbook statics: slide at a/g = mu, tip at a/g = half-width / COM height."""
import numpy as np

from lab.transport import limit

HULL = np.array([[x, y, z] for x in (-0.03, 0.03) for y in (-0.03, 0.03) for z in (0., 0.12)], np.float32)
POLY = np.array([[-.5, -.5], [.5, -.5], [.5, .5], [-.5, .5]])


def _obj():
    return {"id": 1, "base": np.zeros(3), "hull": HULL.copy(), "mass": 0.2, "friction": 0.5, "bounce": 0.0}


def test_low_friction_slides_at_mu():
    g, what = limit(_obj(), POLY, -0.75, 0.5, 0.0, mu=0.3)
    assert what == "slide" and abs(g - 0.3) < 0.06, (g, what)


def test_diagonal_slides_at_mu_too():
    g, what = limit(_obj(), POLY, -0.75, 0.5, 45.0, mu=0.3)     # elliptic cone: no direction is weaker
    assert what == "slide" and abs(g - 0.3) < 0.06, (g, what)


def test_high_friction_tips_at_width_over_height():
    g, what = limit(_obj(), POLY, -0.75, 0.5, 0.0, mu=0.9)
    assert what == "tip" and abs(g - 0.5) < 0.1, (g, what)       # 0.03 / 0.06; a short pulse tips a bit later
