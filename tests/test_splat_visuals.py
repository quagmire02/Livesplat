"""Focused offline checks for contact ownership and learned appearance."""
import numpy as np

from viewer.contact import split_contact, at_capture
from scene_io.appearance import view_rgb


def test_contact_keeps_vertical_base_but_anchors_table():
    a = {"means": np.array([[0, 0, .005], [0, 0, .005], [0, 0, -.001], [0, 0, .15]]),
         "scales": np.array([[.02, .02, .001], [.001, .02, .02], [.01]*3, [.01]*3]),
         "opacity": np.ones(4), "quats": np.tile([1., 0, 0, 0], (4, 1))}
    body, contact = split_contact(a, 1.)
    np.testing.assert_array_equal(body, [False, True, False, True])
    assert np.all(body ^ contact)
    assert at_capture((np.zeros(3), np.array([1., 0, 0, 0])), np.zeros(3), 1.)
    assert not at_capture((np.zeros(3), np.array([0., 0, 0, 1.])), np.zeros(3), 1.)
    assert not at_capture((np.ones(3), np.array([1., 0, 0, 0])), np.zeros(3), 1.)


def test_sh_matches_gsplat_reference():
    import torch
    from gsplat.cuda._torch_impl import _spherical_harmonics
    rng = np.random.default_rng(7)
    dirs = rng.normal(size=(20, 3)).astype(np.float32)
    dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
    coeff = rng.normal(0, .1, (20, 16, 3)).astype(np.float32)
    expected = _spherical_harmonics(3, torch.from_numpy(dirs), torch.from_numpy(coeff)).numpy()
    actual = view_rgb(coeff[:, 0], coeff[:, 1:].transpose(0, 2, 1).reshape(20, 45), dirs)
    np.testing.assert_allclose(actual, np.clip(expected + .5, 0, 1), atol=2e-7)
