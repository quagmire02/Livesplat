"""Camera test conventions: OpenCV look-at, principal point, stereo depth-noise model."""
import numpy as np

from lab.camera import intrinsics, load_camera, look_at
from scene_io.thrift import project


def test_look_at_principal_point_and_up():
    cam = load_camera("realsense_d435")
    K, W, H, f = intrinsics(cam)
    c2w = look_at(np.array([0.0, -1.0, 0.0]), np.zeros(3))
    uv, z = project(np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.1], [0.1, 0.0, 0.0]]), K, c2w)
    assert np.allclose(uv[0], [W / 2, H / 2]) and np.isclose(z[0], 1.0)
    assert uv[1][1] < uv[0][1]                       # world up = image up (OpenCV y down)
    assert uv[2][0] > uv[0][0]                       # world +x is image right when looking along +y


def test_d435_focal_and_noise():
    cam = load_camera("realsense_d435")
    K, W, H, f = intrinsics(cam)
    assert abs(f - 640 / np.tan(np.radians(43.5))) < 1e-6
    sigma_1m = 1.0 ** 2 * cam["subpixel_px"] / (f * cam["baseline_m"])
    assert 0.001 < sigma_1m < 0.004                  # ~2.4 mm at 1 m: stereo error grows with z^2
