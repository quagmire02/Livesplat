"""3D look: wxyz quaternion product and the glb y-up -> scene z-up rotation."""
import numpy as np
from scipy.spatial.transform import Rotation

from viewer.app import Q_YUP_TO_ZUP, qmul


def _rot(q):                                  # wxyz -> scipy (xyzw)
    return Rotation.from_quat([q[1], q[2], q[3], q[0]])


def test_yup_to_zup():
    assert np.allclose(_rot(Q_YUP_TO_ZUP).apply([0, 1, 0]), [0, 0, 1])      # glb up is scene up
    assert np.allclose(_rot(Q_YUP_TO_ZUP).apply([0, 0, 1]), [0, -1, 0])


def test_qmul_matches_scipy():
    a = np.array([0.9, 0.1, -0.3, 0.2]); a /= np.linalg.norm(a)
    b = np.array([0.5, 0.5, 0.5, -0.5])
    want = (_rot(a) * _rot(b)).as_quat()       # xyzw: apply b, then a
    got = qmul(a, b)
    assert np.allclose([got[1], got[2], got[3], got[0]], want) or np.allclose([got[1], got[2], got[3], got[0]], -want)
    assert np.allclose(qmul([1, 0, 0, 0], Q_YUP_TO_ZUP), Q_YUP_TO_ZUP)
