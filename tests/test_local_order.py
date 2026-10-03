"""Crystal invariants and rotation checks for optional bond order."""

import itertools
import numpy as np
import pytest

from httk.analyse.matsci.local_order import bond_order

pytest.importorskip('scipy.special')


def test_fcc_q4_q6_and_rotational_invariance():
    basis = np.array([[0, 0, 0], [0, 0.5, 0.5], [0.5, 0, 0.5], [0.5, 0.5, 0]])
    positions = np.concatenate([basis + shift for shift in itertools.product(range(3), repeat=3)])
    cell = np.eye(3) * 3
    q4 = bond_order(positions, cell, 0.8, 4)
    q6 = bond_order(positions, cell, 0.8, 6)
    np.testing.assert_allclose(q4.local, np.sqrt(7 / 192), atol=1e-12)
    np.testing.assert_allclose(q6.local, np.sqrt(169 / 512), atol=1e-12)
    assert q4.global_order == pytest.approx(q4.local[0])
    rotation, _ = np.linalg.qr(np.random.default_rng(4).normal(size=(3, 3)))
    rotated = bond_order(positions @ rotation, cell @ rotation, 0.8, 6)
    np.testing.assert_allclose(rotated.local, q6.local, atol=1e-12)
    assert rotated.global_order == pytest.approx(q6.global_order)


def test_isolated_and_zero_degree():
    isolated = bond_order([[0, 0, 0]], np.eye(3) * 10, 1, 6)
    assert isolated.local == (None,) and isolated.global_order is None
    assert bond_order([[0, 0, 0], [0.5, 0, 0]], np.eye(3) * 10, 1, 0).local == pytest.approx((1, 1))
    with pytest.raises(ValueError, match='coincident'):
        bond_order([[0, 0, 0], [0, 0, 0]], np.eye(3) * 10, 1, 6)
