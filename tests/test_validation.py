"""Analytic validation diagnostics on already paired model results."""

import numpy as np
import pytest

from httk.analyse.matsci.validation import committee_spread, energy_drift, force_energy_consistency, property_parity

IRI = 'https://schemas.httk.org/defs/v0.1/properties/mechanics/bulk_modulus'


def test_parity_labels_raw_errors_and_units():
    result = property_parity([1, 2], [2, 0], labels=['A', 'B'], definition=IRI)
    assert result.residuals == (1, -2)
    assert result.statistics.rmse == pytest.approx(np.sqrt(2.5))
    for labels, definition in [('AB', IRI), (['A'], IRI), (['A', 'B'], '')]:
        with pytest.raises(ValueError):
            property_parity([1, 2], [2, 0], labels=labels, definition=definition)


def test_nve_linear_drift_nonuniform_times():
    time = np.array([1000.0, 1000.1, 1001, 1004])
    result = energy_drift(time, (-3 + 0.01 * (time - 1000)) * 20, atom_count=20, ensemble='NVE')
    assert result.slope == pytest.approx(0.01)
    assert result.intercept == pytest.approx(-3)
    assert result.endpoint_change == pytest.approx(0.04)
    assert result.residual_rms < 1e-14
    with pytest.raises(ValueError):
        energy_drift(time, np.ones(4), atom_count=20, ensemble='NVT')


def test_finite_difference_force_sign_and_step_convergence():
    xyz = np.array([[0.2, -0.4, 0.7]])
    saved = xyz.copy()
    energy = lambda positions: float(np.sum(positions**4))
    force = -4 * xyz**3
    coarse = force_energy_consistency(energy, xyz, force, displacement=0.01)
    fine = force_energy_consistency(energy, xyz, force, displacement=0.005)
    assert coarse.statistics.rmse / fine.statistics.rmse == pytest.approx(4, rel=1e-7)
    assert fine.definition is None
    np.testing.assert_array_equal(xyz, saved)
    wrong = force_energy_consistency(energy, xyz, -force, displacement=0.005)
    assert wrong.statistics.rmse > 1


def test_committee_mean_sample_variance_not_sem():
    result = committee_spread([[1, 2], [3, 4]], definition=IRI, ddof=1)
    assert result.mean == (2, 3)
    assert result.standard_deviation == pytest.approx([np.sqrt(2), np.sqrt(2)])
    with pytest.raises(ValueError):
        committee_spread([[1]], definition=IRI)


def test_finite_difference_uses_representable_coordinate_spacing():
    xyz = np.array([[100000.0, 0, 0]])
    result = force_energy_consistency(lambda pos: float(pos[0, 0] - 100000), xyz, [[-1, 0, 0]], displacement=1e-10)
    assert result.statistics.maximum_absolute_error == 0
