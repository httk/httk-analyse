"""Analytic ensemble fluctuations and block uncertainty checks."""

import numpy as np
import pytest

from httk.analyse.matsci.thermodynamics import equilibrium_response

KB = 1.380649e-23 / 1.602176634e-19


def test_nvt_uses_total_energy_variance_and_explicit_temperature():
    result = equilibrium_response(temperature=300, ensemble='NVT', energies=[-3, -1, -3, -1])
    assert result.names == ('heat_capacity_cv',)
    assert result.values == pytest.approx([1 / (KB * 300**2)])
    assert result.standard_errors is None


def test_npt_covariance_sign_units_and_blocks():
    h = np.array([1, 3, 2, 4, 1, 5, 2, 6.0])
    v = 10 - 0.1 * h
    result = equilibrium_response(temperature=100, ensemble='NPT', enthalpies=h, volumes=v, block_size=4)
    expected = [
        np.var(h) / (KB * 100**2),
        np.var(v) / (KB * 100 * np.mean(v)),
        np.mean((v - v.mean()) * (h - h.mean())) / (KB * 100**2 * np.mean(v)),
    ]
    assert result.values == pytest.approx(expected)
    assert result.values[2] < 0
    block = np.array(result.block_values)
    np.testing.assert_allclose(result.standard_errors, np.std(block, axis=0, ddof=1) / np.sqrt(2))


def test_explicit_tail_drop_is_reported_and_excluded():
    values = [1, 3, 1, 3, 1e10]
    with pytest.raises(ValueError):
        equilibrium_response(temperature=100, ensemble='NVT', energies=values, block_size=2)
    result = equilibrium_response(temperature=100, ensemble='NVT', energies=values, block_size=2, remainder='drop')
    assert result.used_samples == 4 and result.dropped_samples == 1
    assert result.values[0] == pytest.approx(1 / (KB * 100**2))
    assert result.standard_errors == (0.0,)


@pytest.mark.parametrize(
    'arguments',
    [
        dict(ensemble='NVE', energies=[1, 2]),
        dict(ensemble='NVT', enthalpies=[1, 2]),
        dict(ensemble='NPT', energies=[1, 2], volumes=[1, 2]),
        dict(ensemble='NPT', enthalpies=[1, 2], volumes=[1, -1]),
        dict(ensemble='NVT', energies=[1, 2], block_size=1),
        dict(ensemble='NVT', energies=[1 + 2j, 2]),
    ],
)
def test_invalid_ensemble_contract(arguments):
    with pytest.raises(ValueError):
        equilibrium_response(temperature=300, **arguments)
