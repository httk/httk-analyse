"""Analytic ensemble fluctuations and block uncertainty checks."""

import numpy as np
import pytest

from httk.analyse._constants import GPA_PER_EV_PER_A3
from httk.analyse.matsci.thermodynamics import EquilibriumResponse, equilibrium_response
from httk.analyse.records import records

KB = 1.380649e-23 / 1.602176634e-19


def test_nvt_uses_total_energy_variance_and_explicit_temperature():
    result = equilibrium_response(temperature=300, ensemble='NVT', energies=[-3, -1, -3, -1])
    assert result.names == ('heat_capacity_constant_volume',)
    assert result.values == pytest.approx([1 / (KB * 300**2)])
    assert result.standard_errors is None


def test_npt_covariance_sign_units_and_blocks():
    h = np.array([1, 3, 2, 4, 1, 5, 2, 6.0])
    v = 10 - 0.1 * h
    result = equilibrium_response(temperature=100, ensemble='NPT', enthalpies=h, volumes=v, block_size=4)
    expected = [
        np.var(h) / (KB * 100**2),
        np.var(v) / (KB * 100 * np.mean(v)) / GPA_PER_EV_PER_A3,
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
        {'ensemble': 'NVE', 'energies': [1, 2]},
        {'ensemble': 'NVT', 'enthalpies': [1, 2]},
        {'ensemble': 'NPT', 'energies': [1, 2], 'volumes': [1, 2]},
        {'ensemble': 'NPT', 'enthalpies': [1, 2], 'volumes': [1, -1]},
        {'ensemble': 'NVT', 'energies': [1, 2], 'block_size': 1},
        {'ensemble': 'NVT', 'energies': [1 + 2j, 2]},
    ],
)
def test_invalid_ensemble_contract(arguments):
    with pytest.raises(ValueError):
        equilibrium_response(temperature=300, **arguments)


def test_whole_system_input_contract_scaling_and_numpy_block_size():
    n = 500
    rng = np.random.default_rng(3)
    h = 100.0 + rng.normal(size=400)
    v = 1000.0 + 0.5 * (h - 100.0) + rng.normal(size=400)
    total = equilibrium_response(temperature=300, ensemble='NPT', enthalpies=h, volumes=v)
    per_atom = equilibrium_response(temperature=300, ensemble='NPT', enthalpies=h / n, volumes=v / n)
    cp, kappa, alpha = (a / b for a, b in zip(total.values, per_atom.values))
    assert cp == pytest.approx(n**2) and kappa == pytest.approx(n) and alpha == pytest.approx(n)
    blocked = equilibrium_response(temperature=300, ensemble='NVT', energies=h, block_size=np.int64(100))
    assert blocked.used_samples == 400


def test_unknown_response_names_raise_on_binding():
    legacy = EquilibriumResponse('NVT', ('heat_capacity_cv',), (1.0,), 300.0, (), None, 4, 0)
    with pytest.raises(ValueError, match='heat_capacity_cv'):
        records(legacy)
