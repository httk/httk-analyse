"""Independent direct-sum and SI checks for Green–Kubo estimators."""

import numpy as np
import pytest

from httk.analyse.matsci.transport import replica_transport, thermal_conductivity, viscosity

KB = 1.380649e-23
EV = 1.602176634e-19


def test_heat_tensor_matches_direct_sum_in_si():
    current = np.random.default_rng(91).normal(size=(20, 3))
    original = current.copy()
    result = thermal_conductivity(current, 0.01, temperature=300, volume=125, max_lag=6)
    si = (current - current.mean(axis=0)) * EV * 1e-10 / 1e-12
    correlation = np.array(
        [
            [sum(si[t, i] * si[t + lag, j] for t in range(20 - lag)) / (20 - lag) for i in range(3) for j in range(3)]
            for lag in range(7)
        ]
    )
    integral = np.zeros_like(correlation)
    integral[1:] = np.cumsum((correlation[:-1] + correlation[1:]) / 2 * 0.01e-12, axis=0) / (KB * 300**2 * 125e-30)
    np.testing.assert_allclose(result.integrals, integral, rtol=1e-13, atol=1e-15)
    np.testing.assert_array_equal(current, original)
    assert result.counts == tuple(range(20, 13, -1))
    np.testing.assert_allclose(result.isotropic, integral[:, [0, 4, 8]].mean(axis=1))


def test_constant_shear_si_prefactor_and_centering():
    stress = np.tile([[0, 2, 3], [2, 0, 4], [3, 4, 0]], (5, 1, 1))
    result = viscosity(stress, 0.1, temperature=100, volume=10, max_lag=3, remove_mean=False)
    expected = np.outer(np.arange(4) * 0.1e-12, (np.array([2, 3, 4]) * EV / 1e-30) ** 2) * 10e-30 / (KB * 100)
    np.testing.assert_allclose(result.integrals, expected)
    assert viscosity(stress, 0.1, temperature=100, volume=10, max_lag=3).isotropic == (0,) * 4


def test_replica_error_and_protocol_mismatch():
    a = thermal_conductivity(np.ones((5, 3)), 1, temperature=300, volume=10, max_lag=3, remove_mean=False)
    b = thermal_conductivity(np.ones((5, 3)) * 2, 1, temperature=300, volume=10, max_lag=3, remove_mean=False)
    result = replica_transport([a, b])
    np.testing.assert_allclose(result.mean, np.array(a.integrals) * 2.5)
    np.testing.assert_allclose(result.standard_error, np.array(a.integrals) * 1.5)
    with pytest.raises(ValueError):
        replica_transport([a])
    c = thermal_conductivity(np.ones((5, 3)), 1, temperature=400, volume=10, max_lag=3, remove_mean=False)
    with pytest.raises(ValueError):
        replica_transport([a, c])


@pytest.mark.parametrize('changes', [{'volume': 0}, {'temperature': -1}, {'max_lag': 5}, {'timestep': float('nan')}])
def test_invalid_metadata(changes):
    kwargs = dict(timestep=1, temperature=300, volume=10, max_lag=2)
    kwargs.update(changes)
    with pytest.raises(ValueError):
        thermal_conductivity(np.ones((5, 3)), **kwargs)
