"""Direct-sum and analytic checks of displacement and spectral conventions."""

import numpy as np
import pytest

from httk.analyse.matsci.dynamics import (
    TensorSeries,
    diffusion_from_msd,
    integrate_vacf,
    intermediate_scattering,
    mean_squared_displacement,
    van_hove_self,
    velocity_autocorrelation,
    velocity_spectrum,
)


def test_ballistic_tensor_and_com_removal():
    velocity = np.array([[1.0, 2.0, 0.0], [3.0, 0.0, 1.0]])
    positions = np.arange(9)[:, None, None] * 0.2 * velocity
    result = mean_squared_displacement(positions, 0.2, max_lag=5)
    expected = np.einsum('ni,nj->ij', velocity, velocity) / 2
    for k, tensor in enumerate(result.tensors):
        np.testing.assert_allclose(tensor, expected * (k * 0.2) ** 2, atol=1e-14)
    assert result.counts == (9, 8, 7, 6, 5, 4)
    moving = positions + np.arange(9)[:, None, None] * np.array([4.0, -2.0, 5.0])
    corrected = mean_squared_displacement(moving, 0.2, max_lag=5, remove_com=True, masses=[1, 3])
    baseline = mean_squared_displacement(positions, 0.2, max_lag=5, remove_com=True, masses=[1, 3])
    np.testing.assert_allclose(corrected.tensors, baseline.tensors, atol=2e-14)


def test_vacf_matches_direct_sum_and_trapezoid():
    rng = np.random.default_rng(56)
    velocities = rng.normal(size=(11, 4, 3))
    result = velocity_autocorrelation(velocities, 0.1, max_lag=4)
    for k, tensor in enumerate(result.tensors):
        direct = sum(np.outer(velocities[t, n], velocities[t + k, n]) for t in range(11 - k) for n in range(4)) / (
            4 * (11 - k)
        )
        np.testing.assert_allclose(tensor, direct, atol=1e-15)
    integrated = integrate_vacf(result)
    np.testing.assert_allclose(integrated.tensors[0], 0)
    np.testing.assert_allclose(
        integrated.tensors[1], 0.05 * (np.array(result.tensors[0]) + np.array(result.tensors[1]))
    )
    centered = velocity_autocorrelation(velocities + 4, 0.1, max_lag=4, remove_mean=True)
    baseline = velocity_autocorrelation(velocities, 0.1, max_lag=4, remove_mean=True)
    np.testing.assert_allclose(centered.tensors, baseline.tensors, atol=2e-15)


def test_diffusion_tensor_fit_and_units():
    diffusion = np.array([[1, 0.2, 0], [0.2, 2, 0.1], [0, 0.1, 3]])
    times = np.linspace(0, 10, 21)
    tensors = 2 * times[:, None, None] * diffusion + np.eye(3)[None, :, :] * 0.3
    result = diffusion_from_msd(TensorSeries(tuple(times), tensors, (50,) * 21, 'MSD angstrom^2'), window=(3, 16))
    np.testing.assert_allclose(result.tensor, diffusion, atol=1e-14)
    np.testing.assert_allclose(result.intercept, np.eye(3) * 0.3, atol=1e-14)
    assert result.isotropic == pytest.approx(2.0)
    assert np.max(result.rmse) < 1e-13


def test_self_van_hove_retains_outside_range_probability():
    positions = np.arange(5)[:, None, None] * np.array([[[1.0, 0.0, 0.0], [3.0, 0.0, 0.0]]])
    result = van_hove_self(positions, 0.2, lag=1, bins=[0, 2])
    assert result.counts == (4,)
    assert result.samples == 8
    assert result.density[0] * (4 * np.pi / 3) * 8 == pytest.approx(0.5)


def test_scattering_phase_sign_and_zero_wavevector():
    positions = np.arange(8)[:, None, None] * np.array([[[0.1, 0, 0], [0.1, 0, 0]]])
    self_part = intermediate_scattering(positions, 0.5, [[0, 0, 0], [2, 0, 0]], kind='self', max_lag=3)
    coherent = intermediate_scattering(positions, 0.5, [[0, 0, 0], [2, 0, 0]], kind='coherent', max_lag=3)
    for k in range(4):
        assert self_part.values[k][0] == pytest.approx(1)
        assert coherent.values[k][0] == pytest.approx(2)
        assert self_part.values[k][1] == pytest.approx(np.exp(1j * 0.2 * k))
        assert coherent.values[k][1] == pytest.approx(2 * np.exp(1j * 0.2 * k))


@pytest.mark.parametrize('size', [31, 32])
@pytest.mark.parametrize('window', ['none', 'hann'])
def test_spectral_parseval_and_frequency_peak(size, window):
    dt = 0.125
    signal = np.sin(2 * np.pi * 5 * np.arange(size) / size)
    velocities = np.tile(signal[:, None, None], (1, 3, 3))
    spectrum = velocity_spectrum(velocities, dt, window=window)
    assert spectrum.remove_mean is False
    assert sum(spectrum.power) * spectrum.frequency_spacing == pytest.approx(spectrum.mean_square, rel=1e-13)
    peak = np.argmax(spectrum.power)
    assert spectrum.frequencies[peak] == pytest.approx(5 / (size * dt))
    if window == 'none':
        assert spectrum.mean_square == pytest.approx(0.5)


@pytest.mark.parametrize(
    'call',
    [
        lambda: mean_squared_displacement(np.zeros((3, 2, 3)), 0),
        lambda: mean_squared_displacement(np.zeros((3, 2, 3)), 0.1, max_lag=3),
        lambda: mean_squared_displacement(np.zeros((3, 2, 3)), 0.1, remove_com=True),
        lambda: mean_squared_displacement(np.zeros((3, 2, 3)), 0.1, masses=[1, 1]),
        lambda: velocity_autocorrelation(np.full((3, 2, 3), complex(1, 1)), 0.1),
        lambda: velocity_spectrum(np.zeros((2, 1, 3)), 1, window='hann'),
        lambda: van_hove_self(np.zeros((3, 2, 3)), 1, lag=1, bins=[0, 0]),
    ],
)
def test_bad_inputs(call):
    with pytest.raises(ValueError):
        call()


def test_source_arrays_are_unchanged():
    data = np.arange(60, dtype=float).reshape(10, 2, 3)
    before = data.copy()
    mean_squared_displacement(data, 0.2, remove_com=True, masses=[1, 2])
    velocity_autocorrelation(data, 0.2, remove_mean=True)
    velocity_spectrum(data, 0.2, remove_mean=True)
    velocity_spectrum(data, 0.2)
    np.testing.assert_array_equal(data, before)


def test_distinct_van_hove_excludes_same_atom_and_retains_number_density():
    from httk.analyse.matsci.dynamics import van_hove_distinct

    positions = np.tile([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], (4, 1, 1))
    result = van_hove_distinct(positions, 1, lag=2, bins=[0, 0.5, 1.5], cell=np.eye(3) * 4)
    assert result.kind == 'distinct'
    assert result.counts == (0, 4)
    assert result.samples == 4
    assert np.sum(np.array(result.density) * (4 * np.pi / 3) * np.diff(np.array(result.edges) ** 3)) == pytest.approx(1)
