"""Checks for elastic tensors and linear strain fits."""

from dataclasses import FrozenInstanceError

import numpy as np
import pytest

from httk.analyse.matsci.elasticity import ElasticTensor, fit_energy_strain, fit_stress_strain


def _isotropic(lam: float = 100.0, shear: float = 50.0) -> ElasticTensor:
    stiffness = np.zeros((6, 6))
    stiffness[:3, :3] = lam
    np.fill_diagonal(stiffness[:3, :3], lam + 2.0 * shear)
    stiffness[3:, 3:] = np.eye(3) * shear
    return ElasticTensor(stiffness)


def test_isotropic_moduli_directional_response_and_full_tensor_conversions() -> None:
    tensor = _isotropic()

    assert tensor.bulk_modulus_voigt == pytest.approx(400.0 / 3.0)
    assert tensor.bulk_modulus_reuss == pytest.approx(400.0 / 3.0)
    assert tensor.bulk_modulus_hill == pytest.approx(400.0 / 3.0)
    assert tensor.shear_modulus_voigt == pytest.approx(50.0)
    assert tensor.shear_modulus_reuss == pytest.approx(50.0)
    assert tensor.shear_modulus_hill == pytest.approx(50.0)
    assert tensor.universal_anisotropy == pytest.approx(0.0, abs=1e-12)
    assert tensor.young_modulus((1.0, 2.0, 3.0)) == pytest.approx(400.0 / 3.0)
    assert tensor.shear_modulus((1.0, 0.0, 0.0), (0.0, 1.0, 0.0)) == pytest.approx(50.0)
    assert tensor.poisson_ratio((1.0, 0.0, 0.0), (0.0, 1.0, 0.0)) == pytest.approx(1.0 / 3.0)
    assert ElasticTensor.from_full(tensor.to_full()) == tensor
    full_compliance = np.asarray(tensor.compliance_full())
    assert full_compliance[0, 1, 0, 1] == pytest.approx(1.0 / 200.0)
    assert full_compliance[0, 0, 1, 1] == pytest.approx(-1.0 / 400.0)


def test_cubic_moduli_rotation_invariance_and_pressure_stability() -> None:
    stiffness = np.zeros((6, 6))
    np.fill_diagonal(stiffness, (200.0, 200.0, 200.0, 80.0, 80.0, 80.0))
    stiffness[:3, :3] += np.full((3, 3), 120.0) - np.eye(3) * 120.0
    tensor = ElasticTensor(stiffness)
    angle = 0.37
    rotation = np.array([[np.cos(angle), -np.sin(angle), 0.0], [np.sin(angle), np.cos(angle), 0.0], [0.0, 0.0, 1.0]])
    rotated = tensor.rotate(rotation)

    assert tensor.bulk_modulus_voigt == pytest.approx(440.0 / 3.0)
    assert tensor.shear_modulus_voigt == pytest.approx(64.0)
    assert tensor.shear_modulus_reuss == pytest.approx(400.0 / 7.0)
    assert rotated.bulk_modulus_voigt == pytest.approx(tensor.bulk_modulus_voigt)
    assert rotated.shear_modulus_voigt == pytest.approx(tensor.shear_modulus_voigt)
    assert rotated.stability_eigenvalues == pytest.approx(tensor.stability_eigenvalues)

    stable = _isotropic(lam=20.0, shear=10.0)
    assert stable.is_stable()
    assert stable.is_stable_under_pressure(1.0)
    assert not stable.is_stable_under_pressure(11.0)
    assert stable.pressure_stability_eigenvalues(11.0) == pytest.approx((-2.0, -2.0, -2.0, -2.0, -2.0, 91.0))


def test_stress_fit_recovers_noisy_stiffness_and_offsets_without_mutating_inputs() -> None:
    rng = np.random.default_rng(47)
    basis = rng.normal(size=(6, 6))
    stiffness = basis.T @ basis + np.eye(6) * 20.0
    offset = np.array((0.1, -0.2, 0.05, 0.01, 0.03, -0.04))
    strains = rng.uniform(-0.02, 0.02, size=(80, 6))
    stresses = strains @ stiffness.T + offset + rng.normal(scale=1e-5, size=(80, 6))
    strains_before, stresses_before = strains.copy(), stresses.copy()

    fit = fit_stress_strain(strains, stresses)

    np.testing.assert_array_equal(strains, strains_before)
    np.testing.assert_array_equal(stresses, stresses_before)
    np.testing.assert_allclose(fit.tensor.stiffness, stiffness, rtol=2e-3, atol=2e-3)
    assert fit.stress_offset == pytest.approx(offset, abs=2e-6)
    assert fit.rmse > 0.0
    assert fit.condition_number >= 1.0
    assert len(fit.strain_range) == 6
    assert isinstance(fit.residuals, tuple) and isinstance(fit.residuals[0], tuple)
    with pytest.raises(FrozenInstanceError):
        fit.rmse = 0.0  # type: ignore[misc]


def test_energy_fit_recovers_stiffness_and_both_offsets() -> None:
    rng = np.random.default_rng(92)
    basis = rng.normal(size=(6, 6))
    stiffness = basis.T @ basis + np.eye(6) * 30.0
    stress_offset = np.array((0.3, -0.1, 0.05, 0.02, -0.02, 0.01))
    energy_offset = -15.0
    volume = 32.0
    strains = rng.uniform(-0.04, 0.04, size=(120, 6))
    energies = energy_offset + volume * (
        strains @ stress_offset + 0.5 * np.einsum("ni,ij,nj->n", strains, stiffness, strains)
    )
    energies += rng.normal(scale=1e-6, size=len(energies))

    fit = fit_energy_strain(strains, energies, volume)

    np.testing.assert_allclose(fit.tensor.stiffness, stiffness, rtol=2e-3, atol=2e-3)
    assert fit.stress_offset == pytest.approx(stress_offset, abs=2e-5)
    assert fit.energy_offset == pytest.approx(energy_offset, abs=1e-6)
    assert fit.rmse > 0.0
    assert isinstance(fit.residuals, tuple) and isinstance(fit.residuals[0], float)


def test_energy_fit_without_offsets_assumes_zero_reference_and_stress() -> None:
    rng = np.random.default_rng(12)
    basis = rng.normal(size=(6, 6))
    stiffness = basis.T @ basis + np.eye(6)
    strains = rng.uniform(-0.1, 0.1, size=(80, 6))
    volume = 10.0
    energies = volume * 0.5 * np.einsum("ni,ij,nj->n", strains, stiffness, strains)

    fit = fit_energy_strain(strains, energies, volume, fit_offset=False)

    np.testing.assert_allclose(fit.tensor.stiffness, stiffness, rtol=1e-10, atol=1e-10)
    assert fit.energy_offset == 0.0
    assert fit.stress_offset == (0.0,) * 6


@pytest.mark.parametrize(
    "call",
    [
        lambda: ElasticTensor(np.eye(5)),
        lambda: ElasticTensor(np.full((6, 6), np.nan)),
        lambda: ElasticTensor(np.eye(6, dtype=complex) * (1.0 + 1.0j)),
        lambda: ElasticTensor(np.eye(6) + np.triu(np.ones((6, 6)), 1)),
        lambda: _isotropic().rotate(np.diag((-1.0, 1.0, 1.0))),
        lambda: _isotropic().young_modulus((0.0, 0.0, 0.0)),
        lambda: _isotropic().shear_modulus((1.0, 0.0, 0.0), (1.0, 1.0, 0.0)),
        lambda: _isotropic().poisson_ratio((1.0, 0.0, 0.0), (1.0, 1.0, 0.0)),
        lambda: fit_stress_strain(np.zeros((10, 6)), np.zeros((10, 6))),
        lambda: fit_energy_strain(np.zeros((30, 6)), np.ones(30), 1.0),
        lambda: fit_energy_strain(np.ones((30, 6)), np.ones(30), 0.0),
    ],
)
def test_invalid_and_rank_deficient_inputs_raise_value_error(call: object) -> None:
    with pytest.raises(ValueError):
        call()  # type: ignore[operator]


def test_unstable_tensors_are_retained_but_singular_compliance_is_rejected() -> None:
    unstable = np.eye(6)
    unstable[0, 0] = -1.0
    tensor = ElasticTensor(unstable)

    assert tensor.stability_eigenvalues[0] == pytest.approx(-1.0)
    assert not tensor.is_stable()
    with pytest.raises(ValueError, match="singular"):
        assert _isotropic(shear=0.0).compliance


def test_anisotropic_directional_response_and_pressure_spectrum_rotate_together():
    rng = np.random.default_rng(849)
    matrix = rng.normal(size=(6, 6))
    tensor = ElasticTensor(matrix.T @ matrix + np.eye(6))
    rotation, _ = np.linalg.qr(rng.normal(size=(3, 3)))
    if np.linalg.det(rotation) < 0:
        rotation[:, 0] *= -1
    rotated = tensor.rotate(rotation)
    direction = np.array([1.0, 0.0, 0.0])
    transverse = np.array([0.0, 1.0, 0.0])
    assert rotated.young_modulus(rotation @ direction) == pytest.approx(tensor.young_modulus(direction))
    assert rotated.shear_modulus(rotation @ direction, rotation @ transverse) == pytest.approx(
        tensor.shear_modulus(direction, transverse)
    )
    assert rotated.poisson_ratio(rotation @ direction, rotation @ transverse) == pytest.approx(
        tensor.poisson_ratio(direction, transverse)
    )
    np.testing.assert_allclose(
        rotated.pressure_stability_eigenvalues(0.2), tensor.pressure_stability_eigenvalues(0.2), atol=1e-13
    )
