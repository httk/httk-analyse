import numpy as np
import pytest

from httk.analyse.matsci.phonons import (
    harmonic_thermodynamics,
    harmonic_thermodynamics_from_dos,
    mode_gruneisen,
    quasiharmonic,
)


def test_einstein_mode_zero_and_high_temperature_limits():
    result = harmonic_thermodynamics([2.0, 2.0, 2.0], [0.0, 1_000_000.0])
    assert result.free_energy[0] == result.zero_point_energy
    assert result.internal_energy[0] == result.zero_point_energy
    assert result.entropy[0] == result.heat_capacity[0] == 0.0
    assert result.heat_capacity[1] == pytest.approx(3 * 8.617333262145e-5, rel=1e-8)
    assert result.internal_energy[1] == pytest.approx(3 * 8.617333262145e-5 * 1_000_000, rel=1e-8)


def test_explicit_zero_and_imaginary_policy_weights():
    with pytest.raises(ValueError, match="zero-frequency"):
        harmonic_thermodynamics([0.0, -1.0, 2.0], [0.0])
    result = harmonic_thermodynamics(
        [0.0, -1.0, 2.0], [0.0, 300.0], [2.0, 3.0, 4.0], zero_modes="omit", imaginary="omit"
    )
    assert result.retained_mode_weight == 4.0
    assert result.excluded_zero_weight == 2.0
    assert result.excluded_imaginary_weight == 3.0


def test_dos_trapezoid_counts_endpoints_and_reports_zero_policy():
    result = harmonic_thermodynamics_from_dos([0.0, 1.0, 2.0], [1.0, 1.0, 1.0], [0.0], zero_modes="omit")
    assert result.retained_mode_weight == pytest.approx(1.5)
    assert result.excluded_zero_weight == pytest.approx(0.5)
    positive = harmonic_thermodynamics_from_dos([1.0, 2.0, 3.0], [1.0, 1.0, 1.0], [0.0])
    assert positive.retained_mode_weight == pytest.approx(2.0)


def test_mode_gruneisen_uses_matched_mode_log_slope():
    volumes = np.array([8.0, 10.0, 12.0, 14.0])
    frequencies = np.column_stack((3 * (volumes / 10) ** -2.0, 5 * (volumes / 10) ** 0.5))
    result = mode_gruneisen(volumes, frequencies, reference_volume=10.0)
    assert result.values == pytest.approx((2.0, -0.5), abs=1e-12)
    assert result.rank == 3
    assert result.degree == 2


def test_quasiharmonic_synthetic_expansion_and_exclusion_consistency():
    volumes = np.linspace(9.0, 11.0, 9)
    v0, b0, bp, e0 = 10.0, 0.2, 4.0, -5.0
    q = (v0 / volumes) ** (2 / 3) - 1
    static = e0 + 9 * v0 * b0 / 16 * (2 * q**2 + (bp - 4) * q**3)
    frequencies = np.asarray([[2.0 * (volume / v0) ** -2, 4.0] for volume in volumes])
    result = quasiharmonic(volumes, static, frequencies, [0.0, 300.0, 600.0])
    assert result.equilibrium_volumes[0] == pytest.approx(v0, abs=0.06)
    assert result.equilibrium_volumes[-1] > result.equilibrium_volumes[0]
    assert result.volumetric_expansion[-1] > 0
    np.testing.assert_allclose(
        result.volumetric_expansion,
        np.gradient(result.equilibrium_volumes, [0, 300, 600], edge_order=2) / np.array(result.equilibrium_volumes),
    )
    altered = frequencies.copy()
    altered[-1, 0] = 0.0
    with pytest.raises(ValueError, match="same caller-matched modes"):
        quasiharmonic(volumes, static, altered, [0.0, 300.0, 600.0], zero_modes="omit")
    swapped = np.tile([0.0, -1.0, 2.0], (len(volumes), 1))
    swapped[-1] = [-1.0, 0.0, 2.0]
    with pytest.raises(ValueError, match="same caller-matched modes"):
        quasiharmonic(volumes, static, swapped, [0.0, 300.0, 600.0], zero_modes="omit", imaginary="omit")


def test_complex_frequencies_are_rejected():
    with pytest.raises(ValueError, match="real one-dimensional"):
        harmonic_thermodynamics(np.asarray([1 + 1j]), [0.0])


def test_inputs_and_result_sequences_are_immutable():
    frequencies = np.asarray([1.0, 2.0])
    temperatures = np.asarray([0.0, 100.0])
    original_frequencies, original_temperatures = frequencies.copy(), temperatures.copy()
    result = harmonic_thermodynamics(frequencies, temperatures)
    assert np.array_equal(frequencies, original_frequencies)
    assert np.array_equal(temperatures, original_temperatures)
    assert isinstance(result.free_energy, tuple)


def test_debye_dos_low_temperature_cubic_heat_capacity():
    kb = 1.380649e-23 / 1.602176634e-19
    h = 6.62607015e-34 / 1.602176634e-19 * 1e12
    theta = 100.0
    cutoff = kb * theta / h
    frequencies = np.linspace(0, cutoff, 20001)
    density = 9 * frequencies**2 / cutoff**3
    result = harmonic_thermodynamics_from_dos(frequencies, density, [1, 2])
    expected = 12 * np.pi**4 / 5 * kb * (np.array([1.0, 2.0]) / theta) ** 3
    np.testing.assert_allclose(result.heat_capacity, expected, rtol=2e-7)
    assert result.retained_mode_weight == pytest.approx(3, rel=1e-8)


def test_cutoff_omits_gamma_acoustic_noise_of_either_sign():
    normal = [1.0, 2.0, 3.5]
    weights = [1 / 64, 1 / 64, 1 / 64, 0.25, 0.5, 0.25]
    noisy = [1e-6, -1e-6, 1e-6, *normal]
    temps = [0.0, 100.0, 300.0]
    with pytest.raises(ValueError, match="negative frequencies"):
        harmonic_thermodynamics(noisy, temps, weights, zero_modes="omit")
    with pytest.raises(ValueError, match="negative frequencies"):
        harmonic_thermodynamics(noisy, temps, weights, zero_modes="omit", cutoff_frequency=1e-7)
    result = harmonic_thermodynamics(noisy, temps, weights, zero_modes="omit", cutoff_frequency=1e-3)
    clean = harmonic_thermodynamics(normal, temps, weights[3:])
    for name in ("free_energy", "internal_energy", "entropy", "heat_capacity"):
        np.testing.assert_allclose(getattr(result, name), getattr(clean, name), rtol=1e-14, atol=1e-18)
    assert result.zero_point_energy == pytest.approx(clean.zero_point_energy, rel=1e-14)
    assert result.excluded_zero_weight == pytest.approx(3 / 64)
    assert result.excluded_imaginary_weight == 0.0
    assert result.retained_mode_weight == pytest.approx(1.0)
    assert result.cutoff_frequency == 1e-3
    # Without a cutoff, tiny positive noise adds a large log-divergent term (about -19 meV at 300 K).
    kept = harmonic_thermodynamics([1e-6] * 3 + normal, [300.0], weights)
    assert kept.free_energy[0] - clean.free_energy[2] < -0.018
    # Zero-classification is strict (< cutoff) and a mode at the cutoff stays retained.
    assert harmonic_thermodynamics([1e-3, 1.0], [0.0], cutoff_frequency=1e-3).retained_mode_weight == 2.0
    # Default behaviour is unchanged and records a zero cutoff.
    assert harmonic_thermodynamics(normal, temps).cutoff_frequency == 0.0


def test_cutoff_applies_to_dos_zero_nodes():
    result = harmonic_thermodynamics_from_dos(
        [0.0, 1e-4, 1.0, 2.0], [1.0, 1.0, 1.0, 1.0], [0.0], zero_modes="omit", cutoff_frequency=1e-3
    )
    assert result.excluded_zero_weight == pytest.approx(5e-5 + 0.5)
    assert result.cutoff_frequency == 1e-3


def test_quasiharmonic_cutoff_tolerates_gamma_noise_sign_flip():
    volumes = np.linspace(9.0, 11.0, 9)
    v0, b0, bp, e0 = 10.0, 0.2, 4.0, -5.0
    q = (v0 / volumes) ** (2 / 3) - 1
    static = e0 + 9 * v0 * b0 / 16 * (2 * q**2 + (bp - 4) * q**3)
    normal = np.asarray([[2.0 * (volume / v0) ** -2, 4.0] for volume in volumes])
    noise = np.where(np.arange(len(volumes)) % 2 == 0, 1e-6, -1e-6)
    noisy = np.column_stack((noise, -noise, normal))
    temps = [0.0, 300.0, 600.0]
    with pytest.raises(ValueError):
        quasiharmonic(volumes, static, noisy, temps, zero_modes="omit", imaginary="omit")
    result = quasiharmonic(volumes, static, noisy, temps, zero_modes="omit", cutoff_frequency=1e-3)
    clean = quasiharmonic(volumes, static, normal, temps)
    np.testing.assert_allclose(result.equilibrium_volumes, clean.equilibrium_volumes, rtol=1e-12)
    np.testing.assert_allclose(result.free_energies, clean.free_energies, rtol=1e-12)
    assert result.retained_mode_weight == 2.0
    assert result.excluded_zero_weight == 2.0
    assert result.excluded_imaginary_weight == 0.0
    assert result.cutoff_frequency == 1e-3


@pytest.mark.parametrize("bad", [-1e-3, float("nan"), float("inf"), "x"])
def test_invalid_cutoff_frequency_raises(bad):
    with pytest.raises(ValueError, match="cutoff_frequency"):
        harmonic_thermodynamics([1.0, 2.0], [300.0], cutoff_frequency=bad)
    with pytest.raises(ValueError, match="cutoff_frequency"):
        harmonic_thermodynamics_from_dos([1.0, 2.0], [1.0, 1.0], [300.0], cutoff_frequency=bad)
