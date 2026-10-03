import numpy as np
import pytest

from httk.analyse.matsci.phonons import (
    harmonic_from_phonopy,
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


def test_phonopy_adapter_normalizes_q_weights_and_rejects_complex_inputs():
    class Mesh:
        frequencies = np.asarray([[1.0, 2.0, 3.0], [1.0, 2.0, 3.0]])
        weights = np.asarray([1, 3])

    result = harmonic_from_phonopy(type("Phonon", (), {"mesh": Mesh()})(), [0.0])
    assert result.retained_mode_weight == pytest.approx(3.0)
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
    expected = 12 * np.pi**4 / 5 * kb * (np.array([1., 2.]) / theta)**3
    np.testing.assert_allclose(result.heat_capacity, expected, rtol=2e-7)
    assert result.retained_mode_weight == pytest.approx(3, rel=1e-8)
