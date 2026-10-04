"""Synthetic checks for electronic and magnetic analysis contracts."""

import numpy as np
import pytest
from httk.core import load_property_definition

from httk.analyse import definitions as defs
from httk.analyse.matsci.electronic import (
    align_energies,
    band_edges,
    electron_count,
    fit_effective_mass,
    integrate_dos,
    magnetic_moments,
    solve_chemical_potential,
    summarize_dielectric,
)
from httk.analyse.records import bound_values, records


def test_piecewise_dos_bounds_spin_multiplier_and_zero_temperature_count() -> None:
    energies = (0.0, 1.0, 2.0)
    dos = (0.0, 2.0, 0.0)
    assert integrate_dos(energies, dos, 0.5, 1.5, spin_degeneracy=1) == pytest.approx(1.5)
    assert integrate_dos(energies, dos, -1.0, 3.0, spin_degeneracy=2) == pytest.approx(4.0)
    assert electron_count(energies, dos, 1.0, temperature=0, spin_degeneracy=1) == pytest.approx(1.0)


def test_fermi_dirac_count_and_chemical_potential_bracket_solve() -> None:
    energies = np.linspace(-2, 2, 2001)
    dos = np.ones_like(energies)
    mu = solve_chemical_potential(energies, dos, 2.1, temperature=300, spin_degeneracy=2)
    assert mu == pytest.approx(-0.95, abs=1e-5)
    assert electron_count(energies, dos, mu, temperature=300, spin_degeneracy=2) == pytest.approx(2.1, abs=1e-10)


@pytest.mark.parametrize("upper,chemical_potential", ((10.0, 1.0), (100.0, 0.02), (100.0, 80.0)))
def test_finite_temperature_constant_dos_matches_analytic_integral(upper: float, chemical_potential: float) -> None:
    temperature = 300.0
    thermal_width = 8.617333262145e-5 * temperature
    expected = thermal_width * (
        np.logaddexp(0.0, chemical_potential / thermal_width)
        - np.logaddexp(0.0, (chemical_potential - upper) / thermal_width)
    )
    assert electron_count(
        (0.0, upper), (1.0, 1.0), chemical_potential, temperature=temperature, spin_degeneracy=1
    ) == pytest.approx(expected, abs=1e-13)


def test_zero_temperature_count_clamps_outside_grid_and_solver_finds_small_count() -> None:
    assert electron_count((0.0, 1.0), (1.0, 1.0), -1.0, temperature=0, spin_degeneracy=1) == 0
    assert electron_count((0.0, 1.0), (1.0, 1.0), 2.0, temperature=0, spin_degeneracy=1) == pytest.approx(1)
    assert solve_chemical_potential((0.0, 1.0), (1.0, 1.0), 0.1, temperature=0, spin_degeneracy=1) == pytest.approx(0.1)


def test_finite_temperature_linear_dos_matches_dense_independent_quadrature() -> None:
    energies = np.array((0.0, 0.03, 0.5, 10.0))
    density = np.array((0.1, 4.0, 0.5, 0.2))
    chemical_potential, temperature = 0.05, 350.0
    fine_energy = np.linspace(energies[0], energies[-1], 400_001)
    thermal_width = 8.617333262145e-5 * temperature
    reference = np.trapezoid(
        np.interp(fine_energy, energies, density) / (np.exp((fine_energy - chemical_potential) / thermal_width) + 1),
        fine_energy,
    )
    assert electron_count(
        energies, density, chemical_potential, temperature=temperature, spin_degeneracy=1
    ) == pytest.approx(reference, abs=1e-9)


def test_band_edges_reference_degeneracy_direct_and_indirect_gap() -> None:
    result = band_edges(
        ((-1.0, -0.8), (0.5, 0.3)),
        ((2.0, 2.0), (0.0, 0.0)),
        maximum_occupation=2,
        energy_reference=-0.2,
    )
    assert not result.metallic
    assert result.indirect_gap == pytest.approx(1.1)
    assert result.direct_gap == pytest.approx(1.1)
    assert (result.vbm_band, result.vbm_kpoint) == (0, 1)
    assert (result.cbm_band, result.cbm_kpoint) == (1, 1)
    assert align_energies((-1, 1), -0.2) == pytest.approx((-0.8, 1.2))


def test_metal_classification_and_missing_direct_pair() -> None:
    result = band_edges(
        ((-1, -1), (0, 0), (1, 1)),
        ((1, 0.5), (0, 0), (0, 0)),
        maximum_occupation=1,
        energy_reference=0,
    )
    assert result.metallic
    with pytest.raises(ValueError, match="both occupied and empty"):
        band_edges(((0,),), ((0.5,),), maximum_occupation=1, energy_reference=0)
    with pytest.raises(ValueError, match="same-k"):
        band_edges(((0, 1),), ((1, 0),), maximum_occupation=1, energy_reference=0)


def test_anisotropic_quadratic_fit_recovers_signed_rotated_masses() -> None:
    rng = np.random.default_rng(18)
    k = rng.uniform(-0.15, 0.15, size=(120, 3))
    rotation, _ = np.linalg.qr(rng.normal(size=(3, 3)))
    curvature = rotation @ np.diag((2.0, 4.0, -1.0)) @ rotation.T
    energy = 0.7 + 0.3 * k[:, 0] - 0.2 * k[:, 1] + np.einsum("ni,ij,nj->n", k, curvature, k) / 2
    result = fit_effective_mass(k, energy, (0, 0, 0))
    assert result.rank == 10
    assert result.residual_rms < 1e-13
    assert np.asarray(result.hessian) == pytest.approx(curvature, abs=1e-12)
    assert result.eigenvalues == pytest.approx((-_MASS_FACTOR, _MASS_FACTOR / 4, _MASS_FACTOR / 2), rel=1e-12)


def test_effective_mass_rejects_rank_deficient_k_samples() -> None:
    x = np.linspace(-1, 1, 12)
    k = np.column_stack((x, x**2, x**3))
    with pytest.raises(ValueError, match="rank deficient"):
        fit_effective_mass(k, x**2, (0, 0, 0))


def test_dos_and_occupation_boundary_validation() -> None:
    with pytest.raises(ValueError, match="increasing"):
        integrate_dos((0, 1, 1), (1, 1, 1), 0, 1, spin_degeneracy=1)
    with pytest.raises(ValueError, match="exceeds"):
        solve_chemical_potential((0, 1), (1, 1), 2, temperature=0, spin_degeneracy=1)
    with pytest.raises(ValueError, match="occupations"):
        band_edges(((0,), (1,)), ((2,), (-0.1,)), maximum_occupation=1, energy_reference=0)


def test_moment_sum_correlations_and_explicit_normalization() -> None:
    moments = ((2, 0, 0), (0, -3, 0), (1, 1, 0))
    result = magnetic_moments(moments, ((0, 1), (0, 2)), normalize_pairs=True)
    assert result.total == (3.0, -2.0, 0.0)
    assert result.correlations == pytest.approx((0.0, 1 / np.sqrt(2)))
    raw = magnetic_moments(moments, ((0, 2),))
    assert raw.correlations == (2.0,)
    with pytest.raises(ValueError, match="zero moments"):
        magnetic_moments(((0, 0, 0), (1, 0, 0)), ((0, 1),), normalize_pairs=True)


def test_static_dielectric_principal_values_and_symmetry_validation() -> None:
    summary = summarize_dielectric(((3, 0, 0), (0, 2, 0), (0, 0, 1)))
    assert summary.eigenvalues == (1.0, 2.0, 3.0)
    assert summary.mean == pytest.approx(2.0)
    with pytest.raises(ValueError, match="symmetric"):
        summarize_dielectric(((1, 2, 0), (0, 1, 0), (0, 0, 1)))


_MASS_FACTOR = 7.619964231073853


def test_thermal_counts_do_not_extrapolate_past_sampled_dos():
    for mu, expected in [(-10.0, 0.0), (10.0, 1.0)]:
        assert electron_count([0, 1], [1, 1], mu, temperature=300, spin_degeneracy=1) == pytest.approx(
            expected, abs=1e-14
        )
    mu = solve_chemical_potential([0, 1], [1, 1], 0.9999, temperature=300, spin_degeneracy=1)
    assert electron_count([0, 1], [1, 1], mu, temperature=300, spin_degeneracy=1) == pytest.approx(0.9999, abs=1e-10)


def test_chemical_potential_rejects_unreachable_even_within_solver_tolerance():
    with pytest.raises(ValueError, match="exceeds"):
        solve_chemical_potential([0, 1], [1, 1], 1 + 5e-11, temperature=300, spin_degeneracy=1)


def _checked(result: object, **selection: object):
    bound = bound_values(result, **selection)
    for item in bound:
        load_property_definition(item.binding.definition).check(item.value)
    return bound


def test_dielectric_summary_binding_requires_kind() -> None:
    summary = summarize_dielectric(((3, 0, 0), (0, 2, 0), (0, 0, 1)))
    (static,) = _checked(summary, kind="static")
    assert (static.binding.definition, static.value) == (defs.STATIC_RELATIVE_PERMITTIVITY, 2.0)
    (high,) = _checked(summary, kind="high_frequency")
    assert high.binding.definition == defs.HIGH_FREQUENCY_RELATIVE_PERMITTIVITY
    assert records(summary, kind="static")[0].name == "static_relative_permittivity"
    with pytest.raises(TypeError):
        bound_values(summary)
    with pytest.raises(ValueError, match="kind"):
        bound_values(summary, kind="optical")
    with pytest.raises(TypeError, match="selection"):
        bound_values(summary, kind="static", lag_index=1)


def test_effective_mass_and_magnetic_moment_bindings() -> None:
    x = np.linspace(-0.2, 0.2, 7)
    k = np.array([(a, b, c) for a in x for b in x for c in x])
    energy = 0.5 * 7.619964231073853 * (k[:, 0] ** 2 / 0.5 + k[:, 1] ** 2 / 1.0 + k[:, 2] ** 2 / 2.0)
    fit = fit_effective_mass(k, energy, (0, 0, 0))
    (bound,) = _checked(fit)
    assert bound.binding.definition == defs.RELATIVE_EFFECTIVE_MASS
    assert bound.value["center"] == [0.0, 0.0, 0.0]
    assert bound.value["tensor"][0][0] == pytest.approx(0.5)
    moments = magnetic_moments(((1.0, 0.0, 0.0), (0.5, 2.0, 0.0)))
    (total,) = _checked(moments)
    assert total.binding.definition == defs.TOTAL_MAGNETIC_MOMENT
    assert total.value == [1.5, 2.0, 0.0]
    assert records(moments)[0].name == "total_magnetic_moment"
