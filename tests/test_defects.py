"""Analytic checks for explicit-reference defects and activated rates."""

import math
from dataclasses import FrozenInstanceError
from typing import cast

import pytest

from httk.analyse.matsci.defects import (
    adsorption_energy,
    charge_transition_levels,
    defect_formation_energy,
    fit_arrhenius,
    neb_profile,
    segregation_energy,
    surface_energy,
)


def test_defect_energy_terms_charge_sign_and_bulk_cancellation() -> None:
    bulk = defect_formation_energy(-12.0, -12.0, {}, {}, charge=0, fermi_level=0, vbm=0, alignment=0, correction=0)
    assert bulk.energy == 0
    result = defect_formation_energy(
        -9.0,
        -10.0,
        {"A": 1, "B": -1},
        {"A": -2.0, "B": -3.0},
        charge=2,
        fermi_level=0.5,
        vbm=1.0,
        alignment=0.1,
        correction=-0.2,
    )
    assert result.atom_deltas == (("A", 1), ("B", -1))
    assert result.terms == (
        ("defect_energy", -9.0),
        ("-host_energy", 10.0),
        ("-atom_reservoirs", -1.0),
        ("charge_fermi_vbm_alignment", 3.2),
        ("correction", -0.2),
    )
    assert result.energy == pytest.approx(3.0)
    with pytest.raises(FrozenInstanceError):
        result.energy = 0.0  # type: ignore[misc]
    with pytest.raises(ValueError, match="integer"):
        defect_formation_energy(
            -1,
            -1,
            cast(dict[str, int], {"A": 0.5}),
            {"A": -1},
            charge=0,
            fermi_level=0,
            vbm=0,
            alignment=0,
            correction=0,
        )
    source_terms = [["value", 1.0]]
    copied = type(result)(1.0, source_terms, [], 0)  # type: ignore[arg-type]
    source_terms[0][1] = 9.0
    assert copied.terms == (("value", 1.0),)


def test_charge_transitions_keep_only_lower_envelope_crossings() -> None:
    transitions = charge_transition_levels({1: 0.0, 0: -1.0, -1: 0.0, 3: -20.0}, (-3.0, 3.0))
    assert transitions == ()  # charge 3 keeps every other crossing metastable
    transitions = charge_transition_levels({1: 0.0, 0: -1.0, -1: 0.0}, (-2.0, 2.0))
    assert tuple(item.fermi_level for item in transitions) == pytest.approx((-1.0, 1.0))
    assert transitions[0].tied_charges == (0, 1)
    assert transitions[0].left_charges == (1,)
    assert transitions[0].right_charges == (0,)
    assert transitions[1].tied_charges == (-1, 0)
    tolerant = charge_transition_levels({1: 0.0, 0: 0.1, -1: 0.0}, (-1.0, 1.0), tolerance=0.2)
    assert len(tolerant) == 1
    assert tolerant[0].fermi_level == pytest.approx(0.0)
    assert tolerant[0].tied_charges == (-1, 0, 1)


def test_surface_adsorption_and_segregation_signs() -> None:
    addition = surface_energy(
        -5.0,
        -1.0,
        bulk_reference_atom_count=4,
        excess_atom_deltas={"A": 1},
        chemical_potentials={"A": -1.0},
        total_exposed_area=2.0,
    )
    assert addition.energy == pytest.approx(0.0)
    assert addition.surface_energy == pytest.approx(0.0)
    vacancy = surface_energy(
        -3.0,
        -1.0,
        bulk_reference_atom_count=4,
        excess_atom_deltas={"A": -1},
        chemical_potentials={"A": -1.0},
        total_exposed_area=2.0,
    )
    assert vacancy.energy == pytest.approx(0.0)
    assert vacancy.surface_energy == pytest.approx(0.0)
    assert adsorption_energy(-12.0, -10.0, {"H2": 1}, {"H2": -1.0}) == -1.0
    assert segregation_energy(-2.0, -3.5) == -1.5


def test_neb_profile_preserves_order_and_reports_forward_reverse_barriers() -> None:
    result = neb_profile((0, 0.2, 0.7, 1.0), (2.0, 5.0, 5.0, 1.0))
    assert result.forward_barrier == 3.0
    assert result.reverse_barrier == 4.0
    assert result.reaction_energy == -1.0
    assert result.saddle_index == 1
    assert result.tied_saddle_indices == (1, 2)
    with pytest.raises(ValueError, match="strictly increasing"):
        neb_profile((0, 1, 1), (0, 2, 0))


def test_arrhenius_fit_recovers_noisy_reference_with_weighted_diagnostics() -> None:
    activation, prefactor = 0.72, 4.0e12
    temperatures = (300.0, 400.0, 500.0, 700.0, 900.0)
    perturbations = (0.01, -0.02, 0.015, -0.005, 0.0)
    rates = tuple(
        prefactor * math.exp(-activation / (8.617333262145e-5 * t) + error)
        for t, error in zip(temperatures, perturbations, strict=True)
    )
    fit = fit_arrhenius(temperatures, rates, rate_unit="s^-1", weights=(1, 2, 3, 2, 1))
    expected_logs = tuple(math.log(rate) for rate in rates)
    weighted_x = tuple(1 / temperature for temperature in temperatures)
    assert fit.activation_energy == pytest.approx(activation, abs=0.004)
    assert fit.prefactor == pytest.approx(prefactor, rel=0.02)
    assert fit.rate_unit == "s^-1"
    assert fit.rank == 2
    assert fit.condition_number >= 1
    assert fit.weighted_rmse > 0
    assert len(fit.log_residuals) == len(expected_logs) == len(weighted_x)
    assert sum(fit.log_residuals) == pytest.approx(0.0, abs=0.01)


def test_arrhenius_fit_recovers_exact_line_with_orthogonal_noise() -> None:
    activation, prefactor = 0.4, 7.0e8
    reciprocal_temperatures = (0.001, 0.002, 0.003, 0.004, 0.005)
    temperatures = tuple(1.0 / value for value in reciprocal_temperatures)
    noise = (0.02, -0.04, 0.02, 0.0, 0.0)
    rates = tuple(
        prefactor * math.exp(-activation / (8.617333262145e-5 * temperature) + error)
        for temperature, error in zip(temperatures, noise, strict=True)
    )
    fit = fit_arrhenius(temperatures, rates, rate_unit="ps^-1")
    with pytest.raises(ValueError):
        fit_arrhenius(temperatures, rates, rate_unit="1/s")
    assert fit.activation_energy == pytest.approx(activation, rel=1e-11)
    assert fit.prefactor == pytest.approx(prefactor, rel=1e-11)
    assert fit.log_residuals == pytest.approx(noise, abs=1e-11)


def test_neb_saddle_index_is_actual_maximum_even_with_near_ties():
    result = neb_profile([0, 1, 2, 3], [0, 0.9995, 1, 0.2], tie_tolerance=0.001)
    assert result.saddle_index == 2
    assert result.tied_saddle_indices == (1, 2)
    assert result.forward_barrier == 1
