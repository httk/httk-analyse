"""Checks for machine-learning potential residual metrics."""

from dataclasses import FrozenInstanceError

import numpy as np
import pytest
from httk.core import DerivedDataRecord, load_property_definition
from httk.core.definition_ids import ATOMIC_FORCE, STRESS_TENSOR

from httk.analyse import definitions as defs
from httk.analyse.matsci.mlip import energy_errors, force_errors, stress_errors
from httk.analyse.records import bound_values, records


def test_energy_errors_keep_raw_values_and_apply_only_explicit_offset() -> None:
    result = energy_errors([0.0, 0.0], [0.0, 10.0], atom_counts=[1, 4], offset_per_atom=1.0, weighting="atom")

    assert result.residuals == (0.0, 2.5)
    assert result.statistics.count == 2
    assert result.statistics.bias == pytest.approx(2.0)
    assert result.statistics.mae == pytest.approx(2.0)
    assert result.statistics.rmse == pytest.approx(5.0**0.5)
    assert result.statistics.percentile95_absolute_error == pytest.approx(2.375)
    assert result.corrected_residuals == (-1.0, 1.5)
    assert result.corrected_statistics is not None
    assert result.corrected_statistics.bias == pytest.approx(1.0)
    assert result.weighting == "atom"


def test_energy_errors_weight_configurations_equally_by_default() -> None:
    result = energy_errors([0.0, 0.0], [0.0, 10.0], atom_counts=[1, 4])

    assert result.statistics.bias == pytest.approx(1.25)
    assert result.statistics.rmse == pytest.approx(3.125**0.5)
    assert result.weighting == "configuration"


def test_energy_difference_uses_scaled_fallback_when_total_subtraction_overflows() -> None:
    result = energy_errors([1.7e308], [-1.7e308], atom_counts=[4])

    assert result.residuals == pytest.approx((-8.5e307,))


def test_energy_errors_do_not_calibrate_without_offset() -> None:
    result = energy_errors([10.0, 20.0], [12.0, 22.0], atom_counts=[2, 2])

    assert result.residuals == (1.0, 1.0)
    assert result.offset_per_atom is None
    assert result.corrected_residuals is None
    assert result.corrected_statistics is None


def test_force_errors_report_components_vectors_configurations_and_species() -> None:
    reference = [np.zeros((1, 3)), np.zeros((3, 3))]
    predicted = [np.array([[3.0, 4.0, 0.0]]), np.zeros((3, 3))]
    result = force_errors(reference, predicted, species=[["Si"], ["O", "Si", "O"]])

    assert result.component_statistics[0].bias == pytest.approx(0.75)
    assert result.component_statistics[1].rmse == pytest.approx(2.0)
    assert result.mean_vector_error == pytest.approx(1.25)
    assert result.species == (("Si",), ("O", "Si", "O"))
    assert result.rms_vector_error == pytest.approx(2.5)
    assert result.per_configuration_mean_vector_error == pytest.approx((5.0, 0.0))
    species = dict(result.per_species_component_statistics)
    assert species["Si"][0].bias == pytest.approx(1.5)
    assert species["Si"][1].bias == pytest.approx(2.0)
    assert species["O"][0].bias == 0.0


def test_force_configuration_weighting_changes_aggregate_weight() -> None:
    reference = [np.zeros((1, 3)), np.zeros((3, 3))]
    predicted = [np.array([[10.0, 0.0, 0.0]]), np.zeros((3, 3))]
    species = [["A"], ["A", "A", "A"]]

    atom = force_errors(reference, predicted, species=species, weighting="atom")
    configuration = force_errors(reference, predicted, species=species, weighting="configuration")

    assert atom.component_statistics[0].bias == pytest.approx(2.5)
    assert configuration.component_statistics[0].bias == pytest.approx(5.0)
    assert configuration.per_species_component_statistics[0][1][0].bias == pytest.approx(5.0)


def test_per_configuration_vector_metrics_average_multiple_atoms() -> None:
    result = force_errors(
        [[[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]],
        [[[3.0, 0.0, 0.0], [0.0, 4.0, 0.0]]],
        species=[["A", "B"]],
    )

    assert result.per_configuration_mean_vector_error == pytest.approx((3.5,))
    assert result.per_configuration_rms_vector_error == pytest.approx((12.5**0.5,))


def test_stress_errors_use_tensile_residuals_and_documented_shear_order() -> None:
    ref = np.zeros((1, 3, 3))
    pred = np.array([[[1.0, 2.0, 3.0], [2.0, 4.0, 5.0], [3.0, 5.0, 6.0]]])
    result = stress_errors(ref, pred)

    assert result.residuals == ((1.0, 4.0, 6.0, 5.0, 3.0, 2.0),)
    assert tuple(stat.bias for stat in result.component_statistics) == (1.0, 4.0, 6.0, 5.0, 3.0, 2.0)


@pytest.mark.parametrize(
    "call",
    [
        lambda: energy_errors([1.0], [2.0, 3.0], atom_counts=[1]),
        lambda: energy_errors([1.0], [2.0], atom_counts=[0]),
        lambda: energy_errors([1.0], [2.0], atom_counts=[True]),
        lambda: energy_errors([1.0], [2.0], atom_counts=[10**400]),
        lambda: energy_errors([1.0], [2.0], atom_counts=[1], weighting="bad"),
        lambda: energy_errors([1.0], [complex(2, 1)], atom_counts=[1]),
        lambda: energy_errors(["1.0"], [2.0], atom_counts=[1]),
        lambda: energy_errors([1.0], [float("inf")], atom_counts=[1]),
        lambda: force_errors([[[1.0, 2.0]]], [[[1.0, 2.0]]], species=[["Si"]]),
        lambda: force_errors([[[1.0, 2.0, 3.0]]], [[[1.0, 2.0, 3.0]]], species=[["Si", "O"]]),
        lambda: force_errors([[[1.0, 2.0, 3.0]]], [[[1.0, 2.0, 3.0]]], species=[[""]]),
        lambda: force_errors([np.zeros((2, 3))], [np.zeros((2, 3))], species=["Si"]),
        lambda: force_errors([np.zeros((1, 3))], [np.zeros((1, 3))], species="H"),
        lambda: force_errors([np.zeros((1, 3))], [np.zeros((1, 3))], species=None),
        lambda: force_errors([[[1.0, 2.0, 3.0]]], [[[1.0, 2.0, 3.0]]], species=[["Si"]], weighting="bad"),
        lambda: stress_errors(np.zeros((1, 3, 3)), np.zeros((3, 3))),
        lambda: stress_errors(np.zeros((1, 3, 3)), np.array([[[0, 1, 0], [0, 0, 0], [0, 0, 0]]])),
        lambda: stress_errors(np.zeros((1, 3, 3)), np.full((1, 3, 3), complex(1, 1))),
    ],
)
def test_invalid_inputs_raise_value_error(call: object) -> None:
    with pytest.raises(ValueError):
        call()  # type: ignore[operator]


def test_nested_results_are_immutable_and_inputs_are_unchanged() -> None:
    ref = np.zeros((1, 1, 3))
    pred = np.array([[[1.0, 0.0, 0.0]]])
    ref_before = ref.copy()
    pred_before = pred.copy()
    result = force_errors(ref, pred, species=[["Si"]])

    np.testing.assert_array_equal(ref, ref_before)
    np.testing.assert_array_equal(pred, pred_before)
    assert isinstance(result.residuals, tuple)
    assert isinstance(result.residuals[0], tuple)
    assert isinstance(result.residuals[0][0], tuple)
    with pytest.raises(FrozenInstanceError):
        result.weighting = "configuration"  # type: ignore[misc]


@pytest.mark.parametrize("scale", [1e-250, 1e-150, 1e150, 1e250])
def test_force_statistics_remain_finite_at_small_and_large_scales(scale: float) -> None:
    result = force_errors(
        [[[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]],
        [[[scale, 0.0, 0.0], [-scale, 0.0, 0.0]]],
        species=[["A", "A"]],
    )

    assert result.component_statistics[0].bias == pytest.approx(0.0)
    assert result.component_statistics[0].rmse == pytest.approx(scale)
    assert result.rms_vector_error == pytest.approx(scale)


_DERIVATIONS = (defs.RMSE, defs.MAE, defs.BIAS, defs.MAXIMUM_ABSOLUTE_ERROR)


def test_energy_error_statistics_bind_raw_only() -> None:
    result = energy_errors([0.0, 0.0], [0.0, 10.0], atom_counts=[1, 4], offset_per_atom=1.0)
    bound = bound_values(result)
    assert tuple(b.binding.derivation for b in bound) == _DERIVATIONS
    assert {b.binding.definition for b in bound} == {defs.TOTAL_ENERGY_PER_ATOM}
    assert next(b.field for b in bound) == "statistics.rmse"
    assert [b.value for b in bound] == pytest.approx([3.125**0.5, 1.25, 1.25, 2.5])
    for b in bound:
        load_property_definition(b.binding.definition).check(b.value)
    made = records(result)
    assert all(isinstance(r, DerivedDataRecord) for r in made)
    assert (made[0].definition_id, made[0].derivation) == (defs.TOTAL_ENERGY_PER_ATOM, defs.RMSE)


def test_stress_error_statistics_bind_voigt_lists() -> None:
    pred = np.array([[[1.0, 2.0, 3.0], [2.0, 4.0, 5.0], [3.0, 5.0, 6.0]]])
    bound = bound_values(stress_errors(np.zeros((1, 3, 3)), pred))
    assert tuple(b.binding.derivation for b in bound) == _DERIVATIONS
    assert {b.binding.definition for b in bound} == {STRESS_TENSOR}
    assert bound[2].value == [1.0, 4.0, 6.0, 5.0, 3.0, 2.0]  # bias, Voigt xx yy zz yz xz xy
    for b in bound:
        load_property_definition(b.binding.definition).check(b.value)


def test_statistics_bind_only_for_their_natural_population(caplog) -> None:
    with caplog.at_level("WARNING", logger="httk.analyse.records"):
        assert records(energy_errors([0.0], [1.0], atom_counts=[1], weighting="atom")) == ()
    assert "EnergyErrors binds no property values" in caplog.text
    assert bound_values(energy_errors([0.0, 0.0], [0.0, 10.0], atom_counts=[1, 4], weighting="atom")) == ()
    forces = ([[[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]], [[[1.0, 2.0, 3.0], [-1.0, 0.0, 0.0]]])
    assert bound_values(force_errors(*forces, species=[["Si", "Si"]], weighting="configuration")) == ()
    result = force_errors(*forces, species=[["Si", "Si"]], weighting="atom")
    made = records(result)
    assert len(made) == 4 and all(isinstance(r, DerivedDataRecord) for r in made)
    assert {r.definition_id for r in made} == {ATOMIC_FORCE}
    assert tuple(r.derivation for r in made) == _DERIVATIONS
    bound = bound_values(result)
    assert bound[2].value == pytest.approx([0.0, 1.0, 1.5])  # bias
    assert all(len(b.value) == 3 for b in bound)
