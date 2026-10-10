"""Analysis results bind to property definitions and become data records."""

import dataclasses
import json
from importlib.util import find_spec

import numpy as np
import pytest
from httk.core import DataRecord, DerivedDataRecord, RunEdge, TotalEnergyRecord, load_property_definition
from httk.core.definition_ids import TEMPERATURE, TOTAL_ENERGY, VOLUME
from httk.core.property_records import RECORD_KINDS as CORE_KINDS
from httk.core.property_records import TemperatureRecord, VolumeRecord

from httk.analyse import definitions as defs
from httk.analyse._constants import M2_PER_S_PER_A2_PER_PS
from httk.analyse.integrations.vasp import VaspDOS
from httk.analyse.matsci import (
    ElasticTensor,
    PhaseDiagram,
    RadialDistribution,
    TensorSeries,
    band_edges,
    chemical_potential_region,
    diffusion_from_msd,
    energy_drift,
    equilibrium_response,
    fit_birch_murnaghan,
    fit_eos,
    fit_stress_strain,
    formation_energy,
    harmonic_thermodynamics,
    integrate_vacf,
    mean_squared_displacement,
    mode_gruneisen,
    property_parity,
    quasiharmonic,
    radial_distribution,
    replica_transport,
    thermal_conductivity,
    velocity_autocorrelation,
    viscosity,
)
from httk.analyse.matsci.mlip import energy_errors, force_errors, stress_errors
from httk.analyse.property_records import DERIVED_RECORD_KINDS as DERIVED_KINDS
from httk.analyse.property_records import RECORD_KINDS as ANALYSE_KINDS
from httk.analyse.property_records import ThermalConductivityTensorStandardErrorRecord, TotalEnergyRmseRecord
from httk.analyse.records import bound_values, records

# Nonlinear EOS fits and bond orders need SciPy, in the [ci] extra; those tests skip without it.
NEEDS_SCIPY = pytest.mark.skipif(find_spec("scipy") is None, reason="needs the httk-analyse[scipy] extra")


def _bm3(volumes, v0=10.0, e0=-5.0, b0=0.2, bp=4.5):
    q = (v0 / np.asarray(volumes)) ** (2 / 3) - 1
    return e0 + 9 * v0 * b0 / 16 * (2 * q**2 + (bp - 4) * q**3)


VOLUMES = np.linspace(8.5, 11.5, 9)


def _cubic():
    c11, c12, c44 = 250.0, 120.0, 80.0
    return ElasticTensor(
        [[c11 if i == j else c12 for j in range(3)] + [0.0] * 3 for i in range(3)]
        + [[0.0] * 3 + [c44 if i == j else 0.0 for j in range(3)] for i in range(3)]
    )


def _msd():
    times = np.linspace(0, 10, 21)
    tensors = 2 * times[:, None, None] * np.diag([1.0, 2.0, 3.0])
    return diffusion_from_msd(TensorSeries(tuple(times), tensors, (50,) * 21, "msd"), window=(3, 16))


def _heat(seed=0):
    return thermal_conductivity(
        np.random.default_rng(seed).normal(size=(40, 3)), 0.01, temperature=300, volume=125, max_lag=6
    )


def _stress(seed=1):
    noise = np.random.default_rng(seed).normal(size=(30, 3, 3))
    return viscosity(noise + noise.swapaxes(1, 2), 0.1, temperature=100, volume=10, max_lag=3)


def _harmonic():
    return harmonic_thermodynamics([2.0, 4.0], [0.0, 300.0, 600.0])


def _qha():
    frequencies = [[2.0 * (v / 10.0) ** -2, 4.0] for v in VOLUMES]
    return quasiharmonic(VOLUMES, _bm3(VOLUMES), frequencies, [0.0, 300.0, 600.0])


def _npt():
    h = [1.0, 2.0, 1.5, 2.5, 1.2, 2.2, 1.7, 2.6]
    v = [10.0, 10.4, 10.1, 10.6, 10.0, 10.5, 10.2, 10.7]
    return equilibrium_response(temperature=100, ensemble="NPT", enthalpies=h, volumes=v, block_size=4)


def _velocities():
    return np.random.default_rng(2).normal(size=(12, 2, 3))


def _rdf():
    positions = [[0.0, 0.0, 0.0], [1.2, 0.0, 0.0], [0.0, 1.7, 0.0], [1.2, 1.7, 0.0]]
    return radial_distribution([positions], np.eye(3) * 5.0, [0.0, 1.5, 2.0], species="ABAB", pair=("A", "B"))


def _dos(spin_basis="spin-summed"):
    return VaspDOS((-1.0, 0.0, 1.0), (0.5, 2.0, 0.5), (0.0, 1.0, 2.0), 0.1, spin_basis, "total")


# name: (result factory, selection keywords, number of derivation-qualified bindings)
CASES = {
    "birch_murnaghan": (lambda: fit_birch_murnaghan(VOLUMES, _bm3(VOLUMES)), {}, 1),
    "eos": (lambda: fit_eos(VOLUMES, _bm3(VOLUMES), model="vinet"), {}, 1),
    "elastic": (_cubic, {}, 0),
    "formation": (lambda: formation_energy(-10.0, {"A": 2, "B": 1}, {"A": -3.0, "B": -2.0}), {}, 0),
    "insulator": (
        lambda: band_edges(((-1.0, -0.8), (0.5, 0.3)), ((2, 2), (0, 0)), maximum_occupation=2, energy_reference=0),
        {},
        0,
    ),
    "npt": (_npt, {}, 3),
    "harmonic": (_harmonic, {}, 0),
    "qha": (_qha, {}, 0),
    "diffusion": (_msd, {}, 0),
    "msd": (lambda: mean_squared_displacement(np.cumsum(_velocities(), axis=0), 0.1, max_lag=4), {}, 0),
    "vacf": (lambda: velocity_autocorrelation(_velocities(), 0.1, max_lag=4), {}, 0),
    "vacf_integral": (lambda: integrate_vacf(velocity_autocorrelation(_velocities(), 0.1, max_lag=4)), {}, 0),
    "rdf": (_rdf, {}, 0),
    "dos": (_dos, {}, 0),
    "conductivity_series": (_heat, {}, 0),
    "conductivity": (_heat, {"lag_index": 4}, 0),
    "viscosity": (_stress, {"lag_index": 2}, 0),
    "replica": (lambda: replica_transport([_heat(0), _heat(1)]), {"lag_index": 4}, 3),
    "nvt": (
        lambda: equilibrium_response(
            temperature=100, ensemble="NVT", energies=[1.0, 2.0, 1.5, 2.5, 1.2, 2.2, 1.7, 2.6], block_size=4
        ),
        {},
        1,
    ),
    "energy_errors": (lambda: energy_errors([0.0, 0.0], [1.0, 3.0], atom_counts=[1, 1]), {}, 4),
    "force_errors": (
        lambda: force_errors([np.zeros((1, 3))], [np.array([[3.0, 4.0, 0.0]])], species=[["Si"]]),
        {},
        4,
    ),
    "stress_errors": (lambda: stress_errors(np.zeros((1, 3, 3)), np.eye(3)[None] * 2.0), {}, 4),
    "replica_viscosity": (lambda: replica_transport([_stress(1), _stress(2)]), {"lag_index": 2}, 1),
    "parity": (lambda: property_parity([1.0, 2.0], [1.5, 1.0], labels=["a", "b"], definition=defs.BULK_MODULUS), {}, 4),
}


def _name(record):
    """The (base) property name of a record."""
    return load_property_definition(record.definition_id).name


def _statistic(record):
    return getattr(record, "derivation", None) is not None


def _json(value):
    return json.loads(json.dumps(value))


@pytest.mark.parametrize(
    "case", [pytest.param(case, marks=NEEDS_SCIPY) if case == "eos" else case for case in sorted(CASES)]
)
def test_every_bound_value_is_a_checked_record(case):
    factory, selection, derived = CASES[case]
    result = factory()
    bound = bound_values(result, **selection)
    made = records(result, **selection)
    assert len(made) == len(bound)
    assert sum(_statistic(record) for record in made) == derived
    for value, record in zip(bound, made, strict=True):
        definition = load_property_definition(value.binding.definition)
        definition.check(value.value)
        assert _name(record) == definition.name
        assert _json(record.value) == _json(value.value)
        if value.binding.derivation is None:
            iri = value.binding.definition
            assert type(record) is (ANALYSE_KINDS.get(iri) or CORE_KINDS[iri])
            assert record.definition_id == iri
        else:
            key = (value.binding.definition, value.binding.derivation)
            # Only a caller-chosen base (PropertyParity) has no generated statistic kind.
            assert type(record) is (DerivedDataRecord if case == "parity" else DERIVED_KINDS[key])
            assert (record.definition_id, record.derivation) == key


@NEEDS_SCIPY
def test_cases_emit_every_generated_statistic_kind():
    emitted = {
        (record.definition_id, record.derivation)
        for name, (factory, selection, _) in CASES.items()
        if name != "parity"
        for record in records(factory(), **selection)
        if _statistic(record)
    }
    assert emitted == set(DERIVED_KINDS)


def test_core_definitions_yield_core_and_hand_written_records():
    made = records(_heat(), lag_index=4)
    assert [type(r) for r in made[-2:]] == [TemperatureRecord, VolumeRecord]
    assert (made[-2].value, made[-1].value) == (300.0, 125.0)

    class Energy:
        def _bound_values(self):
            return (
                defs.BoundValue("energy", defs.FieldBinding(TOTAL_ENERGY), -3),
                defs.BoundValue("bulk_modulus", defs.FieldBinding(defs.BULK_MODULUS), None),
            )

    energy, unknown = records(Energy(), product_of=[RunEdge("source", "runs", "run-1")])
    assert type(unknown) is DataRecord and unknown.value is None  # A null value has no typed record.
    assert type(energy) is TotalEnergyRecord
    assert energy.total_energy == -3.0
    assert energy.product_of[0].label == "source"


@NEEDS_SCIPY
def test_every_binding_definition_resolves():
    for factory, selection, _ in CASES.values():
        for value in bound_values(factory(), **selection):
            assert load_property_definition(value.binding.definition).definition_id == value.binding.definition
            assert value.binding.derivation in (
                None,
                *(
                    getattr(defs, n)
                    for n in ("MEAN", "STANDARD_ERROR", "RMSE", "MAE", "BIAS", "MAXIMUM_ABSOLUTE_ERROR")
                ),
            )


def _by_field(result, **selection):
    return {b.field: b for b in bound_values(result, **selection)}


def test_eos_values_equal_fields():
    fit = fit_birch_murnaghan(VOLUMES, _bm3(VOLUMES))
    bound = _by_field(fit)
    assert bound["bulk_modulus"].value == fit.bulk_modulus
    assert bound["bulk_modulus"].binding.definition == defs.BULK_MODULUS
    assert bound["rmse"].binding == defs.FieldBinding(TOTAL_ENERGY, derivation=defs.RMSE)
    made = records(fit)
    assert [_name(r) for r in made] == [
        "equilibrium_volume",
        "equilibrium_energy",
        "bulk_modulus",
        "bulk_modulus_pressure_derivative",
        "total_energy",
    ]
    assert type(made[-1]) is TotalEnergyRmseRecord
    assert (made[-1].definition_id, made[-1].derivation) == (TOTAL_ENERGY, defs.RMSE)
    assert made[-1].value == fit.rmse


def test_elastic_tensor_records():
    tensor = _cubic()
    made = {_name(r): r.value for r in records(tensor)}
    assert made["elastic_tensor"] == [list(row) for row in tensor.stiffness]
    assert made["bulk_modulus_voigt"] == pytest.approx(490 / 3)
    assert made["universal_anisotropy_index"] == pytest.approx(tensor.universal_anisotropy)
    assert len(made) == 9


def test_metal_gaps_are_zero():
    metal = band_edges(((-1, -1), (0, 0), (1, 1)), ((1, 0.5), (0, 0), (0, 0)), maximum_occupation=1, energy_reference=0)
    assert metal.metallic
    assert {_name(r): r.value for r in records(metal)} == {"band_gap": 0.0, "direct_band_gap": 0.0}


def test_equilibrium_response_names_and_standard_errors():
    result = _npt()
    bound = _by_field(result)
    assert bound["temperature"].binding.definition == TEMPERATURE
    assert bound["standard_errors.heat_capacity_constant_pressure"].binding == defs.FieldBinding(
        defs.HEAT_CAPACITY_CONSTANT_PRESSURE, derivation=defs.STANDARD_ERROR
    )
    made = records(result)
    assert [_name(r) for r in made if not _statistic(r)] == [*result.names, "temperature"]
    assert [_name(r) for r in made if _statistic(r)] == list(result.names)


def _record_values(result, **selection):
    return {_name(r): r.value for r in records(result, **selection) if not _statistic(r)}


def test_harmonic_series_is_one_dictionary_record():
    result = _harmonic()
    made = _record_values(result)
    assert list(made) == ["zero_point_energy", "vibrational_thermodynamics"]
    assert made["vibrational_thermodynamics"] == {
        "temperatures": list(result.temperatures),
        "helmholtz_free_energies": list(result.free_energy),
        "internal_energies": list(result.internal_energy),
        "entropies": list(result.entropy),
        "heat_capacities": list(result.heat_capacity),
    }


def test_quasiharmonic_series_carries_total_free_energies():
    result = _qha()
    made = _record_values(result)
    assert list(made) == ["quasiharmonic_thermodynamics"]
    series = made["quasiharmonic_thermodynamics"]
    assert series["total_helmholtz_free_energies"] == list(result.free_energies)
    assert series["bulk_moduli"] == list(result.bulk_moduli)
    assert series["volumetric_thermal_expansions"] == list(result.volumetric_expansion)


def test_tensor_series_bind_by_kind():
    vacf = velocity_autocorrelation(_velocities(), 0.1, max_lag=4)
    assert _record_values(vacf)["velocity_autocorrelation"]["origin_counts"] == list(vacf.counts)
    integral = integrate_vacf(vacf)
    series = _record_values(integral)["diffusion_running_integral"]
    assert set(series) == {"lag_times", "diffusion_tensors"}
    np.testing.assert_allclose(series["diffusion_tensors"], np.asarray(integral.tensors) * M2_PER_S_PER_A2_PER_PS)


def test_rdf_series_and_edge_count():
    result = _rdf()
    series = _record_values(result)["radial_distribution_function"]
    assert series == {"bin_edges": list(result.edges), "g": list(result.g), "pair": ["A", "B"]}
    total = RadialDistribution(result.edges, result.centers, result.g, result.hist_counts, 1, 0.0)
    assert "pair" not in _record_values(total)["radial_distribution_function"]
    broken = RadialDistribution(result.edges[:-1], result.centers, result.g, result.hist_counts, 1, 0.0)
    with pytest.raises(ValueError, match="one more bin edge"):
        records(broken)


def test_dos_binds_spin_summed_and_spin_channel_data():
    made = _record_values(_dos())
    assert made["electronic_density_of_states"]["density"] == [0.5, 2.0, 0.5]
    assert made["fermi_energy"] == 0.1
    channel = _record_values(_dos("up"))
    assert list(channel) == ["spin_channel_electronic_density_of_states", "fermi_energy"]
    assert channel["spin_channel_electronic_density_of_states"]["spin"] == "up"
    assert channel["spin_channel_electronic_density_of_states"]["density"] == [0.5, 2.0, 0.5]
    assert channel["fermi_energy"] == 0.1
    down = _record_values(_dos("down"))
    assert down["spin_channel_electronic_density_of_states"]["spin"] == "down"
    assert down["fermi_energy"] == 0.1


def test_gruneisen_fit_binds_exact_degree():
    volumes = [9.0, 10.0, 11.0]
    result = mode_gruneisen(volumes, [[2.0 * (v / 10.0) ** -2, 4.0] for v in volumes], reference_volume=10.0)
    (value,) = bound_values(result)
    assert list(value.value) == ["reference_volume", "values", "degree"]
    assert type(value.value["degree"]) is int and value.value["degree"] == 2
    assert value.value["values"] == list(result.values)
    assert len(records(result)) == 1


def test_energy_drift_binds_exact_sample_count():
    time = np.linspace(0.0, 10.0, 11)
    result = energy_drift(time, (-3 + 0.01 * time) * 20, atom_count=20, ensemble="NVE")
    (value,) = bound_values(result)
    assert type(value.value["samples"]) is int and value.value["samples"] == 11
    assert value.value["slope"] == pytest.approx(0.01)
    assert value.value["stop"] == 10.0
    assert len(records(result)) == 1


def test_chemical_potential_region_binds_with_and_without_competitors():
    region = chemical_potential_region({"A": 1, "B": 2}, -9.0, [{"A": 1}, {"B": 1, "C": 1}], [-2.0, -5.0])
    (value,) = bound_values(region)
    assert value.value == {
        "elements": ["A", "B", "C"],
        "host_coefficients": [1.0, 2.0, 0.0],
        "host_energy": -9.0,
        "competing_coefficients": [[1.0, 0.0, 0.0], [0.0, 1.0, 1.0]],
        "competing_energies": [-2.0, -5.0],
    }
    assert len(records(region)) == 1
    alone = records(chemical_potential_region({"A": 1}, -2.0, [], []))
    assert alone[0].value["competing_coefficients"] == []


def test_elastic_fit_delegates_to_its_tensor():
    strains = np.random.default_rng(3).normal(scale=0.01, size=(20, 6))
    tensor = _cubic()
    fit = fit_stress_strain(strains, strains @ np.asarray(tensor.stiffness).T)
    assert [r.value for r in records(fit)] == [r.value for r in records(fit.tensor)]
    with pytest.raises(TypeError):
        records(fit, lag_index=1)


def test_phase_diagram_unknown_energy_element_widens_elements():
    diagram = PhaseDiagram.from_compositions(
        [{"A": 1}, {"B": 1}, {"A": 1, "B": 1}, {"C": 1, "A": 1}], [0.0, 0.0, -1.0, None], ids=["A", "B", "AB", "CA"]
    )
    (value,) = bound_values(diagram)
    assert value.value["elements"] == ["A", "B", "C"]
    assert value.value["phase_ids"] == ["A", "B", "AB"]
    assert value.value["compositions"] == [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.5, 0.5, 0.0]]
    assert len(records(diagram)) == 1


def test_phase_diagram_binds_hull_distances_and_tolerance():
    diagram = PhaseDiagram.from_compositions(
        [{"A": 1}, {"B": 1}, {"A": 1, "B": 1}, {"A": 1, "B": 1}, {"A": 1, "B": 3}, {"A": 2, "B": 2}],
        [0.0, 0.0, -2.0, -1.0, None, -4.0 + 0.0005],
        ids=["A", "B", "AB", "AB2", "AB3?", "AB-near"],
        tolerance=1e-3,
    )
    (value,) = bound_values(diagram)
    data = value.value
    assert data["phase_ids"] == ["A", "B", "AB", "AB2", "AB-near"]
    assert data["elements"] == ["A", "B"]
    assert data["compositions"] == [[1.0, 0.0], [0.0, 1.0], [0.5, 0.5], [0.5, 0.5], [0.5, 0.5]]
    assert data["energies_per_atom"] == pytest.approx([0.0, 0.0, -1.0, -0.5, -1.0 + 0.000125])
    assert data["stable"] == [True, True, True, False, True]
    assert type(data["stable"][0]) is bool
    assert data["energies_above_hull_per_atom"] == pytest.approx([0.0, 0.0, 0.0, 0.5, 0.000125], abs=1e-12)
    assert data["tolerance"] == 1e-3
    assert len(records(diagram)) == 1


def test_transport_series_and_optional_plateau():
    result = _heat()
    series_only = _record_values(result)
    assert list(series_only) == ["thermal_conductivity_running_integral", "temperature", "volume"]
    tensors = series_only["thermal_conductivity_running_integral"]["thermal_conductivity_tensors"]
    assert np.ravel(tensors[4]).tolist() == list(result.integrals[4])
    made = _record_values(result, lag_index=4)
    assert list(made) == [
        "thermal_conductivity_running_integral",
        "thermal_conductivity_tensor",
        "thermal_conductivity",
        "temperature",
        "volume",
    ]
    assert made["thermal_conductivity"] == pytest.approx(result.isotropic[4])
    assert np.ravel(made["thermal_conductivity_tensor"]).tolist() == list(result.integrals[4])
    assert made["temperature"] == 300.0
    assert _by_field(result, lag_index=4)["volume"].binding.definition == VOLUME
    shear = _stress()
    made = _record_values(shear, lag_index=2)
    assert made["shear_viscosity"] == pytest.approx(shear.isotropic[2])
    assert made["shear_viscosity_running_integral"]["shear_viscosities"] == [list(row) for row in shear.integrals]
    with pytest.raises(ValueError):
        records(result, lag_index=7)
    with pytest.raises(TypeError):
        records(result, lag_index=1, plateau=3)


@pytest.mark.parametrize("factory", (_heat, _stress))
def test_transport_series_rejects_permuted_components(factory):
    result = factory()
    permuted = dataclasses.replace(
        result, components=(result.components[1], result.components[0], *result.components[2:])
    )
    with pytest.raises(ValueError, match="components"):
        records(permuted)


def test_replica_bindings_are_derived_records():
    result = replica_transport([_heat(0), _heat(1)])
    bound = _by_field(result, lag_index=4)
    assert bound["mean[4]"].binding == defs.FieldBinding(defs.THERMAL_CONDUCTIVITY_TENSOR, defs.MEAN)
    assert bound["mean.isotropic[4]"].binding == defs.FieldBinding(defs.THERMAL_CONDUCTIVITY, defs.MEAN)
    assert bound["standard_error[4]"].binding.derivation == defs.STANDARD_ERROR
    assert "standard_error.isotropic[4]" not in bound
    error = records(result, lag_index=4)[-1]
    assert type(error) is ThermalConductivityTensorStandardErrorRecord
    assert (error.definition_id, error.derivation, _name(error)) == (
        defs.THERMAL_CONDUCTIVITY_TENSOR,
        defs.STANDARD_ERROR,
        "thermal_conductivity_tensor",
    )
    assert np.ravel(error.value).tolist() == list(result.standard_error[4])
    with pytest.raises(TypeError):
        records(result)


def test_parity_without_definition_binds_nothing():
    assert bound_values(property_parity([1.0], [2.0], labels=["a"], definition=None)) == ()


def test_unsupported_results_and_keywords_raise():
    with pytest.raises(TypeError, match="object"):
        records(object())
    with pytest.raises(TypeError):
        records(_cubic(), lag_index=1)


def _checked(result):
    (value,) = bound_values(result)
    load_property_definition(value.binding.definition).check(value.value)
    (record,) = records(result)
    assert record.value == value.value
    return value.binding.definition, value.value


def test_dynamics_series_bind_and_check():
    from dataclasses import replace

    from httk.analyse.matsci import (
        intermediate_scattering,
        van_hove_distinct,
        van_hove_self,
        velocity_spectrum,
    )

    positions = np.cumsum(np.random.default_rng(3).normal(size=(8, 3, 3)), axis=0) * 0.1
    q = [[0, 0, 0], [2, 0, 0]]
    for kind, definition in (
        ("self", defs.SELF_INTERMEDIATE_SCATTERING_FUNCTION),
        ("coherent", defs.INTERMEDIATE_SCATTERING_FUNCTION),
    ):
        result = intermediate_scattering(positions, 0.5, q, kind=kind, max_lag=3)
        found, value = _checked(result)
        assert found == definition
        np.testing.assert_array_equal(np.array(value["real"]) + 1j * np.array(value["imaginary"]), result.values)
        assert value["origin_counts"] == list(result.counts)
    hove = van_hove_self(positions, 0.2, lag=1, bins=[0, 1, 2])
    assert _checked(hove)[0] == defs.SELF_VAN_HOVE_FUNCTION
    far = van_hove_distinct(positions, 0.2, lag=1, bins=[0, 1, 2], cell=np.eye(3) * 9)
    found, value = _checked(far)
    assert found == defs.DISTINCT_VAN_HOVE_FUNCTION
    assert value["lag_time"] == far.time and value["samples"] == far.samples
    with pytest.raises(ValueError, match="one more bin edge"):
        records(replace(hove, edges=hove.edges[:-1]))
    spectrum = velocity_spectrum(np.random.default_rng(4).normal(size=(16, 2, 3)), 0.2, remove_mean=True)
    found, value = _checked(spectrum)
    assert found == defs.VELOCITY_POWER_SPECTRUM
    assert value["mean_removed"] is True and value["window"] == spectrum.window


@NEEDS_SCIPY
def test_bond_order_binds_cutoff_and_none_entries():
    from httk.analyse.matsci.local_order import bond_order

    result = bond_order([[0, 0, 0], [0.5, 0, 0], [5, 5, 5]], np.eye(3) * 10, 1, 2)
    found, value = _checked(result)
    assert found == defs.STEINHARDT_BOND_ORDER
    assert value["cutoff"] == 1.0 and value["local_orders"][2] is None
    assert value["coordination_numbers"] == [1, 1, 0]
    alone = _checked(bond_order([[0, 0, 0]], np.eye(3) * 10, 1, 6))[1]
    assert alone["global_order"] is None and alone["local_orders"] == [None]


def test_dynamics_reject_bad_kind_and_window():
    from dataclasses import replace

    from httk.analyse.matsci import intermediate_scattering, van_hove_self, velocity_spectrum

    positions = np.random.default_rng(5).normal(size=(6, 2, 3))
    with pytest.raises(ValueError, match="kind"):
        replace(van_hove_self(positions, 0.2, lag=1, bins=[0, 1, 2]), kind="bogus")
    with pytest.raises(ValueError, match="kind"):
        replace(intermediate_scattering(positions, 0.5, [[0, 0, 0]], kind="self", max_lag=2), kind="bogus")
    with pytest.raises(ValueError, match="window"):
        replace(velocity_spectrum(positions, 0.2), window="bogus")


def test_recorded_spectrum_integrates_to_mean_square():
    from httk.analyse.matsci import velocity_spectrum

    spectrum = velocity_spectrum(np.random.default_rng(6).normal(size=(16, 2, 3)), 0.2, window="hann")
    value = _record_values(spectrum)["velocity_power_spectrum"]
    assert sum(value["power"]) * spectrum.frequency_spacing == pytest.approx(value["mean_square"])
