"""Analysis results bind to property definitions and become data records."""

import dataclasses
import json

import numpy as np
import pytest
from httk.core import DataRecord, DerivedDataRecord, load_property_definition
from httk.core.definition_ids import TEMPERATURE, TOTAL_ENERGY, VOLUME

from httk.analyse import definitions as defs
from httk.analyse._constants import M2_PER_S_PER_A2_PER_PS
from httk.analyse.integrations.vasp import VaspDOS
from httk.analyse.matsci import (
    ElasticTensor,
    RadialDistribution,
    TensorSeries,
    band_edges,
    diffusion_from_msd,
    equilibrium_response,
    fit_birch_murnaghan,
    fit_eos,
    formation_energy,
    harmonic_thermodynamics,
    integrate_vacf,
    mean_squared_displacement,
    property_parity,
    quasiharmonic,
    radial_distribution,
    replica_transport,
    thermal_conductivity,
    velocity_autocorrelation,
    viscosity,
)
from httk.analyse.records import bound_values, records


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


def _stress():
    noise = np.random.default_rng(1).normal(size=(30, 3, 3))
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
    "parity": (lambda: property_parity([1.0, 2.0], [1.5, 1.0], labels=["a", "b"], definition=defs.BULK_MODULUS), {}, 4),
}


@pytest.mark.parametrize("case", sorted(CASES))
def test_every_bound_value_is_a_checked_record(case):
    factory, selection, derived = CASES[case]
    result = factory()
    bound = bound_values(result, **selection)
    made = records(result, **selection)
    assert len(made) == len(bound)
    assert sum(isinstance(record, DerivedDataRecord) for record in made) == derived
    for value, record in zip(bound, made, strict=True):
        definition = load_property_definition(value.binding.definition)
        definition.check(value.value)
        assert record.definition_id == value.binding.definition
        assert record.name == definition.name
        assert json.loads(record.value_json) == value.value
        if value.binding.derivation is None:
            assert type(record) is DataRecord
        else:
            assert record.derivation == value.binding.derivation


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
    assert [r.name for r in made] == [
        "equilibrium_volume",
        "equilibrium_energy",
        "bulk_modulus",
        "bulk_modulus_pressure_derivative",
        "total_energy",
    ]
    assert isinstance(made[-1], DerivedDataRecord)
    assert (made[-1].definition_id, made[-1].derivation) == (TOTAL_ENERGY, defs.RMSE)
    assert json.loads(made[-1].value_json) == fit.rmse


def test_elastic_tensor_records():
    tensor = _cubic()
    made = {r.name: json.loads(r.value_json) for r in records(tensor)}
    assert made["elastic_tensor"] == [list(row) for row in tensor.stiffness]
    assert made["bulk_modulus_voigt"] == pytest.approx(490 / 3)
    assert made["universal_anisotropy_index"] == pytest.approx(tensor.universal_anisotropy)
    assert len(made) == 9


def test_metal_gaps_are_zero():
    metal = band_edges(((-1, -1), (0, 0), (1, 1)), ((1, 0.5), (0, 0), (0, 0)), maximum_occupation=1, energy_reference=0)
    assert metal.metallic
    assert {r.name: json.loads(r.value_json) for r in records(metal)} == {"band_gap": 0.0, "direct_band_gap": 0.0}


def test_equilibrium_response_names_and_standard_errors():
    result = _npt()
    bound = _by_field(result)
    assert bound["temperature"].binding.definition == TEMPERATURE
    assert bound["standard_errors.heat_capacity_constant_pressure"].binding == defs.FieldBinding(
        defs.HEAT_CAPACITY_CONSTANT_PRESSURE, derivation=defs.STANDARD_ERROR
    )
    made = records(result)
    assert [r.name for r in made if type(r) is DataRecord] == [*result.names, "temperature"]
    assert [r.name for r in made if type(r) is DerivedDataRecord] == list(result.names)


def _record_values(result, **selection):
    return {r.name: json.loads(r.value_json) for r in records(result, **selection) if type(r) is DataRecord}


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


def test_dos_binds_only_spin_summed_data():
    made = _record_values(_dos())
    assert made["electronic_density_of_states"]["density"] == [0.5, 2.0, 0.5]
    assert made["fermi_energy"] == 0.1
    with pytest.raises(ValueError, match="spin-summed"):
        records(_dos("up"))


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
    assert isinstance(error, DerivedDataRecord)
    assert (error.definition_id, error.derivation, error.name) == (
        defs.THERMAL_CONDUCTIVITY_TENSOR,
        defs.STANDARD_ERROR,
        "thermal_conductivity_tensor",
    )
    assert np.ravel(json.loads(error.value_json)).tolist() == list(result.standard_error[4])
    with pytest.raises(TypeError):
        records(result)


def test_parity_without_definition_binds_nothing():
    assert bound_values(property_parity([1.0], [2.0], labels=["a"], definition=None)) == ()


def test_unsupported_results_and_keywords_raise():
    with pytest.raises(TypeError, match="object"):
        records(object())
    with pytest.raises(TypeError):
        records(_cubic(), lag_index=1)
