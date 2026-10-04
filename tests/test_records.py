"""Analysis results bind to property definitions and become data records."""

import json

import numpy as np
import pytest
from httk.core import load_property_definition
from httk.core.definition_ids import TEMPERATURE, TOTAL_ENERGY, VOLUME

from httk.analyse import definitions as defs
from httk.analyse.matsci import (
    ElasticTensor,
    TensorSeries,
    band_edges,
    diffusion_from_msd,
    equilibrium_response,
    fit_birch_murnaghan,
    fit_eos,
    formation_energy,
    harmonic_thermodynamics,
    property_parity,
    quasiharmonic,
    replica_transport,
    thermal_conductivity,
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


# name: (result factory, selection keywords, number of envelope-only bindings)
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
    "harmonic": (_harmonic, {}, 5),
    "qha": (_qha, {}, 5),
    "diffusion": (_msd, {}, 0),
    "conductivity": (_heat, {"lag_index": 4}, 0),
    "viscosity": (_stress, {"lag_index": 2}, 0),
    "replica": (lambda: replica_transport([_heat(0), _heat(1)]), {"lag_index": 4}, 3),
    "parity": (lambda: property_parity([1.0, 2.0], [1.5, 1.0], labels=["a", "b"], definition=defs.BULK_MODULUS), {}, 4),
}


@pytest.mark.parametrize("case", sorted(CASES))
def test_records_are_the_checked_scalar_and_fixed_size_bindings(case):
    factory, selection, envelope_only = CASES[case]
    result = factory()
    bound = bound_values(result, **selection)
    recorded = [b for b in bound if b.binding.axis is None and b.binding.derivation is None]
    assert len(bound) - len(recorded) == envelope_only
    made = records(result, **selection)
    assert len(made) == len(recorded)
    for value, record in zip(recorded, made, strict=True):
        definition = load_property_definition(value.binding.definition)
        definition.check(value.value)
        assert record.definition_id == value.binding.definition
        assert record.name == definition.name
        assert json.loads(record.value_json) == value.value


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
            assert value.binding.axis in (None, "temperatures")


def _by_field(result, **selection):
    return {b.field: b for b in bound_values(result, **selection)}


def test_eos_values_equal_fields():
    fit = fit_birch_murnaghan(VOLUMES, _bm3(VOLUMES))
    bound = _by_field(fit)
    assert bound["bulk_modulus"].value == fit.bulk_modulus
    assert bound["bulk_modulus"].binding.definition == defs.BULK_MODULUS
    assert bound["rmse"].binding == defs.FieldBinding(TOTAL_ENERGY, derivation=defs.RMSE)
    assert [r.name for r in records(fit)] == [
        "equilibrium_volume",
        "equilibrium_energy",
        "bulk_modulus",
        "bulk_modulus_pressure_derivative",
    ]


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
    assert [r.name for r in records(result)] == [*result.names, "temperature"]


def test_harmonic_series_stay_in_envelope():
    result = _harmonic()
    assert [r.name for r in records(result)] == ["zero_point_energy"]
    assert _by_field(result)["free_energy"].binding.axis == "temperatures"
    assert _by_field(result)["free_energy"].value == list(result.free_energy)


def test_transport_lag_selection():
    result = _heat()
    made = {r.name: json.loads(r.value_json) for r in records(result, lag_index=4)}
    assert made["thermal_conductivity"] == pytest.approx(result.isotropic[4])
    assert np.ravel(made["thermal_conductivity_tensor"]).tolist() == list(result.integrals[4])
    assert made["temperature"] == 300.0
    assert _by_field(result, lag_index=4)["volume"].binding.definition == VOLUME
    shear = _stress()
    assert {r.name: json.loads(r.value_json) for r in records(shear, lag_index=2)}["shear_viscosity"] == pytest.approx(
        shear.isotropic[2]
    )
    with pytest.raises(TypeError):
        records(result)
    with pytest.raises(ValueError):
        records(result, lag_index=7)
    with pytest.raises(TypeError):
        records(result, lag_index=1, plateau=3)


def test_replica_bindings_are_derivations():
    bound = _by_field(replica_transport([_heat(0), _heat(1)]), lag_index=4)
    assert bound["mean[4]"].binding == defs.FieldBinding(defs.THERMAL_CONDUCTIVITY_TENSOR, defs.MEAN)
    assert bound["mean.isotropic[4]"].binding == defs.FieldBinding(defs.THERMAL_CONDUCTIVITY, defs.MEAN)
    assert bound["standard_error[4]"].binding.derivation == defs.STANDARD_ERROR
    assert "standard_error.isotropic[4]" not in bound


def test_parity_without_definition_binds_nothing():
    assert bound_values(property_parity([1.0], [2.0], labels=["a"], definition=None)) == ()


def test_unsupported_results_and_keywords_raise():
    with pytest.raises(TypeError, match="object"):
        records(object())
    with pytest.raises(TypeError):
        records(_cubic(), lag_index=1)
