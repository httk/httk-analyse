"""Tests for static third-order Birch–Murnaghan fitting."""

import math
from dataclasses import FrozenInstanceError

import pytest

from httk.analyse.matsci.eos import BirchMurnaghanFit, fit_birch_murnaghan


def _closed_form_bm3(
    volume: float, equilibrium_volume: float, energy0: float, bulk_modulus: float, derivative: float
) -> float:
    eta = (equilibrium_volume / volume) ** (2.0 / 3.0)
    delta = eta - 1.0
    return energy0 + 9.0 * equilibrium_volume * bulk_modulus / 16.0 * (2.0 * delta**2 + (derivative - 4.0) * delta**3)


def _closed_form_pressure(volume: float, equilibrium_volume: float, bulk_modulus: float, derivative: float) -> float:
    eta = (equilibrium_volume / volume) ** (2.0 / 3.0)
    strain = (eta - 1.0) / 2.0
    return 3.0 * bulk_modulus * strain * eta**2.5 * (1.0 + 1.5 * (derivative - 4.0) * strain)


@pytest.mark.parametrize("derivative", [4.0, 4.00001, 5.4])
def test_closed_form_bm3_recovers_parameters_and_independent_pressure(derivative: float) -> None:
    v0, e0, b0 = 18.7, -7.35, 0.72
    volumes = tuple(v0 * factor for factor in (0.82, 0.88, 0.94, 1.0, 1.06, 1.12, 1.18))
    energies = tuple(_closed_form_bm3(volume, v0, e0, b0, derivative) for volume in volumes)

    fit = fit_birch_murnaghan(volumes, energies)

    assert fit.equilibrium_volume == pytest.approx(v0, rel=2e-10)
    assert fit.equilibrium_energy == pytest.approx(e0, abs=2e-12)
    assert fit.bulk_modulus == pytest.approx(b0, rel=2e-9)
    assert fit.bulk_modulus_derivative == pytest.approx(derivative, abs=2e-7)
    assert fit.bulk_modulus_gpa == pytest.approx(b0 * 160.2176634)
    assert fit.volumes == volumes
    assert fit.energies == energies
    assert max(abs(residual) for residual in fit.residuals) < 2e-12
    assert fit.rmse < 2e-12
    assert fit.condition_number < 20.0
    for volume in (v0 * 0.93, v0, v0 * 1.07):
        assert fit.energy(volume) == pytest.approx(_closed_form_bm3(volume, v0, e0, b0, derivative), abs=3e-12)
        assert fit.pressure(volume) == pytest.approx(
            _closed_form_pressure(volume, v0, b0, derivative), rel=2e-9, abs=2e-12
        )


def test_pressure_matches_energy_finite_difference_and_pressure_sign() -> None:
    v0, e0, b0, derivative = 12.0, -3.0, 1.1, 4.8
    volumes = tuple(v0 * factor for factor in (0.8, 0.9, 1.0, 1.1, 1.2, 1.3))
    energies = tuple(_closed_form_bm3(v, v0, e0, b0, derivative) for v in volumes)
    fit = fit_birch_murnaghan(volumes, energies)
    step = v0 * 1e-5
    numeric_pressure = -(fit.energy(v0 + step) - fit.energy(v0 - step)) / (2.0 * step)

    assert fit.pressure(v0 * 0.9) > 0.0
    assert fit.pressure(v0 * 1.1) < 0.0
    assert fit.pressure(v0) == pytest.approx(0.0, abs=1e-14)
    assert fit.pressure(v0) == pytest.approx(numeric_pressure, rel=2e-7, abs=2e-10)


def test_noisy_data_retains_residuals_and_input_order() -> None:
    v0, e0, b0, derivative = 10.0, -4.0, 0.8, 5.1
    volumes = [v0 * factor for factor in (0.84, 0.9, 0.96, 1.0, 1.04, 1.1, 1.16)]
    energies = [
        _closed_form_bm3(v, v0, e0, b0, derivative) + noise
        for v, noise in zip(volumes, (0.0002, -0.0001, 0.0001, -0.0003, 0.0002, 0.0001, -0.0002), strict=True)
    ]
    fit = fit_birch_murnaghan(volumes, energies)
    assert fit.rmse > 0.0
    assert fit.residuals == pytest.approx(
        tuple(observed - fit.energy(volume) for volume, observed in zip(volumes, energies, strict=True)),
        abs=2e-11,
    )
    order = (4, 0, 6, 1, 5, 2, 3)
    permuted = fit_birch_murnaghan([volumes[i] for i in order], [energies[i] for i in order])
    assert permuted.equilibrium_volume == pytest.approx(fit.equilibrium_volume)
    assert permuted.equilibrium_energy == pytest.approx(fit.equilibrium_energy)
    assert permuted.bulk_modulus == pytest.approx(fit.bulk_modulus)
    assert permuted.bulk_modulus_derivative == pytest.approx(fit.bulk_modulus_derivative)
    assert permuted.volumes == tuple(volumes[i] for i in order)


def test_consistent_extensive_rescaling_and_energy_offset() -> None:
    v0, e0, b0, derivative = 15.0, -9.2, 0.6, 4.2
    volumes = tuple(v0 * factor for factor in (0.82, 0.9, 0.96, 1.0, 1.05, 1.12, 1.2))
    energies = tuple(_closed_form_bm3(v, v0, e0, b0, derivative) for v in volumes)
    fit = fit_birch_murnaghan(volumes, energies)
    shifted = fit_birch_murnaghan(volumes, tuple(value + 1000.0 for value in energies))
    scaled = fit_birch_murnaghan(tuple(4.0 * v for v in volumes), tuple(4.0 * e for e in energies))

    assert shifted.equilibrium_energy == pytest.approx(fit.equilibrium_energy + 1000.0, abs=2e-12)
    assert shifted.bulk_modulus == pytest.approx(fit.bulk_modulus)
    assert shifted.bulk_modulus_derivative == pytest.approx(fit.bulk_modulus_derivative)
    assert scaled.equilibrium_volume == pytest.approx(4.0 * fit.equilibrium_volume)
    assert scaled.equilibrium_energy == pytest.approx(4.0 * fit.equilibrium_energy)
    assert scaled.bulk_modulus == pytest.approx(fit.bulk_modulus)
    assert scaled.bulk_modulus_derivative == pytest.approx(fit.bulk_modulus_derivative)


@pytest.mark.parametrize(
    ("volumes", "energies"),
    [
        ([1, 2, 3, 4, 5], [1, 2, 3]),
        ([[1], [2], [3], [4], [5]], [1, 2, 3, 4, 5]),
        ([1, 2, 3, 4, 5], [[1], [2], [3], [4], [5]]),
        ([1, 2, 3, 4, math.nan], [1, 2, 3, 4, 5]),
        ([1, 2, 3, 4, 5], [1, 2, 3, 4, math.inf]),
        ([1, 2, 3, 4, 5], [1, 2, 3, 4, complex(5, 0)]),
        ([1, 2, 3, 4, complex(5, 1)], [1, 2, 3, 4, 5]),
        ([0, 1, 2, 3, 4], [1, 2, 3, 4, 5]),
        ([-1, 1, 2, 3, 4], [1, 2, 3, 4, 5]),
        ([1, 2, 3, 4], [1, 2, 3, 4]),
        ([1, 2, 3, 4, 5], [1, 2, 3, 4]),
        ([1, 2, 3, 4, 4], [1, 2, 3, 4, 5]),
    ],
)
def test_invalid_inputs_are_rejected(volumes: object, energies: object) -> None:
    with pytest.raises(ValueError):
        fit_birch_murnaghan(volumes, energies)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "energies",
    [
        (0.0, 0.0, 0.0, 0.0, 0.0),  # flat
        (0.0, 1.0, 2.0, 3.0, 4.0),  # monotonic
        (4.0, 3.0, 2.0, 1.0, 0.0),  # monotonic
        tuple((x - 1.0) ** 2 for x in (1.0, 1.5, 2.0, 2.5, 3.0)),  # boundary
        tuple((x - 3.0) ** 2 for x in (1.0, 1.5, 2.0, 2.5, 3.0)),  # boundary
        tuple((x - 0.5) ** 2 for x in (1.0, 1.5, 2.0, 2.5, 3.0)),  # outside
    ],
)
def test_absent_or_unbracketed_interior_minimum_is_rejected(energies: tuple[float, ...]) -> None:
    x_values = (1.0, 1.5, 2.0, 2.5, 3.0)
    volumes = tuple(2.0 * x**-1.5 for x in x_values)
    with pytest.raises(ValueError):
        fit_birch_murnaghan(volumes, energies)


def test_selects_positive_curvature_stationary_root_and_rejects_repeated_root() -> None:
    # In the internally scaled coordinate z=(x-2), p=z**3-z has a maximum
    # at -1/sqrt(3) and the required minimum at +1/sqrt(3).
    x_values = (1.0, 1.5, 2.0, 2.5, 3.0)
    volumes = tuple(2.0 * x**-1.5 for x in x_values)
    energies = tuple((x - 2.0) ** 3 - (x - 2.0) for x in x_values)
    fit = fit_birch_murnaghan(volumes, energies)
    selected_x = 2.0 + 1.0 / math.sqrt(3.0)
    assert fit.equilibrium_volume == pytest.approx(2.0 * selected_x**-1.5, rel=2e-12)

    repeated = tuple((x - 2.0) ** 3 for x in x_values)
    with pytest.raises(ValueError):
        fit_birch_murnaghan(volumes, repeated)


def test_clustered_distinct_samples_can_be_rank_deficient() -> None:
    first = 1.0
    volumes = (first, math.nextafter(first, 2.0), math.nextafter(math.nextafter(first, 2.0), 2.0), 1.0 + 1e-15, 2.0)
    energies = tuple(_closed_form_bm3(v, 1.5, 0.0, 1.0, 4.0) for v in volumes)
    with pytest.raises(ValueError, match="rank deficient"):
        fit_birch_murnaghan(volumes, energies)


def test_narrow_window_can_hide_pressure_derivative_sensitivity() -> None:
    v0, b0, derivative = 10.0, 0.8, 5.0
    volumes = tuple(v0 * factor for factor in (0.997, 0.9985, 0.9995, 1.0005, 1.0015, 1.003))
    energies = [_closed_form_bm3(v, v0, -2.0, b0, derivative) for v in volumes]
    perturbed = energies.copy()
    perturbed[0] += 2e-10
    base = fit_birch_murnaghan(volumes, energies)
    changed = fit_birch_murnaghan(volumes, perturbed)

    assert base.condition_number < 100.0
    assert abs(changed.bulk_modulus_derivative - base.bulk_modulus_derivative) > 1e-3


def test_fit_and_result_state_are_immutable_and_query_volumes_are_validated() -> None:
    volumes = [8.0, 9.0, 10.0, 11.0, 12.0]
    energies = [_closed_form_bm3(v, 10.0, -1.0, 0.9, 4.1) for v in volumes]
    fit = fit_birch_murnaghan(volumes, energies)
    volumes[2] = 999.0
    energies[2] = 999.0

    assert fit.volumes == (8.0, 9.0, 10.0, 11.0, 12.0)
    assert isinstance(fit, BirchMurnaghanFit)
    with pytest.raises(FrozenInstanceError):
        fit.equilibrium_volume = 0.0  # type: ignore[misc]
    with pytest.raises(TypeError):
        fit.volumes[0] = 0.0  # type: ignore[index]
    for bad_volume in (0.0, -1.0, math.inf, math.nan, 1 + 2j, [1.0]):
        with pytest.raises(ValueError):
            fit.energy(bad_volume)  # type: ignore[arg-type]
        with pytest.raises(ValueError):
            fit.pressure(bad_volume)  # type: ignore[arg-type]
    assert fit.energy(fit.equilibrium_volume * 1.5) == pytest.approx(
        _closed_form_bm3(
            fit.equilibrium_volume * 1.5,
            fit.equilibrium_volume,
            fit.equilibrium_energy,
            fit.bulk_modulus,
            fit.bulk_modulus_derivative,
        )
    )
