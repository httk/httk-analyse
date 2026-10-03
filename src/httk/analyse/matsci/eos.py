"""Static materials equations of state."""

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

__all__ = ["BirchMurnaghanFit", "fit_birch_murnaghan"]

_EV_A3_TO_GPA = 160.2176634


@dataclass(frozen=True, slots=True)
class BirchMurnaghanFit:
    """A frozen third-order Birch–Murnaghan energy fit.

    Energies and volumes retain the input order and use eV and angstrom³ on one
    consistent extensive basis. The fit describes static energies for one
    composition and phase branch under one relaxation protocol. Its condition
    number describes only the scaled linear design, not parameter confidence.
    Residuals are observed minus predicted energies and are not uncertainty
    estimates.

    :param equilibrium_volume: Fitted minimum volume in angstrom³.
    :param equilibrium_energy: Fitted minimum energy in eV.
    :param bulk_modulus: Fitted bulk modulus in eV/angstrom³.
    :param bulk_modulus_derivative: Dimensionless pressure derivative at equilibrium.
    :param volumes: Input volumes in original order, in angstrom³.
    :param energies: Input energies in original order, in eV.
    :param residuals: Observed minus fitted energies in input order, in eV.
    :param rmse: Root mean square of the energy residuals, in eV.
    :param condition_number: Condition number of the scaled cubic design matrix.
    """

    equilibrium_volume: float
    equilibrium_energy: float
    bulk_modulus: float
    bulk_modulus_derivative: float
    volumes: tuple[float, ...]
    energies: tuple[float, ...]
    residuals: tuple[float, ...]
    rmse: float
    condition_number: float

    def __post_init__(self) -> None:
        """Copy sequence fields into immutable tuples."""
        object.__setattr__(self, "volumes", tuple(self.volumes))
        object.__setattr__(self, "energies", tuple(self.energies))
        object.__setattr__(self, "residuals", tuple(self.residuals))

    @property
    def bulk_modulus_gpa(self) -> float:
        """Return the bulk modulus in gigapascals.

        :return: The bulk modulus in GPa.
        """
        return self.bulk_modulus * _EV_A3_TO_GPA

    def energy(self, volume: float) -> float:
        """Evaluate the fitted energy at a positive volume.

        Extrapolation is allowed; values outside the sampled volume interval are
        model predictions without validated physical accuracy. Finite-temperature
        MD mean energies do not define this static energy EOS.

        :param volume: Query volume in angstrom³.
        :return: Predicted energy in eV on the fit's extensive basis.
        :raises ValueError: If ``volume`` is not a positive finite scalar or the prediction is non-finite.
        """
        query = _positive_scalar(volume, "volume")
        eta_minus_one = _eta_minus_one(self.equilibrium_volume, query)
        prediction = self.equilibrium_energy + (9.0 * self.equilibrium_volume * self.bulk_modulus / 16.0) * (
            2.0 * eta_minus_one**2 + (self.bulk_modulus_derivative - 4.0) * eta_minus_one**3
        )
        if not math.isfinite(prediction):
            raise ValueError("fitted energy is non-finite at the requested volume")
        return prediction

    def pressure(self, volume: float) -> float:
        """Evaluate pressure, positive in compression, at a positive volume.

        Extrapolation is allowed as a model prediction, but its physical accuracy
        outside the sampled interval is not validated.

        :param volume: Query volume in angstrom³.
        :return: Predicted pressure in eV/angstrom³.
        :raises ValueError: If ``volume`` is not a positive finite scalar or the prediction is non-finite.
        """
        query = _positive_scalar(volume, "volume")
        eta_minus_one = _eta_minus_one(self.equilibrium_volume, query)
        strain = eta_minus_one / 2.0
        try:
            eta_power = math.exp((5.0 / 3.0) * (math.log(self.equilibrium_volume) - math.log(query)))
        except (OverflowError, ValueError) as exc:
            raise ValueError("fitted pressure is non-finite at the requested volume") from exc
        pressure = (
            3.0 * self.bulk_modulus * strain * eta_power * (1.0 + 1.5 * (self.bulk_modulus_derivative - 4.0) * strain)
        )
        if not math.isfinite(pressure):
            raise ValueError("fitted pressure is non-finite at the requested volume")
        return pressure


def fit_birch_murnaghan(volumes: Sequence[float], energies: Sequence[float]) -> BirchMurnaghanFit:
    """Fit a third-order Birch–Murnaghan static energy curve by linear least squares.

    The inputs must be finite one-dimensional sequences in angstrom³ and eV,
    respectively, on the same extensive basis. Supply static energies for one
    composition and phase branch with a consistent electronic-energy definition
    and fixed-volume relaxation protocol. Fixed-shape scaling and cell-shape
    relaxation can produce different responses. Finite-temperature MD total or
    potential energies are not isothermal pressure EOS data; use free energies
    or a separate pressure-versus-volume method for that case.

    The fit is an ordinary least-squares cubic in ``x = (Vref/V)**(2/3)``. It
    requires an interior stable minimum and imposes no fit-quality cutoff or
    physical bound on the pressure derivative. The scaled design condition
    number diagnoses only that linear solve. Inspect fit-window sensitivity,
    especially for the pressure derivative, which may be poorly determined even
    when the scaled design is well conditioned.

    :param volumes: Distinct positive sample volumes in angstrom³.
    :param energies: Matching static energies in eV on the same extensive basis.
    :return: The immutable fitted EOS and input-order residuals.
    :raises ValueError: If inputs are invalid, the cubic design is rank deficient, or there is no interior stable minimum.
    """
    volume_values = _finite_sequence(volumes, "volumes")
    energy_values = _finite_sequence(energies, "energies")
    if len(volume_values) != len(energy_values):
        raise ValueError("volumes and energies must have the same length")
    if len(volume_values) < 5:
        raise ValueError("at least five volume-energy samples are required")
    if any(volume <= 0.0 for volume in volume_values):
        raise ValueError("volumes must be positive")
    if len(set(volume_values)) != len(volume_values):
        raise ValueError("volumes must be distinct")

    log_reference = (math.log(min(volume_values)) + math.log(max(volume_values))) / 2.0
    x_reference = math.exp(log_reference)
    x_values = tuple(_eos_x(x_reference, volume) for volume in volume_values)
    x_min, x_max = min(x_values), max(x_values)
    x_center = x_min + (x_max - x_min) / 2.0
    x_scale = (x_max - x_min) / 2.0
    if not math.isfinite(x_scale) or x_scale <= 0.0:
        raise ValueError("volume samples do not span a representable fit interval")
    z_values = tuple((x - x_center) / x_scale for x in x_values)

    energy_min, energy_max = min(energy_values), max(energy_values)
    energy_offset = energy_min / 2.0 + energy_max / 2.0
    energy_scale = max(abs(energy - energy_offset) for energy in energy_values)
    if not math.isfinite(energy_scale) or energy_scale == 0.0:
        raise ValueError("energies do not define a non-flat fit")
    normalized_energies = np.asarray(
        [(energy - energy_offset) / energy_scale for energy in energy_values], dtype=np.float64
    )
    design = np.vander(np.asarray(z_values, dtype=np.float64), N=4, increasing=True)
    coefficients, _, rank, singular_values = np.linalg.lstsq(design, normalized_energies, rcond=None)
    if rank != 4 or not np.isfinite(coefficients).all() or singular_values[-1] <= 0.0:
        raise ValueError("scaled cubic design is rank deficient")
    condition_number = float(singular_values[0] / singular_values[-1])
    if not math.isfinite(condition_number):
        raise ValueError("scaled cubic design condition number is non-finite")
    coeffs = (
        float(coefficients[0]),
        float(coefficients[1]),
        float(coefficients[2]),
        float(coefficients[3]),
    )
    if coeffs[1:] == (0.0, 0.0, 0.0):
        raise ValueError("energies do not define a non-flat fit")

    stationary: list[tuple[float, float]] = []
    root_tolerance = 32.0 * math.ulp(1.0)
    for z in _stationary_roots(coeffs):
        x = x_center + x_scale * z
        if x <= 0.0 or not math.isfinite(x):
            continue
        volume = x_reference * x**-1.5
        curvature_z = 2.0 * coeffs[2] + 6.0 * coeffs[3] * z
        if (
            -1.0 + root_tolerance < z < 1.0 - root_tolerance
            and min(volume_values) < volume < max(volume_values)
            and curvature_z > 0.0
        ):
            stationary.append((z, volume))
    if len(stationary) != 1:
        raise ValueError("fit must have one positive-curvature minimum strictly inside the sampled volume interval")

    z0, equilibrium_volume = stationary[0]
    x0 = x_center + x_scale * z0
    _, _, c2, c3 = coeffs
    p2 = energy_scale * (2.0 * c2 + 6.0 * c3 * z0) / (x_scale * x_scale)
    p3 = energy_scale * 6.0 * c3 / (x_scale * x_scale * x_scale)
    bulk_modulus = 4.0 * x0 * x0 * p2 / (9.0 * equilibrium_volume)
    bulk_modulus_derivative = 4.0 + 2.0 * x0 * p3 / (3.0 * p2)

    predictions = tuple(energy_offset + energy_scale * _poly(coeffs, z) for z in z_values)
    residuals = tuple(observed - predicted for observed, predicted in zip(energy_values, predictions, strict=True))
    rmse = math.sqrt(sum(residual * residual for residual in residuals) / len(residuals))
    equilibrium_energy = energy_offset + energy_scale * _poly(coeffs, z0)
    outputs = (equilibrium_volume, equilibrium_energy, bulk_modulus, bulk_modulus_derivative, rmse)
    if not all(math.isfinite(value) for value in outputs + residuals + predictions):
        raise ValueError("fit produced non-finite results")
    if bulk_modulus <= 0.0:
        raise ValueError("fit produced a non-positive bulk modulus")

    return BirchMurnaghanFit(
        equilibrium_volume=equilibrium_volume,
        equilibrium_energy=equilibrium_energy,
        bulk_modulus=bulk_modulus,
        bulk_modulus_derivative=bulk_modulus_derivative,
        volumes=volume_values,
        energies=energy_values,
        residuals=residuals,
        rmse=rmse,
        condition_number=condition_number,
    )


def _finite_sequence(values: Sequence[float], name: str) -> tuple[float, ...]:
    """Copy and validate a finite one-dimensional real sequence."""
    try:
        rows = tuple(values)
    except TypeError as exc:
        raise ValueError(f"{name} must be a one-dimensional sequence") from exc
    normalized: list[float] = []
    for value in rows:
        if isinstance(value, (str, bytes, complex, np.complexfloating)):
            raise ValueError(f"{name} must contain real scalar values")
        try:
            number = float(value)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(f"{name} must contain finite real scalar values") from exc
        if not math.isfinite(number):
            raise ValueError(f"{name} must contain finite real scalar values")
        normalized.append(number)
    return tuple(normalized)


def _positive_scalar(value: Any, name: str) -> float:
    """Return one positive finite real scalar or raise ``ValueError``."""
    if isinstance(value, (str, bytes, complex, np.complexfloating)):
        raise ValueError(f"{name} must be a positive finite scalar")
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be a positive finite scalar") from exc
    if not math.isfinite(result) or result <= 0.0:
        raise ValueError(f"{name} must be a positive finite scalar")
    return result


def _eos_x(reference: float, volume: float) -> float:
    """Evaluate the BM3 volume coordinate while detecting overflow."""
    try:
        result = math.exp((2.0 / 3.0) * (math.log(reference) - math.log(volume)))
    except OverflowError as exc:
        raise ValueError("volume range cannot be represented in the EOS coordinate") from exc
    if not math.isfinite(result) or result <= 0.0:
        raise ValueError("volume range cannot be represented in the EOS coordinate")
    return result


def _eta_minus_one(equilibrium_volume: float, volume: float) -> float:
    """Evaluate ``(V0/V)**(2/3) - 1`` accurately near equilibrium."""
    try:
        result = math.expm1((2.0 / 3.0) * (math.log(equilibrium_volume) - math.log(volume)))
    except OverflowError as exc:
        raise ValueError("volume range cannot be represented in the EOS coordinate") from exc
    if not math.isfinite(result):
        raise ValueError("volume range cannot be represented in the EOS coordinate")
    return result


def _poly(coefficients: tuple[float, float, float, float], z: float) -> float:
    """Evaluate a cubic by Horner's rule."""
    c0, c1, c2, c3 = coefficients
    return ((c3 * z + c2) * z + c1) * z + c0


def _stationary_roots(coefficients: tuple[float, float, float, float]) -> tuple[float, ...]:
    """Solve the scaled cubic derivative without leaving scaled coordinates."""
    _, c1, c2, c3 = coefficients
    quadratic, linear, constant = 3.0 * c3, 2.0 * c2, c1
    if quadratic == 0.0:
        return () if linear == 0.0 else (-constant / linear,)
    discriminant = linear * linear - 4.0 * quadratic * constant
    if discriminant <= 32.0 * math.ulp(1.0) * (linear * linear + abs(4.0 * quadratic * constant)):
        return ()
    root_discriminant = math.sqrt(discriminant)
    stable_term = -0.5 * (linear + math.copysign(root_discriminant, linear))
    if stable_term == 0.0:
        return (-linear / (2.0 * quadratic),)
    roots = (stable_term / quadratic, constant / stable_term)
    return roots if roots[0] != roots[1] else (roots[0],)
