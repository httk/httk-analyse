"""Harmonic phonon thermodynamics and quasi-harmonic volume response."""

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np

from .. import definitions as defs
from .._constants import H_EV_PER_THZ, KB_EV_PER_K
from ..definitions import BoundValue, FieldBinding, _reject_selection, _series
from .eos import fit_birch_murnaghan

__all__ = [
    "GruneisenFit",
    "HarmonicThermodynamics",
    "QuasiHarmonicResult",
    "harmonic_thermodynamics",
    "harmonic_thermodynamics_from_dos",
    "mode_gruneisen",
    "quasiharmonic",
]


@dataclass(frozen=True, slots=True)
class HarmonicThermodynamics:
    """Tuple-backed harmonic thermodynamic properties in the input normalization.

    :param temperatures: Temperatures in K, in caller order.
    :param free_energy: Helmholtz free energy in eV.
    :param internal_energy: Vibrational internal energy in eV.
    :param entropy: Entropy in eV/K.
    :param heat_capacity: Constant-volume heat capacity in eV/K.
    :param zero_point_energy: Zero-point energy in eV.
    :param retained_mode_weight: Sum of retained mode weights.
    :param excluded_zero_weight: Sum of omitted zero-mode weights.
    :param excluded_imaginary_weight: Sum of omitted negative-frequency weights.
    :param cutoff_frequency: Zero-mode cutoff in THz that produced this result.
    """

    temperatures: tuple[float, ...]
    free_energy: tuple[float, ...]
    internal_energy: tuple[float, ...]
    entropy: tuple[float, ...]
    heat_capacity: tuple[float, ...]
    zero_point_energy: float
    retained_mode_weight: float
    excluded_zero_weight: float
    excluded_imaginary_weight: float
    cutoff_frequency: float

    def __post_init__(self) -> None:
        """Copy property arrays into immutable tuples."""
        for name in ("temperatures", "free_energy", "internal_energy", "entropy", "heat_capacity"):
            object.__setattr__(self, name, tuple(float(value) for value in getattr(self, name)))

    def _bound_values(self, **selection: Any) -> tuple[BoundValue, ...]:
        """Bind the zero-point energy and the vibrational thermodynamics series."""
        _reject_selection(self, selection)
        return (
            BoundValue("zero_point_energy", FieldBinding(defs.ZERO_POINT_ENERGY), float(self.zero_point_energy)),
            _series(
                defs.VIBRATIONAL_THERMODYNAMICS,
                temperatures=list(self.temperatures),
                helmholtz_free_energies=list(self.free_energy),
                internal_energies=list(self.internal_energy),
                entropies=list(self.entropy),
                heat_capacities=list(self.heat_capacity),
            ),
        )


@dataclass(frozen=True, slots=True)
class GruneisenFit:
    """Local polynomial mode Grüneisen estimates for caller-matched modes.

    :param reference_volume: Volume where derivatives are evaluated, in angstrom³.
    :param values: Mode Grüneisen parameters in mode order.
    :param rank: Rank of the shared log-volume polynomial design.
    :param condition_number: Condition number of its centered/scaled design.
    :param degree: Polynomial degree used for the fit.
    """

    reference_volume: float
    values: tuple[float, ...]
    rank: int
    condition_number: float
    degree: int

    def __post_init__(self) -> None:
        """Copy mode estimates into an immutable tuple."""
        object.__setattr__(self, "values", tuple(float(value) for value in self.values))


@dataclass(frozen=True, slots=True)
class QuasiHarmonicResult:
    """Equilibrium QHA properties fitted independently at each temperature.

    :param temperatures: Strictly increasing temperatures in K.
    :param equilibrium_volumes: Equilibrium volumes in angstrom³.
    :param free_energies: Minimized total (static plus vibrational) Helmholtz energies in eV.
    :param bulk_moduli: Equilibrium bulk moduli in GPa.
    :param volumetric_expansion: Finite-difference alpha_V in 1/K.
    :param retained_mode_weight: Common retained mode count at every volume.
    :param excluded_zero_weight: Common omitted zero-mode weight at every volume.
    :param excluded_imaginary_weight: Common omitted imaginary-mode weight at every volume.
    :param cutoff_frequency: Zero-mode cutoff in THz that produced this result.
    """

    temperatures: tuple[float, ...]
    equilibrium_volumes: tuple[float, ...]
    free_energies: tuple[float, ...]
    bulk_moduli: tuple[float, ...]
    volumetric_expansion: tuple[float, ...]
    retained_mode_weight: float
    excluded_zero_weight: float
    excluded_imaginary_weight: float
    cutoff_frequency: float

    def __post_init__(self) -> None:
        """Copy temperature-dependent properties into immutable tuples."""
        for name in ("temperatures", "equilibrium_volumes", "free_energies", "bulk_moduli", "volumetric_expansion"):
            object.__setattr__(self, name, tuple(float(value) for value in getattr(self, name)))

    def _bound_values(self, **selection: Any) -> tuple[BoundValue, ...]:
        """Bind the quasi-harmonic equilibrium series; free energies include the static energy."""
        _reject_selection(self, selection)
        return (
            _series(
                defs.QUASIHARMONIC_THERMODYNAMICS,
                temperatures=list(self.temperatures),
                equilibrium_volumes=list(self.equilibrium_volumes),
                total_helmholtz_free_energies=list(self.free_energies),
                bulk_moduli=list(self.bulk_moduli),
                volumetric_thermal_expansions=list(self.volumetric_expansion),
            ),
        )


def harmonic_thermodynamics(
    frequencies: Sequence[float],
    temperatures: Sequence[float],
    weights: Sequence[float] | None = None,
    *,
    zero_modes: Literal["raise", "omit"] = "raise",
    imaginary: Literal["raise", "omit"] = "raise",
    cutoff_frequency: float = 0.0,
) -> HarmonicThermodynamics:
    """Calculate harmonic free energy, energy, entropy and heat capacity.

    Frequencies are signed THz values; weights are nonnegative mode counts and
    are never normalized automatically. Negative modes are imaginary modes.

    :param frequencies: One-dimensional frequencies in THz.
    :param temperatures: Nonnegative temperatures in K, retained in input order.
    :param weights: Optional nonnegative mode counts matching the frequencies.
    :param zero_modes: Raise on zero frequencies or explicitly omit their weight.
    :param imaginary: Raise on negative frequencies or explicitly omit their weight.
    :param cutoff_frequency: Nonnegative THz; modes with ``|frequency|`` below it count as zero modes, of either sign.
    :return: Immutable properties in eV and eV/K, with mode exclusions reported.
    :raises ValueError: If inputs are malformed, policies reject modes, or results are non-finite.
    """
    freq = _vector(frequencies, "frequencies")
    temp = _vector(temperatures, "temperatures")
    if not len(freq) or not len(temp) or np.any(temp < 0):
        raise ValueError("frequencies and temperatures must be nonempty; temperatures must be nonnegative")
    if zero_modes not in ("raise", "omit") or imaginary not in ("raise", "omit"):
        raise ValueError("zero_modes and imaginary must be 'raise' or 'omit'")
    cutoff = _nonnegative(cutoff_frequency, "cutoff_frequency")
    mode_weights = np.ones(len(freq)) if weights is None else _vector(weights, "weights")
    if mode_weights.shape != freq.shape or np.any(mode_weights < 0) or not np.any(mode_weights > 0):
        raise ValueError("weights must match frequencies, be nonnegative, and have positive total")
    zero, negative = _classify_modes(freq, cutoff)
    if zero_modes == "raise" and np.any(zero & (mode_weights > 0)):
        raise ValueError("zero-frequency modes require zero_modes='omit'")
    if imaginary == "raise" and np.any(negative & (mode_weights > 0)):
        raise ValueError("negative frequencies require imaginary='omit'")
    retained = ~(zero | negative)
    f, w = freq[retained], mode_weights[retained]
    if not np.any(w > 0):
        raise ValueError("no positive-frequency modes remain")
    energies = H_EV_PER_THZ * f
    zpe = float(np.dot(w, energies) / 2.0)
    weighted_energy = w * energies
    free, internal, entropy, capacity = [], [], [], []
    for t in temp:
        if t == 0:
            free.append(zpe)
            internal.append(zpe)
            entropy.append(0.0)
            capacity.append(0.0)
            continue
        x = energies / (KB_EV_PER_K * t)
        small = x < 1e-5
        large = x > 700.0
        occupation = np.zeros_like(x)
        occupation[~large] = 1.0 / np.expm1(x[~large])
        thermal_log = np.zeros_like(x)
        thermal_log[~large] = np.log(-np.expm1(-x[~large]))
        f_value = zpe + KB_EV_PER_K * t * float(np.dot(w, thermal_log))
        u_value = zpe + float(np.dot(weighted_energy, occupation))
        s_terms = np.zeros_like(x)
        s_terms[~large] = x[~large] * occupation[~large] - thermal_log[~large]
        s_value = KB_EV_PER_K * float(np.dot(w, s_terms))
        cv_terms = np.zeros_like(x)
        cv_terms[small] = 1.0 - x[small] ** 2 / 12.0
        middle = ~(small | large)
        decay = np.exp(-x[middle])
        cv_terms[middle] = x[middle] ** 2 * decay / (1.0 - decay) ** 2
        cv_value = KB_EV_PER_K * float(np.dot(w, cv_terms))
        free.append(f_value)
        internal.append(u_value)
        entropy.append(s_value)
        capacity.append(cv_value)
    output = np.asarray([free, internal, entropy, capacity, [zpe] * len(temp)], dtype=np.float64)
    if not np.isfinite(output).all():
        raise ValueError("thermodynamic properties are non-finite")
    return HarmonicThermodynamics(
        tuple(map(float, temp)),
        tuple(free),
        tuple(internal),
        tuple(entropy),
        tuple(capacity),
        zpe,
        float(w.sum()),
        float(mode_weights[zero].sum()),
        float(mode_weights[negative].sum()),
        cutoff,
    )


def harmonic_thermodynamics_from_dos(
    frequencies: Sequence[float],
    density: Sequence[float],
    temperatures: Sequence[float],
    *,
    zero_modes: Literal["raise", "omit"] = "raise",
    imaginary: Literal["raise", "omit"] = "raise",
    cutoff_frequency: float = 0.0,
) -> HarmonicThermodynamics:
    """Integrate a sampled density of states with trapezoid node weights.

    :param frequencies: Strictly increasing DOS grid in THz.
    :param density: Nonnegative states/THz density at each grid point.
    :param temperatures: Nonnegative temperatures in K.
    :param zero_modes: Raise or omit quadrature weight located at zero frequency.
    :param imaginary: Raise or omit quadrature weight on negative frequencies.
    :param cutoff_frequency: Nonnegative THz; quadrature nodes with ``|frequency|`` below it count as zero modes.
    :return: Harmonic properties, with DOS quadrature weights as mode counts.
    :raises ValueError: If the grid, DOS integral, or selected mode policies are invalid.
    """
    grid, dos = _vector(frequencies, "frequencies"), _vector(density, "density")
    if len(grid) < 2 or dos.shape != grid.shape or np.any(np.diff(grid) <= 0) or np.any(dos < 0):
        raise ValueError("DOS requires matching increasing frequencies and nonnegative density")
    steps = np.diff(grid)
    quadrature = np.empty_like(grid)
    quadrature[0], quadrature[-1] = steps[0] / 2, steps[-1] / 2
    quadrature[1:-1] = (steps[:-1] + steps[1:]) / 2
    return harmonic_thermodynamics(
        tuple(map(float, grid)),
        temperatures,
        tuple(map(float, dos * quadrature)),
        zero_modes=zero_modes,
        imaginary=imaginary,
        cutoff_frequency=cutoff_frequency,
    )


def mode_gruneisen(
    volumes: Sequence[float],
    frequencies: Sequence[Sequence[float]],
    reference_volume: float | None = None,
) -> GruneisenFit:
    """Fit log frequency against log volume for already matched positive modes.

    Quadratic fits are used with three or more volumes, otherwise linear fits.
    There is no automatic branch or band matching.

    :param volumes: Distinct positive volumes in angstrom³.
    :param frequencies: Positive frequency matrix with shape (volume, mode), in THz.
    :param reference_volume: Positive derivative volume; defaults to the geometric mean.
    :return: Per-mode ``-d log(omega)/d log(V)`` and design diagnostics.
    :raises ValueError: If values are invalid, modes are nonpositive, or the fit is rank deficient.
    """
    volume = _vector(volumes, "volumes")
    freq = _matrix(frequencies, "frequencies")
    if len(volume) < 2 or freq.shape[0] != len(volume) or freq.shape[1] == 0:
        raise ValueError("at least two volume rows with one or more matched modes are required")
    if np.any(volume <= 0) or len(np.unique(volume)) != len(volume) or np.any(freq <= 0):
        raise ValueError("volumes must be distinct and positive; matched frequencies must be positive")
    reference = (
        math.exp(float(np.mean(np.log(volume))))
        if reference_volume is None
        else _positive(reference_volume, "reference_volume")
    )
    x = np.log(volume)
    center, scale = float(x.mean()), float((x.max() - x.min()) / 2)
    if scale <= 0:
        raise ValueError("volumes do not span a representable range")
    z = (x - center) / scale
    degree = 2 if len(volume) >= 3 else 1
    design = np.vander(z, N=degree + 1, increasing=True)
    values, _, rank, singular = np.linalg.lstsq(design, np.log(freq), rcond=None)
    if rank != degree + 1 or singular[-1] <= 0:
        raise ValueError("log-volume design is rank deficient")
    zref = (math.log(reference) - center) / scale
    derivative_z = values[1] + (2 * values[2] * zref if degree == 2 else 0.0)
    result = -derivative_z / scale
    condition = float(singular[0] / singular[-1])
    if not np.isfinite(result).all() or not math.isfinite(condition):
        raise ValueError("Gruneisen fit is non-finite")
    return GruneisenFit(reference, tuple(map(float, result)), int(rank), condition, degree)


def quasiharmonic(
    volumes: Sequence[float],
    static_energies: Sequence[float],
    frequencies_by_volume: Sequence[Sequence[float]],
    temperatures: Sequence[float],
    weights: Sequence[float] | None = None,
    *,
    zero_modes: Literal["raise", "omit"] = "raise",
    imaginary: Literal["raise", "omit"] = "raise",
    cutoff_frequency: float = 0.0,
) -> QuasiHarmonicResult:
    """Fit static energy plus harmonic Helmholtz free energy at each temperature.

    Frequency columns must identify the same modes at every volume. Temperatures
    must be strictly increasing so the endpoint-aware numerical derivative is
    well defined. The returned alpha_V uses ``numpy.gradient`` at endpoints.

    :param volumes: Distinct positive sampled volumes in angstrom³.
    :param static_energies: Matching static energies in eV on one extensive basis.
    :param frequencies_by_volume: Matched signed mode frequencies, one row per volume.
    :param temperatures: At least three strictly increasing nonnegative K values.
    :param weights: Optional common mode counts.
    :param zero_modes: Explicit zero-mode policy.
    :param imaginary: Explicit negative-frequency policy.
    :param cutoff_frequency: Nonnegative THz zero-mode cutoff, applied identically at every volume.
    :return: Equilibrium volume, minimized free energy, bulk modulus and alpha_V.
    :raises ValueError: If scans, mode exclusions or temperature data are inconsistent.
    """
    volume = _vector(volumes, "volumes")
    static = _vector(static_energies, "static_energies")
    freq = _matrix(frequencies_by_volume, "frequencies_by_volume")
    temp = _vector(temperatures, "temperatures")
    if len(volume) < 5 or len(np.unique(volume)) != len(volume) or np.any(volume <= 0):
        raise ValueError("quasiharmonic fitting requires at least five distinct positive volumes")
    if static.shape != volume.shape or freq.shape[0] != len(volume) or freq.shape[1] == 0:
        raise ValueError("energies and frequency rows must match the volume scan")
    if len(temp) < 3 or np.any(temp < 0) or np.any(np.diff(temp) <= 0):
        raise ValueError("at least three strictly increasing nonnegative temperatures are required")
    temperature_values = tuple(map(float, temp))
    volume_values = tuple(map(float, volume))
    row_results = [
        harmonic_thermodynamics(
            tuple(map(float, row)),
            temperature_values,
            weights,
            zero_modes=zero_modes,
            imaginary=imaginary,
            cutoff_frequency=cutoff_frequency,
        )
        for row in freq
    ]
    exclusions = {(r.retained_mode_weight, r.excluded_zero_weight, r.excluded_imaginary_weight) for r in row_results}
    masks = tuple(_mode_mask(row, cutoff_frequency) for row in freq)
    if len(exclusions) != 1 or any(mask != masks[0] for mask in masks[1:]):
        raise ValueError("mode exclusions must retain the same caller-matched modes at every volume")
    eq_volume, min_energy, bulk = [], [], []
    for ti in range(len(temp)):
        energies = [static[vi] + row_results[vi].free_energy[ti] for vi in range(len(volume))]
        fit = fit_birch_murnaghan(volume_values, energies)
        eq_volume.append(fit.equilibrium_volume)
        min_energy.append(fit.equilibrium_energy)
        bulk.append(fit.bulk_modulus)
    expansion = np.gradient(np.asarray(eq_volume), temp, edge_order=2) / np.asarray(eq_volume)
    if not np.isfinite(expansion).all():
        raise ValueError("thermal expansion is non-finite")
    first = row_results[0]
    return QuasiHarmonicResult(
        tuple(map(float, temp)),
        tuple(eq_volume),
        tuple(min_energy),
        tuple(bulk),
        tuple(map(float, expansion)),
        first.retained_mode_weight,
        first.excluded_zero_weight,
        first.excluded_imaginary_weight,
        first.cutoff_frequency,
    )


def _nonnegative(value: Any, name: str) -> float:
    """Validate a nonnegative finite scalar."""
    if isinstance(value, (str, bytes, complex, np.complexfloating)):
        raise ValueError(f"{name} must be nonnegative and finite")
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be nonnegative and finite") from exc
    if not math.isfinite(result) or result < 0:
        raise ValueError(f"{name} must be nonnegative and finite")
    return result


def _classify_modes(freq: np.ndarray, cutoff: float) -> tuple[np.ndarray, np.ndarray]:
    """Return (zero, imaginary) masks; ``|frequency| < cutoff`` is zero, so cutoff 0 is exact-zero only."""
    zero = (freq == 0) | (np.abs(freq) < cutoff)
    return zero, (freq < 0) & ~zero


def _mode_mask(row: np.ndarray, cutoff: float) -> tuple[int, ...]:
    """Classify modes as 1 retained, 0 zero or -1 imaginary with the same rule as the thermodynamics."""
    zero, imaginary = _classify_modes(row, _nonnegative(cutoff, "cutoff_frequency"))
    return tuple(int(m) for m in np.where(zero, 0, np.where(imaginary, -1, 1)))


def _vector(values: Sequence[float], name: str) -> np.ndarray:
    """Copy a real finite one-dimensional input to float64."""
    try:
        raw = np.asarray(values)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite real one-dimensional sequence") from exc
    if raw.ndim != 1 or np.iscomplexobj(raw):
        raise ValueError(f"{name} must be a finite real one-dimensional sequence")
    try:
        result = np.asarray(raw, dtype=np.float64)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be a finite real one-dimensional sequence") from exc
    if not np.isfinite(result).all():
        raise ValueError(f"{name} must be finite")
    return result.copy()


def _matrix(values: Sequence[Sequence[float]], name: str) -> np.ndarray:
    """Copy a real finite two-dimensional input to float64."""
    try:
        raw = np.asarray(values)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite real two-dimensional sequence") from exc
    if raw.ndim != 2 or np.iscomplexobj(raw):
        raise ValueError(f"{name} must be a finite real two-dimensional sequence")
    try:
        result = np.asarray(raw, dtype=np.float64)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be a finite real two-dimensional sequence") from exc
    if not np.isfinite(result).all():
        raise ValueError(f"{name} must be finite")
    return result.copy()


def _positive(value: Any, name: str) -> float:
    """Validate a positive finite scalar."""
    if isinstance(value, (str, bytes, complex, np.complexfloating)):
        raise ValueError(f"{name} must be positive and finite")
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be positive and finite") from exc
    if not math.isfinite(result) or result <= 0:
        raise ValueError(f"{name} must be positive and finite")
    return result
