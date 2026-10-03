"""Explicit-reference defect, surface, path, and activated-rate summaries."""

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from itertools import pairwise
from typing import cast

import numpy as np

__all__ = [
    "ArrheniusFit",
    "ChargeTransition",
    "DefectFormationEnergy",
    "NEBProfile",
    "SurfaceEnergy",
    "adsorption_energy",
    "charge_transition_levels",
    "defect_formation_energy",
    "fit_arrhenius",
    "neb_profile",
    "segregation_energy",
    "surface_energy",
]

_BOLTZMANN_EV_PER_K = 8.617333262145e-5


@dataclass(frozen=True, slots=True)
class DefectFormationEnergy:
    """Defect formation energy and its signed eV term decomposition.

    :param energy: Total formation energy in eV.
    :param terms: Signed named terms whose sum is ``energy``.
    :param atom_deltas: Exact signed integer atom changes; positive means added.
    :param charge: Integer charge state; positive means electrons removed.
    """

    energy: float
    terms: tuple[tuple[str, float], ...]
    atom_deltas: tuple[tuple[str, int], ...]
    charge: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "energy", _finite(self.energy, "energy"))
        object.__setattr__(self, "terms", tuple((str(name), _finite(value, name)) for name, value in self.terms))
        object.__setattr__(
            self, "atom_deltas", tuple((str(name), _integer(value, name)) for name, value in self.atom_deltas)
        )
        object.__setattr__(self, "charge", _integer(self.charge, "charge"))


@dataclass(frozen=True, slots=True)
class ChargeTransition:
    """Stable charge-state changes in a finite Fermi-level interval.

    :param fermi_level: Crossing energy in eV relative to the supplied band reference.
    :param tied_charges: All charge states equal to the lower envelope at the crossing.
    :param left_charges: Lower-envelope charge states immediately to the left.
    :param right_charges: Lower-envelope charge states immediately to the right.
    """

    fermi_level: float
    tied_charges: tuple[int, ...]
    left_charges: tuple[int, ...]
    right_charges: tuple[int, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "fermi_level", _finite(self.fermi_level, "fermi_level"))
        for name in ("tied_charges", "left_charges", "right_charges"):
            object.__setattr__(self, name, tuple(_integer(value, name) for value in getattr(self, name)))


@dataclass(frozen=True, slots=True)
class SurfaceEnergy:
    """Surface excess energy using caller-supplied total exposed area.

    :param energy: Total excess energy in eV.
    :param surface_energy: Excess energy per total exposed area in eV/angstrom².
    :param exposed_area: Total exposed area in angstrom².
    :param terms: Signed named contributions in eV.
    """

    energy: float
    surface_energy: float
    exposed_area: float
    terms: tuple[tuple[str, float], ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "energy", _finite(self.energy, "energy"))
        object.__setattr__(self, "surface_energy", _finite(self.surface_energy, "surface_energy"))
        object.__setattr__(self, "exposed_area", _finite(self.exposed_area, "exposed_area"))
        object.__setattr__(self, "terms", tuple((str(name), _finite(value, name)) for name, value in self.terms))


@dataclass(frozen=True, slots=True)
class NEBProfile:
    """Sampled NEB path barriers without interpolation between images.

    :param reaction_coordinates: Strictly increasing caller coordinates.
    :param energies: Image energies in eV in path order.
    :param forward_barrier: Maximum image energy minus the first image, in eV.
    :param reverse_barrier: Maximum image energy minus the last image, in eV.
    :param reaction_energy: Last minus first image energy, in eV.
    :param saddle_index: First image attaining the maximum.
    :param tied_saddle_indices: All images within the explicit tie tolerance of the maximum.
    """

    reaction_coordinates: tuple[float, ...]
    energies: tuple[float, ...]
    forward_barrier: float
    reverse_barrier: float
    reaction_energy: float
    saddle_index: int
    tied_saddle_indices: tuple[int, ...]

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "reaction_coordinates", _finite_sequence(self.reaction_coordinates, "reaction_coordinates")
        )
        object.__setattr__(self, "energies", _finite_sequence(self.energies, "energies"))
        for name in ("forward_barrier", "reverse_barrier", "reaction_energy"):
            object.__setattr__(self, name, _finite(getattr(self, name), name))
        object.__setattr__(self, "saddle_index", _integer(self.saddle_index, "saddle_index"))
        object.__setattr__(
            self,
            "tied_saddle_indices",
            tuple(_integer(value, "tied_saddle_indices") for value in self.tied_saddle_indices),
        )


@dataclass(frozen=True, slots=True)
class ArrheniusFit:
    """Weighted linear fit of log rate against reciprocal temperature.

    :param temperatures: Input temperatures in K.
    :param rates: Input positive rates in the caller's unit.
    :param log_residuals: Observed minus fitted natural-log rates.
    :param activation_energy: Fitted activation energy in eV.
    :param prefactor: Fitted positive rate prefactor in the input rate unit.
    :param rate_unit: Caller-supplied label for the rate and prefactor unit.
    :param rank: Rank of the scaled weighted design matrix.
    :param condition_number: Condition number of that scaled design matrix.
    :param weighted_rmse: Weighted RMS log-rate residual.
    """

    temperatures: tuple[float, ...]
    rates: tuple[float, ...]
    log_residuals: tuple[float, ...]
    activation_energy: float
    prefactor: float
    rate_unit: str
    rank: int
    condition_number: float
    weighted_rmse: float

    def __post_init__(self) -> None:
        for name in ("temperatures", "rates", "log_residuals"):
            object.__setattr__(self, name, _finite_sequence(getattr(self, name), name))
        for name in ("activation_energy", "prefactor", "condition_number", "weighted_rmse"):
            object.__setattr__(self, name, _finite(getattr(self, name), name))
        object.__setattr__(self, "rank", _integer(self.rank, "rank"))
        if not isinstance(self.rate_unit, str):
            raise ValueError("rate_unit must be a string")


def defect_formation_energy(
    defect_energy: float,
    host_energy: float,
    atom_deltas: Mapping[str, int],
    chemical_potentials: Mapping[str, float],
    *,
    charge: int,
    fermi_level: float,
    vbm: float,
    alignment: float,
    correction: float,
) -> DefectFormationEnergy:
    """Compute a charged defect energy from explicit eV references.

    The expression is ``Edef-Ehost-sum(delta_n*mu)+q*(EF+VBM+alignment)+correction``.
    Positive atom deltas add atoms; positive charge removes electrons.

    :param defect_energy: Defect supercell total energy in eV.
    :param host_energy: Matching host supercell total energy in eV.
    :param atom_deltas: Exact signed integer atom changes by element.
    :param chemical_potentials: Required elemental reservoirs in eV per atom.
    :param charge: Exact integer charge state.
    :param fermi_level: Fermi energy in eV relative to the supplied reference.
    :param vbm: Valence-band reference energy in eV.
    :param alignment: Explicit potential alignment term in eV.
    :param correction: Explicit finite-size or other correction in eV.
    :return: Immutable total and signed term decomposition.
    :raises ValueError: If atom deltas, references, or energies are invalid.
    """
    deltas = _integer_deltas(atom_deltas)
    mus = _required_values(chemical_potentials, tuple(element for element, _ in deltas), "chemical_potentials")
    q = _integer(charge, "charge")
    e_defect = _finite(defect_energy, "defect_energy")
    e_host = _finite(host_energy, "host_energy")
    ef, band, align, corr = (
        _finite(value, name)
        for value, name in (
            (fermi_level, "fermi_level"),
            (vbm, "vbm"),
            (alignment, "alignment"),
            (correction, "correction"),
        )
    )
    reservoir = _sum(delta * mus[element] for element, delta in deltas)
    terms = (
        ("defect_energy", e_defect),
        ("-host_energy", -e_host),
        ("-atom_reservoirs", -reservoir),
        ("charge_fermi_vbm_alignment", _finite(q * (ef + band + align), "charge term")),
        ("correction", corr),
    )
    return DefectFormationEnergy(_sum(value for _, value in terms), terms, deltas, q)


def charge_transition_levels(
    intercepts: Mapping[int, float],
    fermi_interval: tuple[float, float],
    *,
    tolerance: float = 1e-10,
) -> tuple[ChargeTransition, ...]:
    """Return only charge crossings on the lower envelope in a finite interval.

    Each intercept is the formation energy at ``EF=0``; charge ``q`` gives
    the line ``intercept + q*EF``. Coincident crossings report every tied
    charge. Crossings at interval endpoints are excluded.

    :param intercepts: Nonempty mapping from distinct integer charges to eV intercepts.
    :param fermi_interval: Increasing finite ``(low, high)`` bounds in eV.
    :param tolerance: Absolute eV tolerance for tie recognition.
    :return: Stable transitions ordered by Fermi energy.
    :raises ValueError: If charges, interval, intercepts, or tolerance are invalid.
    """
    if not isinstance(intercepts, Mapping) or len(intercepts) < 2:
        raise ValueError("intercepts must map at least two charge states to energies")
    lines = tuple(sorted((_integer(q, "charge"), _finite(e, f"intercepts[{q}]")) for q, e in intercepts.items()))
    low, high = _finite_sequence(fermi_interval, "fermi_interval", length=2)
    tol = _finite(tolerance, "tolerance")
    if high <= low or tol < 0:
        raise ValueError("fermi_interval must increase and tolerance must be non-negative")
    candidates = sorted({(b - a) / (qa - qb) for (qa, a), (qb, b) in _pairs(lines) if low < (b - a) / (qa - qb) < high})
    points = [low, *candidates, high]
    stable: list[tuple[int, ...]] = []
    for left, right in pairwise(points):
        middle = left + (right - left) / 2.0
        values = tuple((q, intercept + q * middle) for q, intercept in lines)
        minimum = min(value for _, value in values)
        stable.append(tuple(q for q, value in values if value == minimum))
    result: list[ChargeTransition] = []
    for index, level in enumerate(candidates):
        left_charges, right_charges = stable[index], stable[index + 1]
        if left_charges == right_charges:
            continue
        values = tuple((q, intercept + q * level) for q, intercept in lines)
        minimum = min(value for _, value in values)
        tied = tuple(q for q, value in values if value <= minimum + tol)
        result.append(ChargeTransition(level, tied, left_charges, right_charges))
    return tuple(result)


def surface_energy(
    slab_energy: float,
    bulk_energy_per_atom: float,
    bulk_reference_atom_count: int,
    excess_atom_deltas: Mapping[str, int],
    chemical_potentials: Mapping[str, float],
    *,
    total_exposed_area: float,
) -> SurfaceEnergy:
    """Compute surface excess energy with an explicit total exposed area.

    :param slab_energy: Total slab energy in eV.
    :param bulk_energy_per_atom: Bulk reference in eV per atom.
    :param bulk_reference_atom_count: Exact positive atom count in the stoichiometric bulk reference before excess or missing atoms are applied; it may differ from the slab count.
    :param excess_atom_deltas: Exact signed excess atom counts by element.
    :param chemical_potentials: Required elemental reservoirs in eV per atom.
    :param total_exposed_area: Sum of exposed face areas in angstrom².
    :return: Total and area-normalized excess energy with signed terms.
    :raises ValueError: If references, atom counts, or area are invalid.
    """
    count = _positive_integer(bulk_reference_atom_count, "bulk_reference_atom_count")
    deltas = _integer_deltas(excess_atom_deltas)
    mus = _required_values(chemical_potentials, tuple(element for element, _ in deltas), "chemical_potentials")
    area = _finite(total_exposed_area, "total_exposed_area")
    if area <= 0:
        raise ValueError("total_exposed_area must be positive")
    terms = (
        ("slab_energy", _finite(slab_energy, "slab_energy")),
        ("-bulk_reference", -_finite(bulk_energy_per_atom, "bulk_energy_per_atom") * count),
        ("-excess_atom_reservoirs", -_sum(delta * mus[element] for element, delta in deltas)),
    )
    excess = _sum(value for _, value in terms)
    return SurfaceEnergy(excess, excess / area, area, terms)


def adsorption_energy(
    combined_energy: float,
    substrate_energy: float,
    adsorbate_counts: Mapping[str, int],
    reference_energies: Mapping[str, float],
) -> float:
    """Return adsorption energy; negative values indicate favorable binding.

    :param combined_energy: Adsorbate-plus-substrate total energy in eV.
    :param substrate_energy: Clean substrate total energy in eV.
    :param adsorbate_counts: Positive integer counts of each reference species.
    :param reference_energies: Required total reference energies in eV.
    :return: Combined minus substrate minus adsorbate reference energy in eV.
    :raises ValueError: If counts, references, or energies are invalid.
    """
    counts = _positive_counts(adsorbate_counts)
    refs = _required_values(reference_energies, tuple(counts), "reference_energies")
    return _sum(
        (
            _finite(combined_energy, "combined_energy"),
            -_finite(substrate_energy, "substrate_energy"),
            -_sum(count * refs[name] for name, count in counts.items()),
        )
    )


def segregation_energy(bulk_environment_energy: float, target_environment_energy: float) -> float:
    """Return caller-ordered target-minus-bulk defect energy in eV.

    :param bulk_environment_energy: Defect energy in the source environment, eV.
    :param target_environment_energy: Matched defect energy in the target environment, eV.
    :return: Target minus source energy; negative favors the target environment.
    :raises ValueError: If either energy is not finite and real.
    """
    return _finite(
        _finite(target_environment_energy, "target_environment_energy")
        - _finite(bulk_environment_energy, "bulk_environment_energy"),
        "segregation_energy",
    )


def neb_profile(
    reaction_coordinates: Sequence[float], energies: Sequence[float], *, tie_tolerance: float = 1e-10
) -> NEBProfile:
    """Summarize sampled forward and reverse barriers in caller path order.

    No unsampled saddle is interpolated. The first maximum is the saddle index;
    every image within ``tie_tolerance`` of it is retained.

    :param reaction_coordinates: Strictly increasing coordinates for NEB images.
    :param energies: Matching image energies in eV.
    :param tie_tolerance: Non-negative absolute energy tolerance in eV.
    :return: Immutable sampled barriers and saddle indices.
    :raises ValueError: If arrays are mismatched, too short, or invalid.
    """
    coordinates = _finite_sequence(reaction_coordinates, "reaction_coordinates")
    values = _finite_sequence(energies, "energies")
    tolerance = _finite(tie_tolerance, "tie_tolerance")
    if len(values) < 2 or len(coordinates) != len(values) or any(b <= a for a, b in pairwise(coordinates)):
        raise ValueError("matching path arrays need at least two images and strictly increasing coordinates")
    if tolerance < 0:
        raise ValueError("tie_tolerance must be non-negative")
    maximum = max(values)
    tied = tuple(index for index, value in enumerate(values) if maximum - value <= tolerance)
    return NEBProfile(
        coordinates,
        values,
        maximum - values[0],
        maximum - values[-1],
        values[-1] - values[0],
        values.index(maximum),
        tied,
    )


def fit_arrhenius(
    temperatures: Sequence[float],
    rates: Sequence[float],
    *,
    rate_unit: str,
    weights: Sequence[float] | None = None,
) -> ArrheniusFit:
    """Fit positive rates to ``ln(rate)=ln(A)-Ea/(k_B*T)``.

    The fit uses weighted least squares when positive weights are supplied.
    It retains every point; selecting an Arrhenius window remains the caller's
    responsibility. The prefactor has the same unit as ``rates``.

    :param temperatures: Positive temperatures in K.
    :param rates: Positive rates in an explicitly labeled caller unit.
    :param rate_unit: Nonempty label such as ``"s^-1"`` or ``"ps^-1"``.
    :param weights: Optional positive relative squared-residual weights.
    :return: Immutable fit, log residuals, rank, and scaled condition diagnostic.
    :raises ValueError: If data are insufficient, invalid, or rank deficient.
    """
    temp = _finite_sequence(temperatures, "temperatures")
    rate = _finite_sequence(rates, "rates")
    if len(temp) < 3 or len(rate) != len(temp) or any(value <= 0 for value in temp + rate):
        raise ValueError("at least three matching positive temperatures and rates are required")
    if not isinstance(rate_unit, str) or not rate_unit.strip():
        raise ValueError("rate_unit must be a nonempty caller-supplied label")
    weight = (
        np.ones(len(temp)) if weights is None else np.asarray(_finite_sequence(weights, "weights"), dtype=np.float64)
    )
    if weight.shape != (len(temp),) or np.any(weight <= 0):
        raise ValueError("weights must be positive and match temperatures")
    x = 1.0 / np.asarray(temp)
    x_center = float(np.mean(x))
    x_scale = float(np.ptp(x))
    if x_scale == 0.0:
        raise ValueError("temperatures must contain at least two distinct values")
    x_scaled = (x - x_center) / x_scale
    design = np.column_stack((np.ones(len(x)), x_scaled))
    root_weight = np.sqrt(weight)
    weighted_design = design * root_weight[:, None]
    weighted_log_rate = np.log(rate) * root_weight
    coefficients, _, rank, _ = np.linalg.lstsq(weighted_design, weighted_log_rate, rcond=None)
    condition = float(np.linalg.cond(weighted_design))
    if rank != 2 or not math.isfinite(condition):
        raise ValueError("Arrhenius fit is rank deficient")
    slope = float(coefficients[1] / x_scale)
    intercept = float(coefficients[0] - slope * x_center)
    activation = -slope * _BOLTZMANN_EV_PER_K
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        prefactor = math.exp(intercept) if intercept < 710.0 else math.inf
    fitted = intercept + slope * x
    residuals = np.log(rate) - fitted
    rmse = math.sqrt(float(np.average(residuals**2, weights=weight)))
    if not all(math.isfinite(value) for value in (activation, prefactor, rmse, *residuals)) or prefactor <= 0:
        raise ValueError("Arrhenius fit produced non-finite values")
    return ArrheniusFit(
        temp,
        rate,
        tuple(float(v) for v in residuals),
        activation,
        prefactor,
        rate_unit.strip(),
        int(rank),
        condition,
        rmse,
    )


def _finite(value: object, name: str) -> float:
    if isinstance(value, (bool, np.bool_, str, bytes, complex, np.complexfloating)) or not np.isscalar(value):
        raise ValueError(f"{name} must be a finite real scalar")
    try:
        result = float(cast(float, value))
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be a finite real scalar") from exc
    if not math.isfinite(result):
        raise ValueError(f"{name} must be a finite real scalar")
    return result


def _finite_sequence(values: object, name: str, *, length: int | None = None) -> tuple[float, ...]:
    try:
        array = np.asarray(values)
        if array.ndim != 1 or np.iscomplexobj(array):
            raise ValueError(f"{name} must be a one-dimensional real sequence")
        result = tuple(_finite(item, name) for item in array)
    except (TypeError, OverflowError) as exc:
        raise ValueError(f"{name} must be a one-dimensional finite real sequence") from exc
    if not result or (length is not None and len(result) != length):
        raise ValueError(f"{name} must contain {length or 'at least one'} value(s)")
    return result


def _integer(value: object, name: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer")
    return int(value)


def _positive_integer(value: object, name: str) -> int:
    result = _integer(value, name)
    if result <= 0:
        raise ValueError(f"{name} must be positive")
    return result


def _integer_deltas(values: Mapping[str, int]) -> tuple[tuple[str, int], ...]:
    if not isinstance(values, Mapping):
        raise ValueError("atom deltas must map element labels to exact integers")
    deltas: list[tuple[str, int]] = []
    for element, raw in values.items():
        if not isinstance(element, str) or not element:
            raise ValueError("element labels must be nonempty strings")
        deltas.append((element, _integer(raw, f"atom_deltas[{element}]")))
    return tuple(sorted((element, delta) for element, delta in deltas if delta))


def _positive_counts(values: Mapping[str, int]) -> dict[str, int]:
    if not isinstance(values, Mapping) or not values:
        raise ValueError("counts must be a nonempty mapping")
    result = {}
    for key, value in values.items():
        if not isinstance(key, str) or not key:
            raise ValueError("count labels must be nonempty strings")
        result[key] = _positive_integer(value, f"counts[{key}]")
    return result


def _required_values(values: Mapping[str, float], keys: tuple[str, ...], name: str) -> dict[str, float]:
    if not isinstance(values, Mapping):
        raise ValueError(f"{name} must be a mapping")
    missing = [key for key in keys if key not in values]
    if missing:
        raise ValueError(f"missing {name}: {', '.join(missing)}")
    return {key: _finite(values[key], f"{name}[{key}]") for key in keys}


def _sum(values: Sequence[float] | object) -> float:
    try:
        result = math.fsum(values)  # type: ignore[arg-type]
    except (OverflowError, ValueError, TypeError) as exc:
        raise ValueError("result is not finite") from exc
    if not math.isfinite(result):
        raise ValueError("result is not finite")
    return result


def _pairs(values: tuple[tuple[int, float], ...]):
    for first_index, first in enumerate(values):
        for second in values[first_index + 1 :]:
            yield first, second
