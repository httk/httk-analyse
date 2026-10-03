"""Static energy bookkeeping for materials calculations."""

import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, cast

import numpy as np

__all__ = [
    "ChemicalPotentialRegion",
    "ConvergenceTable",
    "FormationEnergy",
    "chemical_potential_region",
    "convergence_table",
    "enthalpy",
    "formation_energy",
    "reaction_energy",
]


@dataclass(frozen=True, slots=True)
class ConvergenceTable:
    """Immutable input-order energies and signed differences per atom.

    :param parameters: Caller-supplied immutable parameter labels.
    :param energies: Total energies in eV.
    :param atom_counts: Positive atom counts for each configuration.
    :param energies_per_atom: Total energies divided by their atom counts.
    :param differences_per_atom: Per-atom energies minus the explicit reference.
    :param reference_index: Index of the reference configuration.
    """

    parameters: tuple[str | float | int, ...]
    energies: tuple[float, ...]
    atom_counts: tuple[int, ...]
    energies_per_atom: tuple[float, ...]
    differences_per_atom: tuple[float, ...]
    reference_index: int

    def __post_init__(self) -> None:
        """Copy every sequence field into immutable tuples."""
        labels = tuple(self.parameters)
        if any(isinstance(value, bool) or not isinstance(value, (str, int, float)) for value in labels):
            raise ValueError("parameters require immutable scalar labels")
        if any(isinstance(value, float) and not math.isfinite(value) for value in labels):
            raise ValueError("parameter labels must be finite")
        object.__setattr__(self, "parameters", labels)
        object.__setattr__(self, "energies", _finite_sequence(self.energies, "energies"))
        object.__setattr__(self, "atom_counts", tuple(int(value) for value in self.atom_counts))
        object.__setattr__(self, "energies_per_atom", _finite_sequence(self.energies_per_atom, "energies_per_atom"))
        object.__setattr__(
            self, "differences_per_atom", _finite_sequence(self.differences_per_atom, "differences_per_atom")
        )


@dataclass(frozen=True, slots=True)
class FormationEnergy:
    """Total and per-atom formation energy for explicit reservoirs.

    :param total: Formation energy in eV, ``E - sum(n_i * mu_i)``.
    :param per_atom: Formation energy divided by the total atom count, in eV/atom.
    :param atom_count: Total number of atoms in the composition.
    """

    total: float
    per_atom: float
    atom_count: float

    def __post_init__(self) -> None:
        """Retain finite real scalar values."""
        for name in ("total", "per_atom", "atom_count"):
            object.__setattr__(self, name, _finite_scalar(getattr(self, name), name))
        if self.atom_count <= 0:
            raise ValueError("atom_count must be positive")


@dataclass(frozen=True, slots=True)
class ChemicalPotentialRegion:
    """Linear host equality and competing-phase chemical-potential bounds.

    Coefficients follow ``elements`` order. The host constraint is
    ``host_coefficients · mu = host_energy``; each competing row is
    ``coefficients · mu <= energy``.

    :param elements: Sorted element labels defining the potential-vector order.
    :param host_coefficients: Host stoichiometry in ``elements`` order.
    :param host_energy: Host total energy in eV.
    :param competing_coefficients: Competing-phase stoichiometries in element order.
    :param competing_energies: Competing-phase total energies in eV.
    """

    elements: tuple[str, ...]
    host_coefficients: tuple[float, ...]
    host_energy: float
    competing_coefficients: tuple[tuple[float, ...], ...]
    competing_energies: tuple[float, ...]

    def __post_init__(self) -> None:
        """Copy coefficient rows and energies into immutable tuples."""
        if any(not isinstance(element, str) or not element for element in self.elements):
            raise ValueError("elements must be nonempty strings")
        object.__setattr__(self, "elements", tuple(self.elements))
        object.__setattr__(self, "host_energy", _finite_scalar(self.host_energy, "host_energy"))
        object.__setattr__(self, "host_coefficients", _finite_sequence(self.host_coefficients, "host_coefficients"))
        object.__setattr__(
            self,
            "competing_coefficients",
            tuple(_finite_sequence(row, "competing_coefficients") for row in self.competing_coefficients),
        )
        object.__setattr__(
            self,
            "competing_energies",
            _finite_sequence(self.competing_energies, "competing_energies", allow_empty=True),
        )

    def contains(self, potentials: Mapping[str, object], tolerance: float = 1e-10) -> bool:
        """Check whether elemental potentials satisfy the stored constraints.

        :param potentials: Elemental reference potentials in eV per atom.
        :param tolerance: Absolute residual tolerance in eV.
        :return: Whether the host equality and all competing-phase bounds hold.
        :raises ValueError: If a required potential is missing or inputs are invalid.
        """
        tol = _nonnegative_scalar(tolerance, "tolerance")
        mu = _potential_vector(potentials, self.elements)
        host_value = _finite_sum((a * b for a, b in zip(self.host_coefficients, mu, strict=True)), "host constraint")
        if abs(host_value - self.host_energy) > tol:
            return False
        return all(
            _finite_sum((a * b for a, b in zip(row, mu, strict=True)), "competing-phase constraint") - energy <= tol
            for row, energy in zip(self.competing_coefficients, self.competing_energies, strict=True)
        )


def convergence_table(
    parameters: Sequence[str | float | int],
    energies: Sequence[float],
    *,
    reference_index: int,
    atom_counts: Sequence[int],
) -> ConvergenceTable:
    """Build an input-order convergence table against an explicit reference.

    Parameter order is retained but is never interpreted as evidence of
    convergence. Energies use eV on each configuration's total-energy basis.

    :param parameters: Immutable labels such as cutoff energies or mesh sizes.
    :param energies: Finite total energies in eV.
    :param reference_index: Index of the configuration used for differences.
    :param atom_counts: Positive integer atom counts for matching configurations.
    :return: Immutable values and signed per-atom differences in input order.
    :raises ValueError: If lengths, labels, counts, energies, or reference are invalid.
    """
    labels = _materialize(parameters, "parameters")
    for label in labels:
        if isinstance(label, bool) or not isinstance(label, (str, int, float)):
            raise ValueError("parameters must contain only string, integer, or float labels")
        if isinstance(label, float) and not math.isfinite(label):
            raise ValueError("float parameter labels must be finite")
    energy_values = _finite_sequence(energies, "energies")
    raw_counts = _materialize(atom_counts, "atom_counts")
    if not labels or len(labels) != len(energy_values) or len(raw_counts) != len(labels):
        raise ValueError("parameters, energies, and atom_counts must have the same non-zero length")
    counts: list[int] = []
    for count in raw_counts:
        if isinstance(count, bool) or not isinstance(count, (int, np.integer)) or count <= 0:
            raise ValueError("atom_counts must contain positive integers")
        counts.append(int(count))
    if isinstance(reference_index, bool) or not isinstance(reference_index, (int, np.integer)):
        raise ValueError("reference_index must be an integer")
    reference = int(reference_index)
    if not 0 <= reference < len(labels):
        raise ValueError("reference_index is outside the input rows")
    per_atom = tuple(energy / count for energy, count in zip(energy_values, counts, strict=True))
    differences = tuple(value - per_atom[reference] for value in per_atom)
    if not all(math.isfinite(value) for value in per_atom + differences):
        raise ValueError("per-atom energies or differences are not finite")
    return ConvergenceTable(labels, energy_values, tuple(counts), per_atom, differences, reference)


def reaction_energy(
    energies: Sequence[float],
    coefficients: Sequence[float],
    compositions: Sequence[Mapping[str, object]],
    *,
    tolerance: float = 1e-10,
) -> float:
    """Return balanced products-minus-reactants energy in eV.

    Positive coefficients denote products and negative coefficients denote
    reactants. The energy and composition at each index refer to the same phase.

    :param energies: Finite total phase energies in eV.
    :param coefficients: Signed reaction coefficients.
    :param compositions: Element-to-stoichiometric-count mappings.
    :param tolerance: Absolute per-element balance tolerance in atoms.
    :return: The signed linear combination of total energies in eV.
    :raises ValueError: If rows differ in length, data are invalid, or the reaction is unbalanced.
    """
    energy_values = _finite_sequence(energies, "energies")
    coefficient_values = _finite_sequence(coefficients, "coefficients")
    composition_values = tuple(
        _composition(composition, f"compositions[{index}]")
        for index, composition in enumerate(_materialize(compositions, "compositions"))
    )
    tol = _nonnegative_scalar(tolerance, "tolerance")
    if (
        not energy_values
        or len(energy_values) != len(coefficient_values)
        or len(energy_values) != len(composition_values)
    ):
        raise ValueError("energies, coefficients, and compositions must have the same non-zero length")
    elements = set().union(*(composition.keys() for composition in composition_values))
    for element in elements:
        imbalance = _finite_sum(
            (
                coefficient * composition.get(element, 0.0)
                for coefficient, composition in zip(coefficient_values, composition_values, strict=True)
            ),
            f"reaction balance for {element}",
        )
        if abs(imbalance) > tol:
            raise ValueError(f"reaction is not balanced for element {element}")
    return _finite_sum(
        (coefficient * energy for coefficient, energy in zip(coefficient_values, energy_values, strict=True)),
        "reaction energy",
    )


def formation_energy(
    energy: float,
    composition: Mapping[str, object],
    chemical_potentials: Mapping[str, object],
) -> FormationEnergy:
    """Compute formation energy relative to explicitly supplied reservoirs.

    :param energy: Total phase energy in eV.
    :param composition: Element-to-stoichiometric-count mapping.
    :param chemical_potentials: Required elemental reference potentials in eV per atom.
    :return: Immutable total and per-atom formation energies.
    :raises ValueError: If a reservoir is missing or an input is invalid.
    """
    total_energy = _finite_scalar(energy, "energy")
    counts = _composition(composition, "composition")
    potentials = _potential_vector(chemical_potentials, tuple(sorted(counts)))
    atom_count = _finite_sum(counts.values(), "composition atom count")
    reservoir_energy = _finite_sum(
        (counts[element] * potential for element, potential in zip(sorted(counts), potentials, strict=True)),
        "reservoir energy",
    )
    total = total_energy - reservoir_energy
    per_atom = total / atom_count
    if not all(math.isfinite(value) for value in (atom_count, total, per_atom)):
        raise ValueError("formation energy is not finite")
    return FormationEnergy(total, per_atom, atom_count)


def chemical_potential_region(
    host_composition: Mapping[str, object],
    host_energy: float,
    competing_compositions: Sequence[Mapping[str, object]],
    competing_energies: Sequence[float],
) -> ChemicalPotentialRegion:
    """Build host equality and competing-phase chemical-potential constraints.

    The host satisfies ``sum(n_i * mu_i) = E_host``. Each competing phase adds
    ``sum(n_i * mu_i) <= E_competitor``. Add elemental reference phases as
    competing phases explicitly when their bounds are desired.

    :param host_composition: Host stoichiometric counts by element.
    :param host_energy: Host total energy in eV.
    :param competing_compositions: Competing phase compositions.
    :param competing_energies: Matching competing phase total energies in eV.
    :return: Immutable coefficient rows and right-hand-side energies.
    :raises ValueError: If compositions, energies, or row counts are invalid.
    """
    host = _composition(host_composition, "host_composition")
    host_value = _finite_scalar(host_energy, "host_energy")
    competitors = tuple(
        _composition(composition, f"competing_compositions[{index}]")
        for index, composition in enumerate(_materialize(competing_compositions, "competing_compositions"))
    )
    energies = _finite_sequence(competing_energies, "competing_energies", allow_empty=True)
    if len(competitors) != len(energies):
        raise ValueError("competing_compositions and competing_energies must have the same length")
    elements = tuple(sorted(set(host).union(*(composition.keys() for composition in competitors))))
    host_row = tuple(host.get(element, 0.0) for element in elements)
    rows = tuple(tuple(composition.get(element, 0.0) for element in elements) for composition in competitors)
    return ChemicalPotentialRegion(elements, host_row, host_value, rows, energies)


def enthalpy(
    energies: Sequence[float],
    volumes: Sequence[float],
    pressures: float | Sequence[float],
) -> tuple[float, ...]:
    """Return static ``E + P*V`` values in eV.

    Energies use eV and volumes use angstrom³ on the same extensive basis.
    Pressure MUST be given in eV/angstrom³, positive under compression; no unit
    conversion is applied. 1 GPa = 1/160.2176634 eV/angstrom³ (GPa values are
    ×160.2 too large, kbar values ×1602 too large, if passed unconverted); VASP
    prints pressure in kB, where 1 kB = 0.1 GPa. A scalar pressure
    broadcasts to all rows; a pressure vector must match the energy and volume
    vectors. The caller must decide whether each branch is comparable.

    :param energies: Non-empty one-dimensional total energies in eV.
    :param volumes: Matching positive volumes in angstrom³.
    :param pressures: Scalar pressure or matching one-dimensional pressures.
    :return: Enthalpies in input order as an immutable tuple.
    :raises ValueError: If arrays have invalid shape, values, or lengths.
    """
    energy_values = _finite_sequence(energies, "energies")
    volume_values = _finite_sequence(volumes, "volumes")
    if len(energy_values) != len(volume_values):
        raise ValueError("energies and volumes must have the same length")
    if any(volume <= 0.0 for volume in volume_values):
        raise ValueError("volumes must be positive")
    if _is_scalar(pressures):
        pressure_values = (_finite_scalar(pressures, "pressures"),) * len(energy_values)
    else:
        pressure_values = _finite_sequence(pressures, "pressures")
        if len(pressure_values) != len(energy_values):
            raise ValueError("pressure vector must match energies and volumes")
    result = tuple(
        energy + pressure * volume
        for energy, pressure, volume in zip(energy_values, pressure_values, volume_values, strict=True)
    )
    if not all(math.isfinite(value) for value in result):
        raise ValueError("enthalpies are not finite")
    return result


def _finite_sequence(values: object, name: str, *, allow_empty: bool = False) -> tuple[float, ...]:
    """Copy a finite one-dimensional real sequence to floats."""
    try:
        array = np.asarray(values)
        if array.ndim != 1 or np.iscomplexobj(array):
            raise ValueError(f"{name} must be a one-dimensional real sequence")
        result = tuple(_finite_scalar(value, name) for value in array)
    except (TypeError, OverflowError) as exc:
        raise ValueError(f"{name} must be a one-dimensional finite real sequence") from exc
    if (not result and not allow_empty) or not all(math.isfinite(value) for value in result):
        raise ValueError(f"{name} must contain finite values")
    return result


def _materialize(values: object, name: str) -> tuple[Any, ...]:
    """Copy an iterable input to a tuple or raise a consistent value error."""
    if not isinstance(values, Iterable):
        raise ValueError(f"{name} must be a sequence")
    return tuple(values)


def _finite_scalar(value: Any, name: str) -> float:
    """Convert one finite real scalar to float."""
    if isinstance(value, (str, bytes, complex, np.complexfloating)) or not np.isscalar(value):
        raise ValueError(f"{name} must be a finite real scalar")
    try:
        result = float(cast(float, value))
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be a finite real scalar") from exc
    if not math.isfinite(result):
        raise ValueError(f"{name} must be a finite real scalar")
    return result


def _finite_sum(values: Any, name: str) -> float:
    """Sum finite real values and report unrepresentable results as input errors."""
    try:
        result = math.fsum(values)
    except (OverflowError, ValueError) as exc:
        raise ValueError(f"{name} is not finite") from exc
    if not math.isfinite(result):
        raise ValueError(f"{name} is not finite")
    return result


def _nonnegative_scalar(value: Any, name: str) -> float:
    """Convert one finite non-negative scalar to float."""
    result = _finite_scalar(value, name)
    if result < 0.0:
        raise ValueError(f"{name} must be non-negative")
    return result


def _composition(values: Mapping[str, object], name: str) -> dict[str, float]:
    """Validate a non-empty stoichiometric composition."""
    if not isinstance(values, Mapping) or not values:
        raise ValueError(f"{name} must be a non-empty element-count mapping")
    result: dict[str, float] = {}
    for element, raw_count in values.items():
        if not isinstance(element, str) or not element:
            raise ValueError(f"{name} element labels must be non-empty strings")
        if isinstance(raw_count, (bool, np.bool_)):
            raise ValueError(f"{name} counts must be numeric stoichiometries, not booleans")
        count = _finite_scalar(raw_count, f"{name}[{element}]")
        if count < 0.0:
            raise ValueError(f"{name} counts must be non-negative")
        if count:
            result[element] = count
    if not result:
        raise ValueError(f"{name} must contain at least one positive count")
    return result


def _potential_vector(potentials: Mapping[str, object], elements: tuple[str, ...]) -> tuple[float, ...]:
    """Read required finite elemental potentials in a fixed order."""
    if not isinstance(potentials, Mapping):
        raise ValueError("chemical_potentials must be an element-to-potential mapping")
    missing = [element for element in elements if element not in potentials]
    if missing:
        raise ValueError(f"missing chemical-potential reservoirs: {', '.join(missing)}")
    return tuple(_finite_scalar(potentials[element], f"chemical_potentials[{element}]") for element in elements)


def _is_scalar(value: object) -> bool:
    """Whether an input is a scalar pressure rather than a vector."""
    return np.isscalar(value)
