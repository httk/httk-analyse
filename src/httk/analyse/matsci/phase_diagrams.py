"""Materials-science phase diagrams built on generic lower convex hulls."""

import fractions
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Self

from httk.atomistic import StructureLike, UnitcellStructureView

from httk.analyse.generic import LowerConvexHull

__all__ = ["PhaseDiagram"]

_COMPOSITION_TOLERANCE = 1e-9


@dataclass(frozen=True, slots=True, init=False)
class PhaseDiagram:
    """A normalized float convex-hull phase diagram.

    Construct with :meth:`from_compositions` or :meth:`from_structures`. Energies supplied
    to either factory are total formula-unit or unit-cell energies and are divided by the
    corresponding atom count. All exposed compositions and energies are plain floats.

    Hull membership is delegated to :class:`~httk.analyse.generic.LowerConvexHull`. At
    duplicated compositions only the lower-energy polymorph is stable, except that every
    polymorph tied within the requested energy ``tolerance`` is stable. ``None``
    decompositions identify stable phases, including uncontested phases outside the convex
    hull of every other input.
    """

    _elements: tuple[str, ...]
    _ids: tuple[str, ...]
    _hull: LowerConvexHull

    def __init__(
        self,
        elements: tuple[str, ...],
        ids: tuple[str, ...],
        compositions: tuple[tuple[float, ...], ...],
        energies_per_atom: tuple[float, ...],
        tolerance: float,
    ) -> None:
        """Build a diagram from normalized internal data.

        Public callers normally use one of the factories, which establish material
        composition coordinates and per-atom energies.
        """
        object.__setattr__(self, "_elements", elements)
        object.__setattr__(self, "_ids", ids)
        object.__setattr__(
            self,
            "_hull",
            LowerConvexHull(compositions, energies_per_atom, tolerance=tolerance),
        )

    @classmethod
    def from_compositions(
        cls,
        compositions: Sequence[Mapping[str, int | float | fractions.Fraction]],
        energies: Sequence[float],
        ids: Sequence[str] | None = None,
        *,
        tolerance: float = 1e-8,
    ) -> Self:
        """Build a diagram from formula-unit compositions and total energies.

        Each composition maps an element label to its count in the formula unit to which
        the matching total energy refers. Counts must be finite and non-negative, with a
        strictly positive total atom count. Elements are sorted independently of mapping
        insertion order before rows are normalized to atomic fractions.
        """
        composition_rows = tuple(compositions)
        energy_values = tuple(float(energy) for energy in energies)
        if not composition_rows:
            raise ValueError("a phase diagram requires at least one phase")
        if len(composition_rows) != len(energy_values):
            raise ValueError("compositions and energies must have the same length")
        energy_tolerance = float(tolerance)
        if not math.isfinite(energy_tolerance) or energy_tolerance < 0.0:
            raise ValueError("tolerance must be a finite non-negative number")
        if not all(math.isfinite(energy) for energy in energy_values):
            raise ValueError("energies must be finite")

        elements = tuple(sorted({element for row in composition_rows for element in row}))
        if not elements:
            raise ValueError("compositions must contain at least one element")

        normalized: list[tuple[float, ...]] = []
        per_atom: list[float] = []
        for row, energy in zip(composition_rows, energy_values, strict=True):
            counts = tuple(float(row.get(element, 0.0)) for element in elements)
            if not all(math.isfinite(count) and count >= 0.0 for count in counts):
                raise ValueError("composition counts must be finite and non-negative")
            atom_count = sum(counts)
            if not math.isfinite(atom_count):
                raise ValueError("a phase composition must have a finite atom count")
            if atom_count <= 0.0:
                raise ValueError("a phase composition must have a positive atom count")
            largest_count = max(counts)
            scaled_total = sum(count / largest_count for count in counts)
            fraction_row = tuple((count / largest_count) / scaled_total for count in counts)
            if not all(math.isfinite(fraction) for fraction in fraction_row):
                raise ValueError("normalized composition fractions must be finite")
            normalized.append(fraction_row)
            normalized_energy = energy / atom_count
            if not math.isfinite(normalized_energy):
                raise ValueError("energy per atom must be finite")
            per_atom.append(normalized_energy)

        if ids is None:
            normalized_ids = tuple(_formula_label(row) for row in composition_rows)
        else:
            normalized_ids = tuple(str(identifier) for identifier in ids)
            if len(normalized_ids) != len(composition_rows):
                raise ValueError("ids must have the same length as compositions")

        return cls(
            elements,
            normalized_ids,
            tuple(normalized),
            tuple(per_atom),
            energy_tolerance,
        )

    @classmethod
    def from_structures(
        cls,
        structures: Sequence[StructureLike],
        energies: Sequence[float],
        ids: Sequence[str] | None = None,
        *,
        tolerance: float = 1e-8,
    ) -> Self:
        """Build a diagram from structures and their total unit-cell energies.

        Each site contributes its named species' ``chemical_symbols`` weighted by
        ``concentration``. ``"vacancy"`` contributes no atoms; ``"X"`` is rejected because
        an unknown element cannot define a composition coordinate. Default identifiers are
        deterministic alphabetically sorted labels using the (possibly fractional) unit-cell
        counts without reducing them.
        """
        structure_values = tuple(structures)
        composition_rows: list[dict[str, float]] = []
        for structure_like in structure_values:
            structure = UnitcellStructureView(structure_like)
            species_by_name = {species.name: species for species in structure.species}
            composition: dict[str, float] = {}
            for name in structure.species_at_sites:
                species = species_by_name[name]
                for symbol, concentration in zip(
                    species.chemical_symbols,
                    species.concentration,
                    strict=True,
                ):
                    if not math.isfinite(concentration) or concentration < 0.0:
                        raise ValueError("species concentrations must be finite and non-negative")
                    if symbol == "vacancy":
                        continue
                    if symbol == "X":
                        raise ValueError("cannot build a phase diagram from unknown chemical symbol 'X'")
                    composition[symbol] = composition.get(symbol, 0.0) + float(concentration)
            composition_rows.append(composition)

        if ids is None:
            ids = tuple(_formula_label(row) for row in composition_rows)
        return cls.from_compositions(
            composition_rows,
            energies,
            ids,
            tolerance=tolerance,
        )

    @property
    def elements(self) -> tuple[str, ...]:
        """Element-coordinate order used by :attr:`compositions`."""
        return self._elements

    @property
    def ids(self) -> tuple[str, ...]:
        """Phase identifiers in input order."""
        return self._ids

    @property
    def compositions(self) -> tuple[tuple[float, ...], ...]:
        """Per-atom composition-fraction rows in :attr:`elements` order."""
        return self._hull.points

    @property
    def energies_per_atom(self) -> tuple[float, ...]:
        """Input energies normalized by formula-unit or unit-cell atom count."""
        return self._hull.values

    @property
    def hull_indices(self) -> tuple[int, ...]:
        """Stable phase indices in input order."""
        return self._hull.hull_indices

    @property
    def energy_above_hull(self) -> tuple[float, ...]:
        """Non-negative energy distance from the lower hull in per-atom units."""
        return self._hull.value_above_hull

    @property
    def phase_lines(self) -> tuple[tuple[int, int], ...]:
        """Sorted midpoint-supported stable tie-lines as ``(smaller, larger)`` indices."""
        return self._hull.supported_segments

    def decomposition(self, index: int) -> tuple[tuple[int, float], ...] | None:
        """Optimal stable-phase ``(index, weight)`` pairs, or ``None`` when stable."""
        return self._hull.decomposition(index)

    def is_stable(self, index: int) -> bool:
        """Return whether the phase at ``index`` is stable within the energy tolerance."""
        return self._hull.is_on_hull(index)

    def __len__(self) -> int:
        """Return the number of phases."""
        return len(self._hull)

    def plot(
        self,
        *,
        ax: Any = None,
        show_unstable: bool = True,
        label_stable: bool = True,
    ) -> Any:
        """Plot the phase diagram and return its matplotlib Axes.

        Binary diagrams use the fraction of the second element on the x axis. The y axis is
        formation energy relative to the lowest stable pure-element endpoints when both are
        present, and raw energy per atom otherwise. One-element diagrams use raw energy at a
        single composition point.

        Diagrams with three or more elements use a regular composition polygon whose corner
        ``k`` is at angle ``2*pi*k/N``. Phase lines are drawn individually in black; stable
        phases are filled and unstable phases are open. Matplotlib is imported only here and
        this method never calls ``show`` or writes a file.
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError as exc:
            raise ImportError("PhaseDiagram.plot() requires matplotlib; install matplotlib") from exc

        if ax is None:
            _, ax = plt.subplots()
        if len(self._elements) <= 2:
            self._plot_binary(ax, show_unstable=show_unstable, label_stable=label_stable)
        else:
            self._plot_polygon(ax, show_unstable=show_unstable, label_stable=label_stable)
        return ax

    def _plot_binary(self, ax: Any, *, show_unstable: bool, label_stable: bool) -> None:
        binary = len(self._elements) == 2
        compositions = self.compositions
        energies = self.energies_per_atom
        xs = [row[1] if binary else 0.0 for row in compositions]
        ys = list(energies)
        formation = False
        if binary:
            left = [index for index in self.hull_indices if abs(xs[index]) <= _COMPOSITION_TOLERANCE]
            right = [index for index in self.hull_indices if abs(xs[index] - 1.0) <= _COMPOSITION_TOLERANCE]
            if left and right:
                left_energy = min(energies[index] for index in left)
                right_energy = min(energies[index] for index in right)
                ys = [
                    energy - ((1.0 - x) * left_energy + x * right_energy)
                    for x, energy in zip(xs, energies, strict=True)
                ]
                formation = True

        binary_lines = sorted(
            self.phase_lines,
            key=lambda pair: (
                min(xs[pair[0]], xs[pair[1]]),
                max(xs[pair[0]], xs[pair[1]]),
                pair,
            ),
        )
        for first, second in binary_lines:
            if xs[first] > xs[second]:
                first, second = second, first
            ax.plot(
                [xs[first], xs[second]],
                [ys[first], ys[second]],
                color="black",
                linewidth=1.5,
            )
        stable_set = set(self.hull_indices)
        stable = list(self.hull_indices)
        unstable = [index for index in range(len(self)) if index not in stable_set]
        if stable:
            ax.plot(
                [xs[index] for index in stable],
                [ys[index] for index in stable],
                linestyle="none",
                marker="o",
                color="black",
            )
        if show_unstable and unstable:
            ax.plot(
                [xs[index] for index in unstable],
                [ys[index] for index in unstable],
                linestyle="none",
                marker="o",
                markerfacecolor="none",
                markeredgecolor="black",
            )
        if label_stable:
            for index in stable:
                ax.annotate(
                    self._ids[index],
                    (xs[index], ys[index]),
                    xytext=(4, 4),
                    textcoords="offset points",
                )
        if binary:
            ax.set_xlim(-0.03, 1.03)
            ax.set_xlabel(f"fraction of {self._elements[1]}")
        else:
            ax.set_xticks([0.0], labels=[self._elements[0]])
            ax.set_xlabel("composition")
        ax.set_ylabel("formation energy / atom" if formation else "energy / atom")

    def _plot_polygon(self, ax: Any, *, show_unstable: bool, label_stable: bool) -> None:
        corners = tuple(
            (
                math.cos(2.0 * math.pi * index / len(self._elements)),
                math.sin(2.0 * math.pi * index / len(self._elements)),
            )
            for index in range(len(self._elements))
        )
        positions = tuple(
            (
                sum(weight * corner[0] for weight, corner in zip(row, corners, strict=True)),
                sum(weight * corner[1] for weight, corner in zip(row, corners, strict=True)),
            )
            for row in self.compositions
        )
        boundary = (*corners, corners[0])
        ax.plot(
            [point[0] for point in boundary],
            [point[1] for point in boundary],
            color="black",
            linewidth=1.0,
        )
        for element, corner in zip(self._elements, corners, strict=True):
            ax.text(
                1.12 * corner[0],
                1.12 * corner[1],
                element,
                horizontalalignment="center",
                verticalalignment="center",
            )
        for first, second in self.phase_lines:
            ax.plot(
                [positions[first][0], positions[second][0]],
                [positions[first][1], positions[second][1]],
                color="black",
                linewidth=1.5,
            )
        stable_set = set(self.hull_indices)
        stable = list(self.hull_indices)
        unstable = [index for index in range(len(self)) if index not in stable_set]
        if stable:
            ax.plot(
                [positions[index][0] for index in stable],
                [positions[index][1] for index in stable],
                linestyle="none",
                marker="o",
                color="black",
            )
        if show_unstable and unstable:
            ax.plot(
                [positions[index][0] for index in unstable],
                [positions[index][1] for index in unstable],
                linestyle="none",
                marker="o",
                markerfacecolor="none",
                markeredgecolor="black",
            )
        if label_stable:
            for index in stable:
                ax.annotate(
                    self._ids[index],
                    positions[index],
                    xytext=(4, 4),
                    textcoords="offset points",
                )
        ax.set_aspect("equal", adjustable="box")
        ax.set_axis_off()


def _formula_label(composition: Mapping[str, int | float | fractions.Fraction]) -> str:
    """Return a compact deterministic unit-cell formula label."""
    parts: list[str] = []
    for element in sorted(composition):
        count = float(composition[element])
        if count <= 0.0:
            continue
        parts.append(element)
        if not math.isclose(count, 1.0, rel_tol=0.0, abs_tol=1e-12):
            parts.append(f"{count:.12g}")
    return "".join(parts) or "empty"
