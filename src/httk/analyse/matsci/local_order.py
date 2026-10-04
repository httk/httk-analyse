"""Optional spherical-harmonic bond order with explicit neighbor geometry."""

import math
from collections.abc import Sequence
from dataclasses import dataclass
from importlib import import_module
from typing import Any

import numpy as np

from .. import definitions as defs
from ..definitions import BoundValue, _reject_selection, _series
from .structure import _cell, _check_unique_cutoff, _neighbor_vectors, _periodicity, _positions, _positive_scalar

__all__ = ["BondOrder", "bond_order"]


@dataclass(frozen=True, slots=True)
class BondOrder:
    """Rotational invariants of local and pooled directed neighbor bonds.

    :param degree: Spherical-harmonic degree l.
    :param local: Per-atom q_l, or None for an isolated atom.
    :param global_order: Q_l from pooled directed bonds, or None without bonds.
    :param coordination: Number of neighbor bonds per atom.
    :param cutoff: Inclusive neighbor radius in angstrom used to find the bonds.
    """

    degree: int
    local: tuple[float | None, ...]
    global_order: float | None
    coordination: tuple[int, ...]
    cutoff: float

    def __post_init__(self) -> None:
        """Copy per-atom data into immutable tuples."""
        object.__setattr__(self, "local", tuple(None if v is None else float(v) for v in self.local))
        object.__setattr__(self, "coordination", tuple(int(v) for v in self.coordination))
        object.__setattr__(self, "cutoff", float(self.cutoff))

    def _bound_values(self, **selection: Any) -> tuple[BoundValue, ...]:
        """Bind the Steinhardt bond order; isolated atoms and an empty bond set stay ``None``."""
        _reject_selection(self, selection)
        return (
            _series(
                defs.STEINHARDT_BOND_ORDER,
                degree=self.degree,
                cutoff=self.cutoff,
                global_order=self.global_order,
                local_orders=list(self.local),
                coordination_numbers=list(self.coordination),
            ),
        )


def bond_order(
    positions: object,
    cell: object,
    cutoff: float,
    degree: int,
    periodicity: Sequence[bool] = (True, True, True),
) -> BondOrder:
    """Compute Steinhardt q_l per atom and bond-weighted global Q_l.

    Averages complex Y_lm over directed bonds, then takes
    ``sqrt(4*pi/(2*l+1) * sum_m abs(mean(Y_lm))**2)``. No neighbor-of-neighbor
    averaging or crystalline/liquid classification is implied.

    The global Q_l pools directed bonds, so each pair contributes both r and -r. Since
    Y_lm(-r) = (-1)^l Y_lm(r), the global value is identically zero (to rounding) for odd l;
    it is meaningful for even l, as in Steinhardt's definition. Per-atom q_l remains valid
    for odd l. The global value is bond-weighted (Steinhardt), not atom-averaged as in
    Lechner-Dellago or pyscal, so it differs from those codes for non-uniform coordination.

    :param positions: Cartesian positions (N,3) in angstrom.
    :param cell: Nonsingular row-vector cell in angstrom.
    :param cutoff: Positive inclusive neighbor radius, strictly below the unique-image limit.
    :param degree: Nonnegative integer harmonic degree, commonly four or six.
    :param periodicity: Periodicity per cell direction.
    :return: Local q_l and directed-bond-weighted global Q_l.
    :raises ImportError: If the optional SciPy spherical-harmonic implementation is unavailable.
    :raises ValueError: If geometry or degree is invalid, or distinct atoms coincide.
    """
    if isinstance(degree, bool) or not isinstance(degree, int) or degree < 0:
        raise ValueError("degree must be a nonnegative integer")
    try:
        harmonic = import_module("scipy.special").sph_harm_y
    except (ImportError, AttributeError) as exc:
        raise ImportError("bond_order requires SciPy with sph_harm_y; install httk-analyse[scipy]") from exc
    xyz = _positions(positions)
    basis, inverse = _cell(cell)
    pbc = _periodicity(periodicity)
    radius = _positive_scalar(cutoff, "cutoff")
    _check_unique_cutoff(radius, basis, inverse, pbc)
    factor = 4 * math.pi / (2 * degree + 1)
    total = np.zeros(2 * degree + 1, dtype=complex)
    local: list[float | None] = []
    counts: list[int] = []
    for index in range(len(xyz)):
        vectors = _neighbor_vectors(xyz, index, radius, basis, inverse, pbc)
        counts.append(len(vectors))
        if not len(vectors):
            local.append(None)
            continue
        norms = np.linalg.norm(vectors, axis=1)
        if np.any(norms == 0):
            raise ValueError("bond angles are undefined for coincident atoms")
        polar = np.arccos(np.clip(vectors[:, 2] / norms, -1, 1))
        azimuth = np.mod(np.arctan2(vectors[:, 1], vectors[:, 0]), 2 * math.pi)
        values = np.asarray([harmonic(degree, m, polar, azimuth).sum() for m in range(-degree, degree + 1)])
        total += values
        local.append(float(np.sqrt(factor * np.sum(np.abs(values / len(vectors)) ** 2))))
    count = sum(counts)
    global_order = float(np.sqrt(factor * np.sum(np.abs(total / count) ** 2))) if count else None
    return BondOrder(degree, tuple(local), global_order, tuple(counts), radius)
