"""Generic lower convex-hull analysis for finite point-and-value collections."""

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from ._simplex import _LPInfeasibleError, _solve_equality_lp

__all__ = ["LowerConvexHull"]

_COORDINATE_TOLERANCE = 1e-9


@dataclass(frozen=True, slots=True, init=False)
class LowerConvexHull:
    """The lower convex hull of scalar values over arbitrary finite coordinates.

    A point belongs to the hull when its value is no greater than the cheapest
    leave-one-out convex mixture at the same coordinates, within ``tolerance``.
    The input order is retained throughout, including for tied duplicate points.
    """

    _points: tuple[tuple[float, ...], ...]
    _values: tuple[float, ...]
    _tolerance: float
    _hull_indices: tuple[int, ...]
    _value_above_hull: tuple[float, ...]
    _decompositions: tuple[tuple[tuple[int, float], ...] | None, ...]
    _supported_segments: tuple[tuple[int, int], ...]

    def __init__(
        self,
        points: Sequence[Sequence[float]],
        values: Sequence[float],
        *,
        tolerance: float = 1e-8,
    ) -> None:
        """Construct and analyze a lower hull.

        Coordinates and values are converted through ``numpy.float64`` and exposed as
        ordinary :class:`float` values. Every coordinate equality is retained in the
        mixture LP, together with the affine ``sum(weights) == 1`` equality.
        """
        point_rows = tuple(points)
        value_rows = tuple(values)
        if not point_rows:
            raise ValueError("a lower convex hull requires at least one point")
        if len(point_rows) != len(value_rows):
            raise ValueError("points and values must have the same length")

        normalized_points: list[tuple[float, ...]] = []
        dimension: int | None = None
        for point in point_rows:
            try:
                coordinates = tuple(point)
            except TypeError as exc:
                raise ValueError("every point must be a coordinate sequence") from exc
            if dimension is None:
                dimension = len(coordinates)
            elif len(coordinates) != dimension:
                raise ValueError("points must have consistent coordinate dimensions")
            normalized_points.append(
                tuple(_finite_float(coordinate, "point coordinates") for coordinate in coordinates)
            )
        normalized_values = tuple(_finite_float(value, "values") for value in value_rows)
        numeric_tolerance = _finite_float(tolerance, "tolerance")
        if numeric_tolerance < 0.0:
            raise ValueError("tolerance must be a finite non-negative number")

        object.__setattr__(self, "_points", tuple(normalized_points))
        object.__setattr__(self, "_values", normalized_values)
        object.__setattr__(self, "_tolerance", numeric_tolerance)
        self._analyze()

    @property
    def points(self) -> tuple[tuple[float, ...], ...]:
        """Input coordinates, preserved in their original order as float tuples."""
        return self._points

    @property
    def values(self) -> tuple[float, ...]:
        """Input scalar values in the same order as :attr:`points`."""
        return self._values

    @property
    def hull_indices(self) -> tuple[int, ...]:
        """Indices of points on the lower hull, in input order."""
        return self._hull_indices

    @property
    def value_above_hull(self) -> tuple[float, ...]:
        """Non-negative leave-one-out value excess for every input point."""
        return self._value_above_hull

    @property
    def supported_segments(self) -> tuple[tuple[int, int], ...]:
        """Midpoint-supported pairs of distinct lower-hull points."""
        return self._supported_segments

    def decomposition(self, index: int) -> tuple[tuple[int, float], ...] | None:
        """Return lower-hull mixture ``(index, weight)`` pairs, or ``None`` on hull."""
        return self._decompositions[index]

    def is_on_hull(self, index: int) -> bool:
        """Return whether ``index`` belongs to :attr:`hull_indices`."""
        return index in self._hull_indices

    def __len__(self) -> int:
        """Return the number of input points."""
        return len(self._points)

    def _mixture(
        self,
        indices: Sequence[int],
        target: tuple[float, ...],
    ) -> tuple[float, tuple[float, ...]]:
        # These origin-relative rows are equivalent to coordinate equality together
        # with sum(weights) == 1, but do not lose precision when every point is
        # translated far from the origin. Every coordinate is still represented.
        # Use a candidate origin rather than the target itself: that leaves the
        # simplex a nonzero right-hand side on ordinary mixtures and avoids a
        # needless degenerate phase-I path.
        origin = self._points[indices[0]] if indices else target
        matrix = [[self._points[index][axis] - origin[axis] for index in indices] for axis in range(len(target))]
        matrix.append([1.0] * len(indices))
        rhs = [target[axis] - origin[axis] for axis in range(len(target))]
        rhs.append(1.0)
        costs = [self._values[index] for index in indices]
        baseline = min(costs, default=0.0)
        value, weights = _solve_equality_lp(
            [cost - baseline for cost in costs],
            matrix,
            rhs,
        )
        return baseline + value, weights

    def _analyze(self) -> None:
        point_count = len(self)
        hull: list[int] = []
        above_hull: list[float] = []
        for index in range(point_count):
            competitors = tuple(candidate for candidate in range(point_count) if candidate != index)
            try:
                value, _ = self._mixture(competitors, self._points[index])
            except _LPInfeasibleError:
                hull.append(index)
                above_hull.append(0.0)
                continue
            difference = self._values[index] - value
            above_hull.append(max(0.0, difference))
            if difference <= self._tolerance:
                hull.append(index)

        object.__setattr__(self, "_hull_indices", tuple(hull))
        object.__setattr__(self, "_value_above_hull", tuple(above_hull))

        decompositions: list[tuple[tuple[int, float], ...] | None] = []
        hull_set = set(hull)
        for index in range(point_count):
            if index in hull_set:
                decompositions.append(None)
                continue
            try:
                _, weights = self._mixture(hull, self._points[index])
            except _LPInfeasibleError as exc:
                raise RuntimeError("lower-hull points do not span a non-hull point") from exc
            decompositions.append(
                tuple((hull[position], weight) for position, weight in enumerate(weights) if weight > 0.0)
            )
        object.__setattr__(self, "_decompositions", tuple(decompositions))

        segments: list[tuple[int, int]] = []
        for position, first in enumerate(hull):
            for second in hull[position + 1 :]:
                first_point = self._points[first]
                second_point = self._points[second]
                if first_point == second_point:
                    continue
                value, _ = self._midpoint_mixture(hull, first, second)
                pair_value = (self._values[first] + self._values[second]) / 2.0
                if pair_value > value + self._tolerance:
                    continue
                if self._is_subsumed(first, second, hull):
                    continue
                segments.append((first, second))
        object.__setattr__(self, "_supported_segments", tuple(segments))

    def _midpoint_mixture(
        self,
        indices: Sequence[int],
        first: int,
        second: int,
    ) -> tuple[float, tuple[float, ...]]:
        """Return the lower-hull value at a pair midpoint without forming it absolutely."""
        origin = self._points[first]
        endpoint = self._points[second]
        matrix = [[self._points[index][axis] - origin[axis] for index in indices] for axis in range(len(origin))]
        matrix.append([1.0] * len(indices))
        rhs = [(endpoint[axis] - origin[axis]) / 2.0 for axis in range(len(origin))]
        rhs.append(1.0)
        costs = [self._values[index] for index in indices]
        baseline = min(costs, default=0.0)
        value, weights = _solve_equality_lp(
            [cost - baseline for cost in costs],
            matrix,
            rhs,
        )
        return baseline + value, weights

    def _is_subsumed(self, first: int, second: int, hull: Sequence[int]) -> bool:
        left = self._points[first]
        right = self._points[second]
        difference = tuple(a - b for a, b in zip(left, right, strict=True))
        if not difference:
            return False
        axis = max(range(len(difference)), key=lambda index: abs(difference[index]))
        if abs(difference[axis]) <= _COORDINATE_TOLERANCE:
            return False
        for middle in hull:
            if middle == first or middle == second:
                continue
            candidate = self._points[middle]
            if _rows_close(candidate, left) or _rows_close(candidate, right):
                continue
            fraction = (candidate[axis] - right[axis]) / difference[axis]
            if not _COORDINATE_TOLERANCE < fraction < 1.0 - _COORDINATE_TOLERANCE:
                continue
            candidate_offset = tuple(value - base for value, base in zip(candidate, right, strict=True))
            segment_offset = tuple(fraction * value for value in difference)
            if not _rows_close(candidate_offset, segment_offset):
                continue
            energy = self._values[second] + fraction * (self._values[first] - self._values[second])
            if abs(self._values[middle] - energy) <= self._tolerance:
                return True
        return False


def _finite_float(value: Any, label: str) -> float:
    """Coerce a scalar through float64 and reject non-finite values."""
    try:
        result = float(np.float64(value))
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{label} must be finite") from exc
    if not math.isfinite(result):
        raise ValueError(f"{label} must be finite")
    return result


def _rows_close(
    first: Sequence[float],
    second: Sequence[float],
    tolerance: float = _COORDINATE_TOLERANCE,
) -> bool:
    """Return whether coordinate rows are close at the segment tolerance."""
    return all(abs(left - right) <= tolerance for left, right in zip(first, second, strict=True))
