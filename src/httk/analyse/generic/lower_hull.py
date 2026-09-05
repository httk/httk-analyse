"""Generic lower convex-hull analysis for finite point-and-value collections."""

import math
from collections.abc import Sequence
from dataclasses import dataclass, field
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

    Coordinates and values are converted through ``numpy.float64`` and exposed as
    ordinary :class:`float` values. Every coordinate equality is retained in the
    mixture LP, together with the affine ``sum(weights) == 1`` equality.

    :param points: Coordinate rows for the input points.
    :param values: Scalar values corresponding to ``points``.
    :param tolerance: Maximum value excess treated as on the lower hull.
    :raises ValueError: If the points, values, or tolerance are invalid.
    """

    _points: tuple[tuple[float, ...], ...]
    _values: tuple[float, ...]
    _tolerance: float
    _hull_indices: tuple[int, ...]
    _value_above_hull: tuple[float, ...]
    _decompositions: tuple[tuple[tuple[int, float], ...] | None, ...]
    _supported_segments: tuple[tuple[int, int], ...] | None = field(
        init=False, default=None, compare=False, hash=False, repr=False
    )

    def __init__(
        self,
        points: Sequence[Sequence[float]],
        values: Sequence[float],
        *,
        tolerance: float = 1e-8,
    ) -> None:
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
        object.__setattr__(self, "_supported_segments", None)
        self._analyze()

    @property
    def points(self) -> tuple[tuple[float, ...], ...]:
        """Return input coordinates in their original order as float tuples.

        :return: The immutable input coordinate rows.
        """
        return self._points

    @property
    def values(self) -> tuple[float, ...]:
        """Return input scalar values in the same order as :attr:`points`.

        :return: The immutable input values.
        """
        return self._values

    @property
    def hull_indices(self) -> tuple[int, ...]:
        """Return indices of points on the lower hull in input order.

        :return: The immutable lower-hull indices.
        """
        return self._hull_indices

    @property
    def value_above_hull(self) -> tuple[float, ...]:
        """Return non-negative leave-one-out value excesses for every input point.

        :return: The immutable value excesses in input order.
        """
        return self._value_above_hull

    @property
    def supported_segments(self) -> tuple[tuple[int, int], ...]:
        """Return midpoint-supported pairs of distinct lower-hull points.

        :return: The immutable supported index pairs in input order.
        """
        segments = self._supported_segments
        if segments is None:
            segments = self._compute_supported_segments()
            object.__setattr__(self, "_supported_segments", segments)
        return segments

    def decomposition(self, index: int) -> tuple[tuple[int, float], ...] | None:
        """Return lower-hull mixture ``(index, weight)`` pairs, or ``None`` on hull.

        :param index: Input point index.
        :return: The stable-point mixture, or ``None`` when the point is on the hull.
        """
        return self._decompositions[index]

    def is_on_hull(self, index: int) -> bool:
        """Return whether ``index`` belongs to :attr:`hull_indices`.

        :param index: Input point index.
        :return: Whether the point is on the lower hull.
        """
        return index in self._hull_indices

    def __len__(self) -> int:
        """Return the number of input points.

        :return: The number of input points.
        """
        return len(self._points)

    def _mixture_lp(
        self,
        indices: Sequence[int],
        origin: tuple[float, ...],
        offsets: Sequence[float],
    ) -> tuple[float, tuple[float, ...]]:
        # These origin-relative rows are equivalent to coordinate equality together
        # with sum(weights) == 1, but do not lose precision when every point is
        # translated far from the origin. Every coordinate is still represented.
        matrix = [[self._points[index][axis] - origin[axis] for index in indices] for axis in range(len(origin))]
        matrix.append([1.0] * len(indices))
        costs = [self._values[index] for index in indices]
        baseline = min(costs, default=0.0)
        value, weights = _solve_equality_lp(
            [cost - baseline for cost in costs],
            matrix,
            [*offsets, 1.0],
        )
        return baseline + value, weights

    def _mixture(
        self,
        indices: Sequence[int],
        target: tuple[float, ...],
    ) -> tuple[float, tuple[float, ...]]:
        # Use a candidate origin rather than the target itself: that leaves the
        # simplex a nonzero right-hand side on ordinary mixtures and avoids a
        # needless degenerate phase-I path.
        origin = self._points[indices[0]] if indices else target
        offsets = [target[axis] - origin[axis] for axis in range(len(target))]
        return self._mixture_lp(indices, origin, offsets)

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

    def _compute_supported_segments(self) -> tuple[tuple[int, int], ...]:
        geometric_points = _normalized_geometry(self._points)
        hull = self._hull_indices
        segments: list[tuple[int, int]] = []
        for position, first in enumerate(hull):
            for second in hull[position + 1 :]:
                first_point = self._points[first]
                second_point = self._points[second]
                if first_point == second_point:
                    continue
                # Evaluate the hull at the pair midpoint without forming it absolutely.
                midpoint_offsets = [(second_point[axis] - first_point[axis]) / 2.0 for axis in range(len(first_point))]
                value, _ = self._mixture_lp(hull, first_point, midpoint_offsets)
                pair_value = (self._values[first] + self._values[second]) / 2.0
                if pair_value > value + self._tolerance:
                    continue
                if self._is_subsumed(first, second, hull, geometric_points):
                    continue
                segments.append((first, second))
        return tuple(segments)

    def _is_subsumed(
        self,
        first: int,
        second: int,
        hull: Sequence[int],
        geometric_points: Sequence[Sequence[float]],
    ) -> bool:
        left = geometric_points[first]
        right = geometric_points[second]
        difference = tuple(a - b for a, b in zip(left, right, strict=True))
        if not difference:
            return False
        axis = max(range(len(difference)), key=lambda index: abs(difference[index]))
        length_scale = abs(difference[axis])
        if length_scale == 0.0:
            return False
        coordinate_tolerance = _COORDINATE_TOLERANCE * length_scale
        for middle in hull:
            if middle == first or middle == second:
                continue
            candidate = geometric_points[middle]
            if _rows_close(candidate, left, coordinate_tolerance) or _rows_close(
                candidate, right, coordinate_tolerance
            ):
                continue
            fraction = (candidate[axis] - right[axis]) / difference[axis]
            if not _COORDINATE_TOLERANCE < fraction < 1.0 - _COORDINATE_TOLERANCE:
                continue
            candidate_offset = tuple(value - base for value, base in zip(candidate, right, strict=True))
            segment_offset = tuple(fraction * value for value in difference)
            if not _rows_close(candidate_offset, segment_offset, coordinate_tolerance):
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


def _normalized_geometry(points: Sequence[Sequence[float]]) -> tuple[tuple[float, ...], ...]:
    """Return origin-relative, per-axis normalized coordinates for geometric predicates."""
    if not points:
        return ()
    origins = tuple(min(point[axis] for point in points) for axis in range(len(points[0])))
    ranges = tuple(max(point[axis] for point in points) - origin for axis, origin in enumerate(origins))
    return tuple(
        tuple(
            (coordinate - origins[axis]) / ranges[axis] if ranges[axis] else 0.0
            for axis, coordinate in enumerate(point)
        )
        for point in points
    )
