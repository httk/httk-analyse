"""Tests for generic lower convex-hull construction and its private LP solver."""

import random

import numpy
import pytest

from httk.analyse.generic import LowerConvexHull
from httk.analyse.generic._simplex import _solve_equality_lp


def test_equality_simplex_unique_optimum() -> None:
    value, weights = _solve_equality_lp([1.0, 2.0], [[1.0, 1.0]], [1.0])
    assert value == pytest.approx(1.0)
    assert weights == pytest.approx((1.0, 0.0))


def test_equality_simplex_bland_rule_handles_multirow_degenerate_pivots() -> None:
    # Beale's cycling example, expressed with three equality slacks. The first
    # two basic slack values start at zero, so this exercises genuinely
    # degenerate multi-row pivots rather than merely a non-unique objective.
    value, weights = _solve_equality_lp(
        [-10.0, 57.0, 9.0, 24.0, 0.0, 0.0, 0.0],
        [
            [0.5, -5.5, -2.5, 9.0, 1.0, 0.0, 0.0],
            [0.5, -1.5, -0.5, 1.0, 0.0, 1.0, 0.0],
            [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
        ],
        [0.0, 0.0, 1.0],
    )
    assert value == pytest.approx(-1.0)
    assert weights[:4] == pytest.approx((1.0, 0.0, 1.0, 0.0))


def test_equality_simplex_reports_infeasible_problem() -> None:
    with pytest.raises(ValueError, match="infeasible"):
        _solve_equality_lp([1.0], [[1.0], [1.0]], [1.0, 2.0])


def test_equality_simplex_guards_unbounded_problem() -> None:
    # Hull objectives cannot be unbounded because sum(w) == 1 and w >= 0. Keep the
    # general solver guard covered with a free negative-cost variable.
    with pytest.raises(ValueError, match="unbounded"):
        _solve_equality_lp([-1.0], [[0.0]], [0.0])


@pytest.mark.parametrize("epsilon", [2e-10, 1e-9, 2e-9])
def test_equality_simplex_near_degenerate_ternary_regression(epsilon: float) -> None:
    points = numpy.asarray(
        [
            (0.5, 0.0, 0.5),
            (0.0, 0.5, 0.5),
            (0.25 - epsilon, 0.25, 0.5 + epsilon),
            (0.75, 0.25, 0.0),
        ],
        dtype=numpy.float64,
    )
    target = 0.9 * points[0] + 0.05 * points[1] + 0.05 * points[2]
    matrix = numpy.vstack((points.T, numpy.ones(4)))

    value, weights = _solve_equality_lp(
        [0.0, 0.0, 0.0, -1.0],
        matrix,
        [*target, 1.0],
    )

    assert numpy.isfinite(value)
    assert numpy.asarray(weights) @ points == pytest.approx(target, abs=1e-10)
    assert sum(weights) == pytest.approx(1.0)


def test_equality_simplex_ratio_test_and_row_scaling_invariance() -> None:
    matrix = [[1.0, 0.0, 1.0], [0.0, 1.0, 1e12]]
    rhs = [1.0 + 5e-12, 1e12]
    costs = [0.0, 0.0, -1.0]

    value, weights = _solve_equality_lp(costs, matrix, rhs)
    scaled_value, scaled_weights = _solve_equality_lp(
        costs,
        [[1.0, 0.0, 1.0], [0.0, 1e-12, 1.0]],
        [1.0 + 5e-12, 1.0],
    )

    assert value == pytest.approx(-1.0)
    assert weights == pytest.approx((5e-12, 0.0, 1.0), abs=1e-15)
    assert scaled_value == pytest.approx(value)
    assert scaled_weights == pytest.approx(weights)


def test_equality_simplex_retains_small_improvements_next_to_large_costs() -> None:
    value, weights = _solve_equality_lp(
        [1e-3, 1e12, 0.0],
        [[1.0, 1.0, 1.0]],
        [1.0],
    )

    assert value == pytest.approx(0.0)
    assert weights == pytest.approx((0.0, 0.0, 1.0))


def test_lower_hull_keeps_every_coordinate_equality() -> None:
    # The final y coordinate makes point 4 a mixture of points 2 and 3. The old
    # composition-specific algorithm dropped this coordinate and instead mixed points 0
    # and 1, which is impossible in the full coordinate space.
    hull = LowerConvexHull(
        [(0.0, 0.0), (1.0, 0.0), (0.0, 1.0), (1.0, 1.0), (0.5, 1.0)],
        [0.0, 0.0, 2.0, 2.0, 2.5],
    )

    assert hull.hull_indices == (0, 1, 2, 3)
    assert hull.value_above_hull == pytest.approx((0.0, 0.0, 0.0, 0.0, 0.5))
    assert hull.decomposition(4) == ((2, 0.5), (3, 0.5))


def test_lower_hull_constraints_are_translation_invariant() -> None:
    translated_triangle = LowerConvexHull(
        [
            (1_000_000.0, 1_000_000.0),
            (1_000_001.0, 1_000_000.0),
            (1_000_000.0, 1_000_001.0),
        ],
        [0.0, 0.0, 0.0],
    )
    translated_line = LowerConvexHull(
        [(1e12,), (1e12 + 1.0,)],
        [0.0, -1.0],
    )

    assert translated_triangle.hull_indices == (0, 1, 2)
    assert translated_triangle.value_above_hull == (0.0, 0.0, 0.0)
    assert translated_line.hull_indices == (0, 1)
    assert translated_line.value_above_hull == (0.0, 0.0)


def test_supported_midpoint_uses_endpoint_relative_coordinates() -> None:
    hull = LowerConvexHull(
        [(1_000_000.1, 1_000_000.2), (1_000_000.3, 1_000_000.4)],
        [0.0, 0.0],
    )

    assert hull.hull_indices == (0, 1)
    assert hull.supported_segments == ((0, 1),)


def test_translated_segment_topology_matches_the_origin_topology() -> None:
    points = ((0.1, 0.2), (0.3, 0.4), (0.2, 0.3))
    values = (0.0, 0.0, 0.0)
    reference = LowerConvexHull(points, values)
    random_source = random.Random(0)

    for _ in range(8):
        offset = (
            1_000_000.0 + 1_000.0 * random_source.random(),
            1_000_000.0 + 1_000.0 * random_source.random(),
        )
        translated = LowerConvexHull(
            [(point[0] + offset[0], point[1] + offset[1]) for point in points],
            values,
        )

        assert translated.hull_indices == reference.hull_indices
        assert translated.supported_segments == reference.supported_segments


def test_lower_hull_distances_decomposition_and_subsumed_segment() -> None:
    hull = LowerConvexHull(
        [(1.0, 0.0), (0.0, 1.0), (0.5, 0.5), (0.25, 0.75)],
        [0.0, 0.0, -1.0, -0.25],
    )

    assert hull.points == ((1.0, 0.0), (0.0, 1.0), (0.5, 0.5), (0.25, 0.75))
    assert hull.values == pytest.approx((0.0, 0.0, -1.0, -0.25))
    assert hull.hull_indices == (0, 1, 2)
    assert hull.value_above_hull == pytest.approx((0.0, 0.0, 0.0, 0.25), abs=1e-9)
    decomposition = hull.decomposition(3)
    assert decomposition is not None
    assert tuple(index for index, _ in decomposition) == (1, 2)
    assert tuple(weight for _, weight in decomposition) == pytest.approx((0.5, 0.5))
    assert hull.supported_segments == ((0, 2), (1, 2))


def test_ternary_interior_point_has_six_supported_segments() -> None:
    hull = LowerConvexHull(
        [(1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0), (1 / 3, 1 / 3, 1 / 3)],
        [0.0, 0.0, 0.0, -1.0],
    )

    assert hull.hull_indices == (0, 1, 2, 3)
    assert hull.supported_segments == (
        (0, 1),
        (0, 2),
        (0, 3),
        (1, 2),
        (1, 3),
        (2, 3),
    )


def test_midpoint_segments_cover_all_supported_coplanar_pairs() -> None:
    hull = LowerConvexHull(
        [
            (1.0, 0.0, 0.0),
            (0.0, 1.0, 0.0),
            (0.0, 0.0, 1.0),
            (0.5, 0.5, 0.0),
            (0.5, 0.0, 0.5),
        ],
        [0.0, 0.0, 0.0, -1.0, -1.0],
    )

    assert hull.hull_indices == (0, 1, 2, 3, 4)
    assert hull.supported_segments == (
        (0, 3),
        (0, 4),
        (1, 2),
        (1, 3),
        (1, 4),
        (2, 3),
        (2, 4),
        (3, 4),
    )


def test_uncontested_points_are_on_hull() -> None:
    hull = LowerConvexHull([(0.0,), (0.5,)], [0.0, -0.5])

    assert hull.hull_indices == (0, 1)
    assert hull.value_above_hull == (0.0, 0.0)
    assert hull.decomposition(1) is None
    assert hull.is_on_hull(1)
    assert not hull.is_on_hull(2)
    assert len(hull) == 2


def test_duplicate_coordinates_keep_only_lower_value_on_hull() -> None:
    hull = LowerConvexHull([(0.5, 0.5), (0.5, 0.5)], [-1.0, -0.75])

    assert hull.hull_indices == (0,)
    assert hull.value_above_hull == pytest.approx((0.0, 0.25))
    assert hull.decomposition(1) == ((0, 1.0),)
    assert hull.supported_segments == ()


def test_duplicate_coordinates_tied_within_tolerance_are_all_on_hull() -> None:
    hull = LowerConvexHull([(0.0,), (0.0,)], [0.0, 1e-12])

    assert hull.hull_indices == (0, 1)
    assert hull.value_above_hull == pytest.approx((0.0, 1e-12))
    assert hull.decomposition(0) is None
    assert hull.decomposition(1) is None


@pytest.mark.parametrize(
    "values",
    [
        (5e-12, 0.0, 3e-12),
        (3e-12, 5e-12, 0.0),
        (0.0, 3e-12, 5e-12),
    ],
)
@pytest.mark.parametrize("tolerance", [0.0, 1e-12])
def test_tiny_value_differences_find_the_true_duplicate_minimum(
    values: tuple[float, float, float],
    tolerance: float,
) -> None:
    hull = LowerConvexHull([(0.0,), (0.0,), (0.0,)], values, tolerance=tolerance)
    minimum = values.index(0.0)

    assert hull.hull_indices == (minimum,)
    assert hull.value_above_hull == pytest.approx(values)


@pytest.mark.parametrize(
    "values",
    [
        (1e-3, 1e12, 0.0, 5e-4),
        (1e12, 5e-4, 1e-3, 0.0),
    ],
)
def test_duplicate_dynamic_range_values_keep_only_the_true_minimum(
    values: tuple[float, float, float, float],
) -> None:
    hull = LowerConvexHull([(0.0,), (0.0,), (0.0,), (0.0,)], values, tolerance=0.0)
    minimum = values.index(0.0)

    assert hull.hull_indices == (minimum,)
    assert hull.value_above_hull == pytest.approx(values)


def test_decomposition_keeps_small_weights_for_exact_reconstruction() -> None:
    hull = LowerConvexHull([(0.0,), (1.0,), (5e-11,)], [0.0, 0.0, 1.0])
    decomposition = hull.decomposition(2)

    assert decomposition is not None
    assert tuple(index for index, _ in decomposition) == (0, 1)
    assert tuple(weight for _, weight in decomposition) == pytest.approx((1.0 - 5e-11, 5e-11))
    assert sum(weight for _, weight in decomposition) == pytest.approx(1.0)
    assert sum(hull.points[index][0] * weight for index, weight in decomposition) == pytest.approx(5e-11)


def test_default_tolerance_counts_tiny_positive_distance_as_on_hull() -> None:
    hull = LowerConvexHull([(0.0,), (1.0,), (0.5,)], [0.0, 0.0, 1e-12])

    assert hull.hull_indices == (0, 1, 2)
    assert hull.value_above_hull[2] == pytest.approx(1e-12)


def test_distinct_nearby_coordinates_receive_a_supported_segment() -> None:
    hull = LowerConvexHull([(0.0,), (5e-10,)], [0.0, 0.0])

    assert hull.points[0] != hull.points[1]
    assert hull.supported_segments == ((0, 1),)


def test_subsumption_requires_middle_point_on_value_segment() -> None:
    hull = LowerConvexHull(
        [(0.0,), (1.0,), (0.25,)],
        [0.0, 0.0, -1.4e-8],
        tolerance=1e-8,
    )

    assert hull.hull_indices == (0, 1, 2)
    assert hull.supported_segments == ((0, 1), (0, 2), (1, 2))


@pytest.mark.parametrize(
    ("points", "values", "message"),
    [
        ([], [], "at least one point"),
        ([(0.0,)], [], "same length"),
        ([(0.0,), (0.0, 1.0)], [0.0, 1.0], "consistent coordinate dimensions"),
        ([(float("nan"),)], [0.0], "point coordinates must be finite"),
        ([(0.0,)], [float("inf")], "values must be finite"),
    ],
)
def test_lower_hull_rejects_invalid_inputs(
    points: list[tuple[float, ...]],
    values: list[float],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        LowerConvexHull(points, values)


@pytest.mark.parametrize("tolerance", [float("nan"), float("inf"), -1e-8])
def test_lower_hull_rejects_invalid_tolerance(tolerance: float) -> None:
    with pytest.raises(ValueError, match="tolerance must be .*finite"):
        LowerConvexHull([(0.0,)], [0.0], tolerance=tolerance)


def test_lower_hull_is_immutable_and_tuple_backed() -> None:
    hull = LowerConvexHull([[0], [1]], [0, 0])

    assert hull.points == ((0.0,), (1.0,))
    assert hull.values == (0.0, 0.0)
    with pytest.raises(AttributeError):
        hull._points = ()  # type: ignore[misc]
