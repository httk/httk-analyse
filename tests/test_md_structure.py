"""Checks for periodic geometry and static structure statistics."""

import itertools
from collections.abc import Iterator

import numpy as np
import pytest

from httk.analyse.matsci.structure import (
    bond_angles,
    coordination_numbers,
    minimum_image,
    radial_distribution,
    static_structure_factor,
)


def _brute_minimum(displacement: np.ndarray, cell: np.ndarray, pbc: tuple[bool, bool, bool]) -> np.ndarray:
    fractional = displacement @ np.linalg.inv(cell)
    ranges = [range(round(fractional[i]) - 12, round(fractional[i]) + 13) if pbc[i] else range(1) for i in range(3)]
    return min(
        (displacement - np.asarray(shift) @ cell for shift in itertools.product(*ranges)),
        key=np.linalg.norm,
    )


def test_skew_cell_minimum_image_matches_integer_brute_force() -> None:
    cell = np.array([[1.0, 0.0, 0.0], [0.9, 0.2, 0.0], [0.0, 0.0, 4.0]])
    vector = np.array([-0.9535514630027344, -0.8060354263435068, 1.2569029623771213])
    got = np.asarray(minimum_image(vector, cell))
    fractional_rounding = vector - np.round(vector @ np.linalg.inv(cell)) @ cell

    assert np.linalg.norm(got) < np.linalg.norm(fractional_rounding)
    assert got == pytest.approx(_brute_minimum(vector, cell, (True, True, True)))


def test_mixed_periodicity_preserves_nonperiodic_component() -> None:
    cell = np.array([[1.0, 0.0, 0.0], [0.9, 0.2, 0.0], [0.0, 0.0, 4.0]])
    vector = np.array([-0.7, -0.3, 8.0])
    got = np.asarray(minimum_image(vector, cell, (True, True, False)))
    assert got[2] == 8.0
    assert got == pytest.approx(_brute_minimum(vector, cell, (True, True, False)))


def test_minimum_image_rejects_unbounded_exact_search() -> None:
    cell = np.array([[1.0, 0.0, 0.0], [0.999999, 0.000001, 0.0], [0.0, 0.0, 1.0]])
    with pytest.raises(ValueError, match="too ill-conditioned"):
        minimum_image([0.4, 0.3, 0.0], cell)


def test_minimum_image_matches_optional_ase_geometry_reference() -> None:
    geometry = pytest.importorskip("ase.geometry")
    cell = np.array([[3.0, 0.0, 0.0], [2.6, 1.1, 0.0], [0.2, 0.3, 4.0]])
    vectors = np.random.default_rng(21).normal(size=(20, 3)) * 4.0
    expected, _ = geometry.find_mic(vectors, cell, pbc=(True, True, True))

    got = np.asarray(minimum_image(vectors, cell))
    assert np.linalg.norm(got, axis=1) == pytest.approx(np.linalg.norm(expected, axis=1))


def test_coordination_and_angles_use_true_images_and_preserve_center_order() -> None:
    cell = np.eye(3) * 6.0
    positions = np.asarray(list(itertools.product((0.0, 2.0, 4.0), repeat=3)))

    assert coordination_numbers(positions, cell, 2.01) == (6,) * len(positions)
    angles = bond_angles(positions, cell, 2.01)
    assert angles[0].count(90.0) == 12
    assert angles[0].count(180.0) == 3
    assert all(len(center_angles) == 15 for center_angles in angles)


def test_partial_rdf_uses_directed_finite_n_pair_count_and_shell_volume() -> None:
    cell = np.eye(3) * 10.0
    positions = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [5.0, 0.0, 0.0], [6.0, 0.0, 0.0]])
    result = radial_distribution([positions], cell, [0.0, 1.5, 2.0], species=("A", "B", "A", "B"), pair=("A", "B"))
    expected_pairs = 2 * 2
    shell_volumes = 4.0 * np.pi / 3.0 * (np.array([1.5, 2.0]) ** 3 - np.array([0.0, 1.5]) ** 3)

    assert result.hist_counts == (2, 0)
    assert result.g == pytest.approx((2.0 / (expected_pairs / 1000.0 * shell_volumes[0]), 0.0))
    assert result.frame_count == 1
    assert result.mean_coordination == pytest.approx(1.0)
    aa = radial_distribution([positions], cell, [0.0, 1.5, 2.0], species=("A", "B", "A", "B"), pair=("A", "A"))
    assert aa.hist_counts == (0, 0)


def test_total_rdf_uses_n_times_n_minus_one_and_streams_single_pass() -> None:
    cell = np.eye(3) * 12.0
    rng = np.random.default_rng(801)

    class Frames:
        def __init__(self) -> None:
            self.used = False

        def __iter__(self) -> Iterator[np.ndarray]:
            assert not self.used
            self.used = True
            for _ in range(3):
                yield rng.uniform(0.0, 12.0, size=(160, 3))

    result = radial_distribution(Frames(), cell, [2.0, 3.0, 4.0, 5.0], periodicity=(True, True, True))
    assert result.frame_count == 3
    assert result.g == pytest.approx((1.0, 1.0, 1.0), abs=0.16)
    assert result.mean_coordination == pytest.approx(sum(result.hist_counts) / (3 * 160))


def test_streamed_cells_must_match_frame_count() -> None:
    frame = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    cells = iter((np.eye(3) * 10.0, np.eye(3) * 10.0))
    with pytest.raises(ValueError, match="exactly one"):
        radial_distribution([frame], cells, [0.0, 1.0])
    with pytest.raises(ValueError, match="ended before"):
        radial_distribution([frame, frame], iter((np.eye(3) * 10.0,)), [0.0, 1.0])


def test_structure_factor_matches_direct_complex_sum_and_keeps_q_zero() -> None:
    positions = np.array([[0.0, 0.0, 0.0], [0.5, 0.0, 0.0], [1.0, 0.0, 0.0]])
    wavevectors = np.array([[0.0, 0.0, 0.0], [np.pi, 0.0, 0.0]])
    weights = np.array([1.0, 2.0, -1.0])
    phase = wavevectors @ positions.T
    expected = tuple(abs(np.sum(weights * np.exp(1j * row))) ** 2 / np.dot(weights, weights) for row in phase)
    result = static_structure_factor(positions, wavevectors, weights)

    assert result == pytest.approx(expected)
    assert static_structure_factor(positions, wavevectors[:1]) == (3.0,)


@pytest.mark.parametrize(
    "call",
    [
        lambda: minimum_image([[1.0, 2.0]], np.eye(3)),
        lambda: minimum_image([1.0, 2.0, 3.0], np.zeros((3, 3))),
        lambda: minimum_image([1.0, complex(2, 1), 3.0], np.eye(3)),
        lambda: coordination_numbers([[0.0, 0.0, np.inf]], np.eye(3), 0.1),
        lambda: coordination_numbers([[0.0, 0.0, 0.0]], np.eye(3), 0.6),
        lambda: radial_distribution([[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]], np.eye(3) * 10, [0.0, 6.0]),
        lambda: radial_distribution(
            [[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]], np.eye(3), [0.0, 1.0], periodicity=(True, True, False)
        ),
        lambda: radial_distribution(
            [[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]], np.eye(3) * 10, [0.0, 1.0], species=("A", "B")
        ),
        lambda: radial_distribution(
            [[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]], np.eye(3) * 10, [0.0, 1.0], species=("A", "B"), pair=("A", "X")
        ),
        lambda: static_structure_factor([[0.0, 0.0, 0.0]], [[0.0, 0.0, 0.0]], [0.0]),
        lambda: static_structure_factor([[0.0, 0.0, 0.0]], [[0.0, 0.0, 0.0]], [complex(1, 1)]),
    ],
)
def test_invalid_geometry_and_structure_inputs_raise_value_error(call: object) -> None:
    with pytest.raises(ValueError):
        call()  # type: ignore[operator]


def test_inputs_are_not_mutated_and_results_are_tuple_backed() -> None:
    positions = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    cell = np.eye(3) * 10
    original_positions, original_cell = positions.copy(), cell.copy()
    result = radial_distribution([positions], cell, [0.0, 2.0])

    assert np.array_equal(positions, original_positions)
    assert np.array_equal(cell, original_cell)
    assert isinstance(result.edges, tuple)
    assert isinstance(result.g, tuple)
    assert isinstance(minimum_image(positions, cell), tuple)


def test_half_cell_bond_angle_cutoff_is_ambiguous():
    with pytest.raises(ValueError, match="below half"):
        bond_angles([[0, 0, 0], [1, 0, 0], [0.6, 0.8, 0]], np.eye(3) * 2, 1)


def test_orthogonal_vectorized_images_match_enumeration_with_partial_pbc():
    from httk.analyse.matsci.structure import _minimum_image_vector

    cell = np.diag([4.0, 6.0, 8.0])
    inverse = np.linalg.inv(cell)
    vectors = np.random.default_rng(331).uniform(-25, 25, size=(50, 3))
    pbc = (True, False, True)
    expected = np.array([_minimum_image_vector(row, cell, inverse, pbc) for row in vectors])
    np.testing.assert_allclose(minimum_image(vectors, cell, pbc), expected, atol=1e-13)
    np.testing.assert_allclose(np.asarray(minimum_image(vectors, cell, pbc))[:, 1], vectors[:, 1])
