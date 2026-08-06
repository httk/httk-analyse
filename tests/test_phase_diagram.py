"""Tests for float-LP phase-diagram construction and plotting."""

import math

import matplotlib
import pytest

matplotlib.use("Agg")

from httk.atomistic import Species, UnitcellStructure

from httk.analyse.matsci import PhaseDiagram, PhaseDiagramBuilder

CUBIC = [[4, 0, 0], [0, 4, 0], [0, 0, 4]]


@pytest.mark.parametrize("epsilon", [2e-10, 1e-9, 2e-9])
def test_phase_diagram_near_degenerate_ternary_regression(epsilon: float) -> None:
    points = [
        (0.5, 0.0, 0.5),
        (0.0, 0.5, 0.5),
        (0.25 - epsilon, 0.25, 0.5 + epsilon),
        (0.75, 0.25, 0.0),
    ]
    target = tuple(0.9 * points[0][axis] + 0.05 * points[1][axis] + 0.05 * points[2][axis] for axis in range(3))
    compositions: list[dict[str, float]] = [
        dict(zip(("A", "B", "C"), point, strict=True)) for point in [*points, target]
    ]

    diagram = PhaseDiagram.from_compositions(
        compositions,
        [0.0, 0.0, 0.0, -1.0, 0.0],
    )

    assert len(diagram) == 5
    assert diagram.elements == ("A", "B", "C")


def test_binary_hull_distances_decomposition_and_subsumed_line() -> None:
    # In per-atom coordinates A=(0, 0), B=(1, 0), AB=(0.5, -1), and
    # AB3=(0.75, -0.25). At x_B=0.75 the AB--B segment is -0.5, so AB3 is
    # 0.25 above hull and decomposes as 0.5 AB + 0.5 B.
    diagram = PhaseDiagram.from_compositions(
        [{"A": 1}, {"B": 1}, {"A": 1, "B": 1}, {"A": 1, "B": 3}],
        [0.0, 0.0, -2.0, -1.0],
        ["A", "B", "AB", "AB3"],
    )

    assert diagram.elements == ("A", "B")
    assert diagram.energies_per_atom == pytest.approx((0.0, 0.0, -1.0, -0.25))
    assert diagram.hull_indices == (0, 1, 2)
    assert diagram.energy_above_hull == pytest.approx((0.0, 0.0, 0.0, 0.25), abs=1e-9)
    decomposition = diagram.decomposition(3)
    assert decomposition is not None
    assert tuple(index for index, _ in decomposition) == (1, 2)
    assert tuple(weight for _, weight in decomposition) == pytest.approx((0.5, 0.5))
    assert diagram.phase_lines == ((0, 2), (1, 2))


def test_ternary_interior_compound_has_six_supported_lines() -> None:
    # A, B, and C have energy 0; ABC is at the barycentre with e=-1/atom.
    # Each element edge is forced by its zero third-element coordinate, while each
    # element--ABC pair is uniquely supported toward the lowered barycentre. The
    # midpoint test therefore gives all six pairs of the four stable phases.
    diagram = PhaseDiagram.from_compositions(
        [{"A": 1}, {"B": 1}, {"C": 1}, {"A": 1, "B": 1, "C": 1}],
        [0.0, 0.0, 0.0, -3.0],
    )

    assert diagram.hull_indices == (0, 1, 2, 3)
    assert diagram.phase_lines == (
        (0, 1),
        (0, 2),
        (0, 3),
        (1, 2),
        (1, 3),
        (2, 3),
    )


def test_midpoint_lines_fix_v1_single_decomposition_neighbor_gap() -> None:
    # A/B/C are the zero-energy corners. AB and AC are at -1/atom. The two
    # compounds subsume the long A--B and A--C segments. Every other pair is
    # midpoint-supported: for example midpoint(B, AC) can also be represented
    # by 0.5 AB + 0.25 B + 0.25 C, at the same -0.5 energy.
    #
    # A v1-style single competitor decomposition for AB finds only A+B, and the
    # one for AC only A+C; the pure corners are composition-extreme. That graph
    # therefore misses B--C, AB--AC, B--AC, and C--AB, all found here.
    diagram = PhaseDiagram.from_compositions(
        [
            {"A": 1},
            {"B": 1},
            {"C": 1},
            {"A": 1, "B": 1},
            {"A": 1, "C": 1},
        ],
        [0.0, 0.0, 0.0, -2.0, -2.0],
    )

    assert diagram.hull_indices == (0, 1, 2, 3, 4)
    assert diagram.phase_lines == (
        (0, 3),
        (0, 4),
        (1, 2),
        (1, 3),
        (1, 4),
        (2, 3),
        (2, 4),
        (3, 4),
    )


def test_uncontested_compositions_are_stable() -> None:
    # Neither endpoint x_B=0 nor x_B=0.5 can be represented by the other alone.
    diagram = PhaseDiagram.from_compositions(
        [{"A": 1}, {"A": 1, "B": 1}],
        [0.0, -1.0],
    )

    assert diagram.hull_indices == (0, 1)
    assert diagram.energy_above_hull == (0.0, 0.0)
    assert diagram.decomposition(1) is None


def test_duplicate_composition_keeps_only_lower_polymorph_stable() -> None:
    diagram = PhaseDiagram.from_compositions(
        [{"A": 1, "B": 1}, {"B": 2, "A": 2}],
        [-2.0, -3.0],
        ["low", "high"],
    )

    assert diagram.energies_per_atom == pytest.approx((-1.0, -0.75))
    assert diagram.hull_indices == (0,)
    assert diagram.energy_above_hull == pytest.approx((0.0, 0.25))
    assert diagram.decomposition(1) == ((0, 1.0),)
    assert diagram.phase_lines == ()


def test_duplicate_compositions_tied_within_tolerance_are_all_stable() -> None:
    diagram = PhaseDiagram.from_compositions(
        [{"A": 1}, {"A": 1}],
        [0.0, 1e-12],
    )
    assert diagram.hull_indices == (0, 1)
    assert diagram.energy_above_hull == pytest.approx((0.0, 1e-12))


def test_nonfinite_atom_count_is_rejected_before_normalization() -> None:
    with pytest.raises(ValueError, match="finite atom count"):
        PhaseDiagram.from_compositions(
            [{"A": 1e308, "B": 1e308}],
            [0.0],
        )


def test_nonfinite_energy_per_atom_is_rejected() -> None:
    with pytest.raises(ValueError, match="energy per atom must be finite"):
        PhaseDiagram.from_compositions(
            [{"A": 1e-320}],
            [1e308],
        )


def test_from_structures_weights_disorder_and_ignores_vacancy() -> None:
    mixed = UnitcellStructure(
        CUBIC,
        [[0, 0, 0]],
        [Species("mix", ("Fe", "Ni"), (0.5, 0.5))],
        ["mix"],
    )
    lithium_with_vacancy = UnitcellStructure(
        CUBIC,
        [[0, 0, 0], [0.5, 0.5, 0.5]],
        [
            Species("Li", ("Li",), (1.0,)),
            Species("vac", ("vacancy",), (1.0,)),
        ],
        ["Li", "vac"],
    )

    diagram = PhaseDiagram.from_structures([mixed, lithium_with_vacancy], [0.0, 0.0])

    assert diagram.elements == ("Fe", "Li", "Ni")
    assert diagram.compositions == ((0.5, 0.0, 0.5), (0.0, 1.0, 0.0))
    assert diagram.ids == ("Fe0.5Ni0.5", "Li")


def test_from_structures_rejects_unknown_element() -> None:
    structure = UnitcellStructure(
        CUBIC,
        [[0, 0, 0]],
        [Species("unknown", ("X",), (1.0,))],
        ["unknown"],
    )
    with pytest.raises(ValueError, match="unknown chemical symbol"):
        PhaseDiagram.from_structures([structure], [0.0])


def test_default_tolerance_counts_tiny_positive_distance_as_stable() -> None:
    # AB is 1e-12/atom above the A--B segment, well inside the default 1e-8.
    diagram = PhaseDiagram.from_compositions(
        [{"A": 1}, {"B": 1}, {"A": 1, "B": 1}],
        [0.0, 0.0, 2e-12],
    )
    assert diagram.hull_indices == (0, 1, 2)
    assert diagram.energy_above_hull[2] == pytest.approx(1e-12)


def test_distinct_nearby_compositions_receive_a_tie_line() -> None:
    diagram = PhaseDiagram.from_compositions(
        [
            {"A": 1.0},
            {"A": 1.0 - 5e-10, "B": 5e-10},
        ],
        [0.0, 0.0],
    )

    assert diagram.compositions[0] != diagram.compositions[1]
    assert diagram.phase_lines == ((0, 1),)


def test_subsumption_requires_middle_phase_on_energy_segment() -> None:
    # M lies at x_B=0.25 and 1.4e-8 below the A--B energy segment. At the A--B
    # midpoint, 2/3 M + 1/3 B has energy -9.333...e-9, so the requested 1e-8
    # stability tolerance still admits A--B as midpoint-supported. M is not on
    # the A--B energy segment within tolerance, however, and therefore must not
    # subsume that long line. All three pair lines are reported.
    diagram = PhaseDiagram.from_compositions(
        [
            {"A": 1.0},
            {"B": 1.0},
            {"A": 0.75, "B": 0.25},
        ],
        [0.0, 0.0, -1.4e-8],
        tolerance=1e-8,
    )

    assert diagram.hull_indices == (0, 1, 2)
    assert diagram.phase_lines == ((0, 1), (0, 2), (1, 2))


def test_binary_and_ternary_plot_smoke() -> None:
    import matplotlib.pyplot as plt

    binary = PhaseDiagram.from_compositions(
        [{"A": 1}, {"B": 1}, {"A": 1, "B": 1}, {"A": 1, "B": 3}],
        [0.0, 0.0, -2.0, -1.0],
    )
    binary_axes = binary.plot()
    assert len(binary_axes.lines) >= len(binary.phase_lines)
    assert tuple(binary_axes.lines[0].get_xdata()) == pytest.approx((0.0, 0.5))
    assert tuple(binary_axes.lines[0].get_ydata()) == pytest.approx((0.0, -1.0))
    assert tuple(binary_axes.lines[1].get_xdata()) == pytest.approx((0.5, 1.0))
    assert tuple(binary_axes.lines[1].get_ydata()) == pytest.approx((-1.0, 0.0))

    supplied_figure, supplied_axes = plt.subplots()
    figure_numbers = tuple(plt.get_fignums())
    assert binary.plot(ax=supplied_axes) is supplied_axes
    assert tuple(plt.get_fignums()) == figure_numbers

    ternary = PhaseDiagram.from_compositions(
        [{"A": 1}, {"B": 1}, {"C": 1}, {"A": 1, "B": 1, "C": 1}],
        [0.0, 0.0, 0.0, -3.0],
    )
    ternary_axes = ternary.plot(label_stable=False)
    assert len(ternary_axes.lines) >= len(ternary.phase_lines)
    plt.close(binary_axes.figure)
    plt.close(supplied_figure)
    plt.close(ternary_axes.figure)


def test_unknown_phases_plot_as_one_marker_line_in_binary_and_polygon() -> None:
    def none_lines(axes):
        return [line for line in axes.lines if line.get_linestyle() == "None"]

    binary_known = PhaseDiagram.from_compositions(
        [{"A": 1}, {"B": 1}, {"A": 1, "B": 1}],
        [0.0, 0.0, -2.0],
    )
    binary_unknown = PhaseDiagram.from_compositions(
        [{"A": 1}, {"B": 1}, {"A": 1, "B": 1}, {"A": 1, "B": 2}, {"A": 2, "B": 1}],
        [0.0, 0.0, -2.0, None, None],
        ids=["A", "B", "AB", "unknown-AB2", "unknown-A2B"],
    )
    binary_known_axes = binary_known.plot()
    binary_axes = binary_unknown.plot()
    binary_marker_lines = none_lines(binary_axes)
    assert len(binary_marker_lines) == len(none_lines(binary_known_axes)) + 1
    binary_unknown_line = [line for line in binary_marker_lines if line.get_marker() == "s"]
    assert len(binary_unknown_line) == 1
    assert tuple(binary_unknown_line[0].get_xdata()) == pytest.approx((2.0 / 3.0, 1.0 / 3.0))
    assert tuple(binary_unknown_line[0].get_ydata()) == pytest.approx((0.0, 0.0))
    assert binary_unknown_line[0].get_markerfacecolor() == "none"

    polygon_known = PhaseDiagram.from_compositions(
        [{"A": 1}, {"B": 1}, {"C": 1}],
        [0.0, 0.0, 0.0],
    )
    polygon_unknown = PhaseDiagram.from_compositions(
        [{"A": 1}, {"B": 1}, {"C": 1}, {"A": 1, "B": 1}, {"B": 1, "C": 1}],
        [0.0, 0.0, 0.0, None, None],
        ids=["A", "B", "C", "unknown-AB", "unknown-BC"],
    )
    polygon_known_axes = polygon_known.plot()
    polygon_axes = polygon_unknown.plot()
    polygon_marker_lines = none_lines(polygon_axes)
    assert len(polygon_marker_lines) == len(none_lines(polygon_known_axes)) + 1
    polygon_unknown_line = [line for line in polygon_marker_lines if line.get_marker() == "s"]
    assert len(polygon_unknown_line) == 1
    assert tuple(polygon_unknown_line[0].get_xdata()) == pytest.approx((0.25, -0.5))
    assert tuple(polygon_unknown_line[0].get_ydata()) == pytest.approx((math.sqrt(3.0) / 4.0, 0.0))
    assert polygon_unknown_line[0].get_markerfacecolor() == "none"

    import matplotlib.pyplot as plt

    for axes in (binary_known_axes, binary_axes, polygon_known_axes, polygon_axes):
        plt.close(axes.figure)


def test_unknown_plot_visibility_and_labels() -> None:
    diagram = PhaseDiagram.from_compositions(
        [{"A": 1}, {"B": 1}, {"A": 1, "B": 1}, {"A": 1, "B": 2}],
        [0.0, 0.0, -2.0, None],
        ids=["A", "B", "AB", "unknown-AB2"],
    )

    with_unknown = diagram.plot()
    without_unknown = diagram.plot(show_unknown=False)
    no_unknown_diagram = PhaseDiagram.from_compositions(
        [{"A": 1}, {"B": 1}, {"A": 1, "B": 1}],
        [0.0, 0.0, -2.0],
    )
    no_unknown_default = no_unknown_diagram.plot()
    no_unknown_hidden = no_unknown_diagram.plot(show_unknown=False)

    def marker_lines(axes):
        return [line for line in axes.lines if line.get_linestyle() == "None"]

    assert len(marker_lines(with_unknown)) == len(marker_lines(without_unknown)) + 1
    assert any(annotation.get_text() == "unknown-AB2" for annotation in with_unknown.texts)
    assert len(no_unknown_default.lines) == len(no_unknown_hidden.lines)

    import matplotlib.pyplot as plt

    for axes in (with_unknown, without_unknown, no_unknown_default, no_unknown_hidden):
        plt.close(axes.figure)


def test_unknown_energy_phases_use_separate_channel_and_widen_elements() -> None:
    diagram = PhaseDiagram.from_compositions(
        [{"A": 1}, {"C": 2}, {"B": 1}, {"A": 1, "C": 1}],
        [0.0, None, 0.0, None],
        ["A-phase", "C2-phase", "B-phase", "AC-phase"],
    )

    assert diagram.elements == ("A", "B", "C")
    assert diagram.ids == ("A-phase", "B-phase")
    assert diagram.compositions == ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0))
    assert diagram.energies_per_atom == (0.0, 0.0)
    assert diagram.unknown_ids == ("C2-phase", "AC-phase")
    assert diagram.unknown_compositions == ((0.0, 0.0, 1.0), (0.5, 0.0, 0.5))
    assert len(diagram) == 2
    assert diagram.hull_indices == (0, 1)
    assert diagram.energy_above_hull == (0.0, 0.0)
    assert diagram.phase_lines == ((0, 1),)
    assert diagram.decomposition(0) is None
    assert diagram.is_stable(1)


def test_interleaved_unknown_energy_preserves_known_phase_alignment() -> None:
    diagram = PhaseDiagram.from_compositions(
        [{"A": 1}, {"A": 1, "B": 1}, {"B": 1}, {"A": 1, "B": 1}, {"A": 1, "B": 3}],
        [2.0, None, 5.0, -2.0, 10.0],
        ["A", "unknown-AB", "B", "AB", "AB3"],
    )

    assert diagram.ids == ("A", "B", "AB", "AB3")
    assert diagram.energies_per_atom == pytest.approx((2.0, 5.0, -1.0, 2.5))
    assert tuple(diagram.decomposition(index) for index in range(len(diagram))) == (
        None,
        None,
        None,
        ((1, 0.5), (2, 0.5)),
    )
    assert tuple(diagram.is_stable(index) for index in range(len(diagram))) == (True, True, True, False)


def test_from_structures_accepts_unknown_energies() -> None:
    mixed = UnitcellStructure(
        CUBIC,
        [[0, 0, 0]],
        [Species("mix", ("Fe", "Ni"), (0.5, 0.5))],
        ["mix"],
    )
    lithium = UnitcellStructure(
        CUBIC,
        [[0, 0, 0]],
        [Species("Li", ("Li",), (1.0,))],
        ["Li"],
    )

    diagram = PhaseDiagram.from_structures([mixed, lithium], [None, 0.0])

    assert diagram.elements == ("Fe", "Li", "Ni")
    assert diagram.ids == ("Li",)
    assert diagram.unknown_ids == ("Fe0.5Ni0.5",)
    assert diagram.unknown_compositions == ((0.5, 0.0, 0.5),)


def test_phase_diagram_builder_matches_one_shot_factory() -> None:
    mixed = UnitcellStructure(
        CUBIC,
        [[0, 0, 0]],
        [Species("ab", ("Fe", "Ni"), (1.0, 1.0))],
        ["ab"],
    )
    builder = PhaseDiagramBuilder(tolerance=2e-8)
    assert builder.add_phase({"Fe": 1}, 0.0, "pure-A") is builder
    assert builder.add_structure(mixed, -2.0) is builder
    builder.add_phase({"Ni": 1}, 0.0)
    builder.add_phase({"C": 1}, None, "unknown-C")

    built = builder.build()
    expected = PhaseDiagram.from_compositions(
        [{"Fe": 1}, {"Fe": 1, "Ni": 1}, {"Ni": 1}, {"C": 1}],
        [0.0, -2.0, 0.0, None],
        ["pure-A", "FeNi", "Ni", "unknown-C"],
        tolerance=2e-8,
    )

    assert built.elements == expected.elements
    assert built.ids == expected.ids
    assert built.compositions == expected.compositions
    assert built.energies_per_atom == expected.energies_per_atom
    assert built.hull_indices == expected.hull_indices
    assert built.energy_above_hull == expected.energy_above_hull
    assert built.phase_lines == expected.phase_lines
    assert built.unknown_ids == expected.unknown_ids
    assert built.unknown_compositions == expected.unknown_compositions
    assert len(built) == len(expected)
    for index in range(len(built)):
        assert built.decomposition(index) == expected.decomposition(index)
        assert built.is_stable(index) == expected.is_stable(index)


def test_phase_diagram_builder_builds_independent_snapshots() -> None:
    builder = PhaseDiagramBuilder().add_phase({"A": 1}, 0.0)
    first = builder.build()

    builder.add_phase({"B": 1}, 0.0)
    second = builder.build()

    assert first.ids == ("A",)
    assert first.elements == ("A",)
    assert len(first) == 1
    assert second.ids == ("A", "B")
    assert second.elements == ("A", "B")
    assert len(second) == 2


def test_phase_diagram_builder_copies_compositions_on_add() -> None:
    composition = {"A": 1}
    builder = PhaseDiagramBuilder().add_phase(composition, 0.0, "A")
    composition["B"] = 1
    builder.add_phase({"B": 1}, 0.0, "B")

    built = builder.build()

    assert built.ids == ("A", "B")
    assert built.compositions == ((1.0, 0.0), (0.0, 1.0))


def test_phase_diagram_factory_and_builder_error_paths() -> None:
    with pytest.raises(ValueError, match="a phase diagram requires at least one phase$"):
        PhaseDiagram.from_compositions([], [])
    with pytest.raises(ValueError, match="a phase diagram requires at least one phase with known energy"):
        PhaseDiagram.from_compositions([{"A": 1}, {"B": 1}], [None, None])
    with pytest.raises(ValueError, match="ids must have the same length as compositions"):
        PhaseDiagram.from_compositions([{"A": 1}], [0.0], [])
    with pytest.raises(ValueError, match="tolerance must be a finite non-negative number"):
        PhaseDiagramBuilder(tolerance=-1.0)
