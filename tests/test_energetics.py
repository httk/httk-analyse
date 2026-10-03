"""Checks for static materials energy bookkeeping."""

from dataclasses import FrozenInstanceError

import pytest

from httk.analyse.matsci.energetics import (
    ChemicalPotentialRegion,
    ConvergenceTable,
    FormationEnergy,
    chemical_potential_region,
    convergence_table,
    enthalpy,
    formation_energy,
    reaction_energy,
)


def test_convergence_table_uses_explicit_reference_and_preserves_input_order() -> None:
    parameters = [400, 600, 500]
    energies = [-20.0, -21.0, -19.5]
    counts = [2, 2, 1]
    table = convergence_table(parameters, energies, reference_index=0, atom_counts=counts)

    assert isinstance(table, ConvergenceTable)
    assert table.parameters == (400, 600, 500)
    assert table.energies == (-20.0, -21.0, -19.5)
    assert table.atom_counts == (2, 2, 1)
    assert table.energies_per_atom == (-10.0, -10.5, -19.5)
    assert table.differences_per_atom == (0.0, -0.5, -9.5)
    assert parameters == [400, 600, 500]
    assert energies == [-20.0, -21.0, -19.5]
    with pytest.raises(FrozenInstanceError):
        table.reference_index = 1  # type: ignore[misc]


def test_public_results_copy_mutable_constructor_inputs() -> None:
    labels = ["cutoff"]
    energies = [-2.0]
    counts = [1]
    per_atom = [-2.0]
    differences = [0.0]
    table = ConvergenceTable(labels, energies, counts, per_atom, differences, 0)  # type: ignore[arg-type]
    labels.append("later")
    energies.append(3.0)
    counts.append(2)
    per_atom.append(1.5)
    differences.append(3.5)
    assert table.parameters == ("cutoff",)
    assert table.energies == (-2.0,)
    assert table.atom_counts == (1,)
    assert table.energies_per_atom == (-2.0,)
    assert table.differences_per_atom == (0.0,)

    elements = ["A"]
    host = [1.0]
    row = [0.5]
    rows = [row]
    phase_energies = [0.0]
    region = ChemicalPotentialRegion(elements, host, -1.0, rows, phase_energies)  # type: ignore[arg-type]
    elements.append("B")
    host.append(1.0)
    row[0] = 4.0
    rows.append([1.0])
    phase_energies.append(2.0)
    assert region.elements == ("A",)
    assert region.host_coefficients == (1.0,)
    assert region.competing_coefficients == ((0.5,),)
    assert region.competing_energies == (0.0,)


@pytest.mark.parametrize(
    ("parameters", "energies", "counts", "reference"),
    [
        ([1, 2], [1.0], [1, 1], 0),
        ([1], [1.0], [0], 0),
        ([1], [1.0], [1.0], 0),
        ([1], [float("nan")], [1], 0),
        ([1], [1.0], [1], 1),
        ([[1]], [1.0], [1], 0),
    ],
)
def test_convergence_table_rejects_invalid_data(
    parameters: object, energies: object, counts: object, reference: int
) -> None:
    with pytest.raises(ValueError):
        convergence_table(parameters, energies, reference_index=reference, atom_counts=counts)  # type: ignore[arg-type]


def test_reaction_energy_balances_signed_products_and_reactants() -> None:
    # 2 Al + 3/2 O2 -> Al2O3; coefficients are products-positive.
    reaction = reaction_energy(
        energies=(-3.0, 0.0, -17.0),
        coefficients=(-2.0, -1.5, 1.0),
        compositions=({"Al": 1}, {"O": 2}, {"Al": 2, "O": 3}),
    )
    assert reaction == pytest.approx(-11.0)


def test_reaction_energy_rejects_unbalanced_and_malformed_reactions() -> None:
    with pytest.raises(ValueError, match="not balanced"):
        reaction_energy([0.0, -1.0], [-1.0, 1.0], [{"A": 1}, {"A": 2}])
    with pytest.raises(ValueError):
        reaction_energy([0.0], [1.0], [{"A": 0}])


def test_formation_energy_uses_explicit_reservoirs() -> None:
    result = formation_energy(-10.0, {"A": 2, "B": 1}, {"A": -3.0, "B": -2.0})
    assert isinstance(result, FormationEnergy)
    assert (result.total, result.per_atom, result.atom_count) == pytest.approx((-2.0, -2.0 / 3.0, 3.0))
    with pytest.raises(ValueError, match="missing chemical-potential"):
        formation_energy(-10.0, {"A": 1, "B": 1}, {"A": -3.0})


def test_chemical_potential_region_exposes_and_checks_linear_constraints() -> None:
    # AB has E=-3; elemental phases each have reference energy 0.
    region = chemical_potential_region(
        {"A": 1, "B": 1},
        -3.0,
        [{"A": 1}, {"B": 1}],
        [0.0, 0.0],
    )
    assert isinstance(region, ChemicalPotentialRegion)
    assert region.elements == ("A", "B")
    assert region.host_coefficients == (1.0, 1.0)
    assert region.competing_coefficients == ((1.0, 0.0), (0.0, 1.0))
    assert region.contains({"A": -1.0, "B": -2.0})
    assert not region.contains({"A": -0.5, "B": -2.0})
    assert not region.contains({"A": -1.0, "B": -1.0})
    with pytest.raises(ValueError, match="missing chemical-potential"):
        region.contains({"A": -1.0})


def test_chemical_potential_region_allows_host_equality_without_competitors() -> None:
    region = chemical_potential_region({"A": 2}, -4.0, (), ())
    assert region.competing_coefficients == ()
    assert region.competing_energies == ()
    assert region.contains({"A": -2.0})


def test_enthalpy_uses_positive_compression_pressure_and_scalar_broadcast() -> None:
    energies = [-2.0, -2.0]
    volumes = [10.0, 12.0]
    result = enthalpy(energies, volumes, 0.1)
    assert result == pytest.approx((-1.0, -0.8))
    assert energies == [-2.0, -2.0]
    assert enthalpy([-2.0, -2.0], [10.0, 12.0], [0.1, -0.1]) == pytest.approx((-1.0, -3.2))


@pytest.mark.parametrize(
    ("energies", "volumes", "pressures"),
    [
        ([[1.0]], [1.0], 0.0),
        ([1.0], [1.0, 2.0], 0.0),
        ([1.0], [0.0], 0.0),
        ([1.0], [1.0], [0.0, 1.0]),
        ([1.0], [1.0], 1.0 + 0j),
    ],
)
def test_enthalpy_rejects_bad_shapes_or_values(energies: object, volumes: object, pressures: object) -> None:
    with pytest.raises(ValueError):
        enthalpy(energies, volumes, pressures)  # type: ignore[arg-type]


def test_result_constructors_reject_mutable_labels_and_nonfinite_values():
    from httk.analyse.matsci.energetics import ConvergenceTable, FormationEnergy

    with pytest.raises(ValueError):
        ConvergenceTable([[]], [1.0], [1], [1.0], [0.0], 0)
    with pytest.raises(ValueError):
        FormationEnergy(float('nan'), 1, 1)
    with pytest.raises(ValueError):
        formation_energy(-2, {'A': True}, {'A': -1})
