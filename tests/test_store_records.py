"""Typed analysis records stored in an httk-store sqlite store and filtered through its OPTIMADE SQL plan.

*httk-analyse* does not depend on *httk-store*: this module skips when it is not installed (module-only CI);
the shared workspace environment has it and runs these tests.
"""

from collections.abc import Iterator
from typing import Any

import numpy as np
import pytest

pytest.importorskip("httk.store")
pytest.importorskip("scipy")

from httk.core import RunEdge
from httk.core.data_records import DataRecord, DataRecordEntry, DerivedDataRecord
from httk.store import EntryIdScheme, FilterTranslationError
from httk.store.backend.sql import Backend, SqlStore, stored_property_sql_plan
from httk.store.storage_layout import family_entry_type_definition

from httk.analyse.matsci import PhaseDiagram, fit_birch_murnaghan
from httk.analyse.matsci.local_order import bond_order
from httk.analyse.matsci.mlip import energy_errors
from httk.analyse.property_records import (
    BornEffectiveChargesRecord,
    ElectronicBandStructureRecord,
    PhononBandStructureRecord,
    PhononDensityOfStatesRecord,
    PiezoelectricTensorRecord,
)
from httk.analyse.records import records

VOLUMES = np.linspace(8.5, 11.5, 9)


def _bm3(b0: float) -> Any:
    q = (10.0 / VOLUMES) ** (2 / 3) - 1
    return -5.0 + 9 * 10.0 * b0 / 16 * (2 * q**2 + 0.5 * q**3)


def _hull(ids: list[str], mixed: float = -1.0) -> PhaseDiagram:
    return PhaseDiagram.from_compositions([{"A": 1}, {"B": 1}, {"A": 1, "B": 1}], [0.0, 0.0, mixed], ids=ids)


POSITIONS = [[0, 0, 0], [0.5, 0, 0], [5, 5, 5]]
RESULTS = {
    "soft": fit_birch_murnaghan(VOLUMES, _bm3(0.2)),  # B0 = 32 GPa
    "stiff": fit_birch_murnaghan(VOLUMES, _bm3(0.6)),  # B0 = 96 GPa
    "conf_small": energy_errors([0.0, 0.0], [0.0, 0.004], atom_counts=[1, 4]),
    "conf_big": energy_errors([0.0, 0.0], [0.0, 0.4], atom_counts=[1, 4]),
    "atom_small": energy_errors([0.0, 0.0], [0.0, 0.004], atom_counts=[1, 4], weighting="atom"),
    "q6": bond_order(POSITIONS, np.eye(3) * 10, 1, 6),
    "q4": bond_order(POSITIONS, np.eye(3) * 10, 1, 4),
    "hull_ab": _hull(["A", "B", "AB"]),
    "hull_ba": _hull(["A", "B", "BA"]),
    "hull_up": _hull(["A", "B", "AB"], mixed=1.0),  # AB above the hull: unstable
}
_KPOINTS = [[0.0, 0.0, 0.0], [0.25, 0.0, 0.25], [0.5, 0.0, 0.5]]
_FREQUENCIES = [[0.0, 2.1, 3.9], [0.0, 2.2, 4.0], [0.0, 3.4, 5.8], [8.1, 8.0, 7.2], [8.1, 8.3, 7.9], [9.6, 9.1, 8.4]]
# Values of the band-structure, phonon and dielectric-response kinds, stored directly as typed records
# (their producers are collect hooks, not httk-analyse results): label -> (record class, value).
TYPED_VALUES: dict[str, tuple[type, Any]] = {
    "bands_spin": (
        ElectronicBandStructureRecord,
        {
            "kpoints": _KPOINTS,
            "kpoint_label_indices": [0, 2],
            "kpoint_labels": ["GAMMA", "X"],
            "energies": [[-5.1, -4.2, -3.0], [1.2, 2.3, 3.4]],
            "energies_spin_down": [[-5.0, -4.1, -2.9], [1.3, 2.4, 3.5]],
            "fermi_energy": 0.4,
        },
    ),
    "bands": (
        ElectronicBandStructureRecord,
        {
            "kpoints": _KPOINTS,
            "kpoint_label_indices": [0, 2],
            "kpoint_labels": ["\\Gamma", "X"],
            "energies": [[-6.25, -5.5, -4.75], [0.5, 1.75, 2.0]],
            "fermi_energy": -0.125,
        },
    ),
    "bands_unlabelled": (
        ElectronicBandStructureRecord,
        {
            "kpoints": _KPOINTS,
            "kpoint_label_indices": [],
            "kpoint_labels": [],
            "energies": [[-7.0, -6.5, -6.0]],
            "fermi_energy": -1.0,
        },
    ),
    "phonon_bands": (
        PhononBandStructureRecord,
        {
            "qpoints": _KPOINTS,
            "qpoint_label_indices": [0, 2],
            "qpoint_labels": ["GAMMA", "X"],
            "frequencies": _FREQUENCIES,
        },
    ),
    "phonon_dos": (
        PhononDensityOfStatesRecord,
        {"frequencies": [-0.5, 0.0, 4.5, 9.0], "density": [0.0, 0.125, 0.75, 0.5]},
    ),
    "born": (
        BornEffectiveChargesRecord,
        [
            [[2.1, 0.01, 0.02], [0.03, 2.2, 0.04], [0.05, 0.06, 2.3]],
            [[-2.1, -0.01, -0.02], [-0.03, -2.2, -0.04], [-0.05, -0.06, -2.3]],
        ],
    ),
    "piezo": (
        PiezoelectricTensorRecord,
        [[0.0, 0.0, 0.0, 0.11, 0.12, 0.13], [0.21, 0.22, 0.23, 0.24, 0.25, 0.26], [-0.31, -0.32, 0.33, 0.0, 0.0, 0.0]],
    ),
}
LABELS = set(RESULTS) | set(TYPED_VALUES)


def _edges(label: str) -> list[RunEdge]:
    return [RunEdge("source", "runs", f"ext-{label}")]


# Each result's records carry an edge naming it, so equal values of different results stay distinct entries.
RECORDS = [
    *(record for label, result in RESULTS.items() for record in records(result, product_of=_edges(label))),
    *(cls.from_value(value, product_of=_edges(label)) for label, (cls, value) in TYPED_VALUES.items()),
]
KINDS = (DataRecord, DerivedDataRecord, *sorted({type(r) for r in RECORDS} - {DerivedDataRecord}, key=str))
ENTRY_RECORDS = {DataRecordEntry: KINDS}


@pytest.fixture(scope="module")
def store() -> Iterator[SqlStore]:
    with Backend.sqlite() as database:
        writer = SqlStore(database, entry_records=ENTRY_RECORDS, entry_ids=EntryIdScheme("httk.test", "1"))
        for record in RECORDS:
            writer.save(record)
        yield SqlStore(database, entry_records=ENTRY_RECORDS, entry_ids=EntryIdScheme("httk.test", "1"))


def _labels(store: SqlStore, filter_string: str) -> set[str]:
    layout = next(item for item in store.entry_layout if item.family is DataRecordEntry)
    plan = stored_property_sql_plan(store, DataRecordEntry, served=family_entry_type_definition(layout).served_form())
    return {
        row[0].product_of[0].entry_id.removeprefix("ext-")
        for searcher in plan.filter_searchers(filter_string)
        for row in searcher.results()
    }


def _stored(store: SqlStore, kind: type) -> list[Any]:
    search = store.searcher()
    return [row.record for row in search.results(record=search.variable(kind))]


def test_typed_records_round_trip(store: SqlStore) -> None:
    stored = [record for kind in KINDS for record in _stored(store, kind)]
    assert len(stored) == len(RECORDS) and set(stored) == set(RECORDS)
    for record in stored:
        assert record.value == next(original for original in RECORDS if original == record).value


@pytest.mark.parametrize("label", sorted(TYPED_VALUES))
def test_band_structure_phonon_and_response_values_round_trip(store: SqlStore, label: str) -> None:
    cls, value = TYPED_VALUES[label]
    (stored,) = (record for record in _stored(store, cls) if record.product_of == tuple(_edges(label)))
    assert stored.value == value  # nested lists back, absent optional members stay absent
    name = cls.__httk_typed_record__.definition.name
    assert cls.__httk_typed_record__.served_name == f"_httk_{name}"
    assert list(vars(cls)["__httk_property_definitions__"]) == [f"_httk_{name}"]


HULL = "_httk_convex_hull_phase_diagram"


@pytest.mark.parametrize(
    ("filter_string", "expected"),
    (
        ("_httk_bulk_modulus > 50", {"stiff"}),
        ("_httk_bulk_modulus < 50", {"soft"}),
        (
            '_httk_energy_prediction_errors.weighting = "configuration" AND _httk_energy_prediction_errors.rmse < 0.005',
            {"conf_small"},
        ),
        ("_httk_energy_prediction_errors.rmse < 0.005", {"conf_small", "atom_small"}),
        ("_httk_steinhardt_bond_order.degree = 6", {"q6"}),
        # Statistics of a fixed base are typed derived kinds, served under synthesized names.
        ("_httk_total_energy_per_atom_rmse < 0.01", {"conf_small"}),
        ("_httk_total_energy_rmse IS KNOWN", {"soft", "stiff"}),
        ('_httk_convex_hull_phase_diagram.phase_ids HAS "AB"', {"hull_ab", "hull_up"}),
        # Zip filters correlate members sharing the phases dimension position by position.
        (f'{HULL}.phase_ids:{HULL}.stable HAS "AB":TRUE', {"hull_ab"}),
        (f'{HULL}.phase_ids:{HULL}.stable HAS "AB":FALSE', {"hull_up"}),
        (f'{HULL}.phase_ids:{HULL}.stable HAS "BA":FALSE', set()),
        (f'{HULL}.phase_ids:{HULL}.stable HAS ALL "A":TRUE, "AB":TRUE', {"hull_ab"}),
        (f'{HULL}.phase_ids:{HULL}.stable HAS ANY "BA":TRUE, "AB":FALSE', {"hull_ba", "hull_up"}),
        (f'{HULL}.phase_ids:{HULL}.stable HAS ONLY "A":TRUE, "B":TRUE, "AB":TRUE', {"hull_ab"}),
        (f'{HULL}.phase_ids:{HULL}.energies_above_hull_per_atom HAS "AB":>0.1', {"hull_up"}),
        (f'NOT {HULL}.phase_ids:{HULL}.stable HAS "AB":TRUE', {"hull_ba", "hull_up"}),
        # The band-structure, phonon and response kinds are served under their definition names.
        ("_httk_electronic_band_structure.fermi_energy > 0", {"bands_spin"}),
        ("_httk_electronic_band_structure.energies_spin_down IS KNOWN", {"bands_spin"}),
        ('_httk_electronic_band_structure.kpoint_labels HAS "GAMMA"', {"bands_spin"}),
        ("_httk_phonon_band_structure.frequencies IS KNOWN", {"phonon_bands"}),
        ("_httk_phonon_density_of_states.density IS KNOWN", {"phonon_dos"}),
        ("_httk_born_effective_charges IS KNOWN", {"born"}),
        ("_httk_piezoelectric_tensor IS KNOWN", {"piezo"}),
    ),
)
def test_filters_select_strict_subsets(store: SqlStore, filter_string: str, expected: set[str]) -> None:
    assert expected < LABELS
    assert _labels(store, filter_string) == expected


def test_zip_over_different_dimensions_is_not_implemented(store: SqlStore) -> None:
    with pytest.raises(FilterTranslationError) as excinfo:
        _labels(store, f'{HULL}.phase_ids:{HULL}.elements HAS "A":"A"')
    assert excinfo.value.category == "not-implemented"
