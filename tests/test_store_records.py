"""Typed analysis records stored in an httk-store sqlite store and filtered through its OPTIMADE SQL plan.

*httk-analyse* does not depend on *httk-store*: this module skips when it is not installed (module-only CI);
the shared workspace environment has it and runs these tests.
"""

from collections.abc import Iterator
from typing import Any

import numpy as np
import pytest

pytest.importorskip("httk.store")

from httk.core import RunEdge
from httk.core.data_records import DataRecord, DataRecordEntry, DerivedDataRecord
from httk.store import EntryIdScheme
from httk.store.backend.sql import Backend, SqlStore, stored_property_sql_plan
from httk.store.storage_layout import family_entry_type_definition

from httk.analyse.matsci import PhaseDiagram, fit_birch_murnaghan
from httk.analyse.matsci.local_order import bond_order
from httk.analyse.matsci.mlip import energy_errors
from httk.analyse.records import records

VOLUMES = np.linspace(8.5, 11.5, 9)


def _bm3(b0: float) -> Any:
    q = (10.0 / VOLUMES) ** (2 / 3) - 1
    return -5.0 + 9 * 10.0 * b0 / 16 * (2 * q**2 + 0.5 * q**3)


def _hull(ids: list[str]) -> PhaseDiagram:
    return PhaseDiagram.from_compositions([{"A": 1}, {"B": 1}, {"A": 1, "B": 1}], [0.0, 0.0, -1.0], ids=ids)


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
}
# Each result's records carry an edge naming it, so equal values of different results stay distinct entries.
RECORDS = [
    record
    for label, result in RESULTS.items()
    for record in records(result, product_of=[RunEdge("source", "runs", f"ext-{label}")])
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
        ('_httk_convex_hull_phase_diagram.phase_ids HAS "AB"', {"hull_ab"}),
    ),
)
def test_filters_select_strict_subsets(store: SqlStore, filter_string: str, expected: set[str]) -> None:
    assert expected < set(RESULTS)
    assert _labels(store, filter_string) == expected
