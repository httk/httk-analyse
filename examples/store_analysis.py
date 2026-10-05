"""Store a JSON analysis summary, its property records and source-file provenance in SQLite.

When the summary's result is an ``httk.analyse.matsci`` result type bound to
property definitions (for example an EOS fit), the result is rebuilt from the
summary and :func:`httk.analyse.records.records` adds one record per bound
value: a typed record per property, such as ``BulkModulusRecord`` in GPa, which
the store serves and filters as ``_httk_bulk_modulus``, and a typed statistic
record per statistic, such as ``TotalEnergyRmseRecord`` (``_httk_total_energy_rmse``)
for the RMSE of the fitted total energies. Results whose
bindings need a selection keyword (replica transport ``lag_index``) are stored
as the summary only. The store declares every generated record kind, so one
database holds the records of any result type.
"""

import argparse
import json
from importlib import import_module
from pathlib import Path
from typing import Any

from httk.core import (
    AverageTotalEnergyRecord,
    DataRecord,
    DataRecordEntry,
    DerivedDataRecord,
    FileEntry,
    FileRecord,
    PropertyDefinition,
    Run,
    RunEdge,
    RunEntry,
    TotalEnergyRecord,
    TypedRecord,
    load_property_definition,
)
from httk.core import property_records as core_records
from httk.core.storage import content_id

from httk.analyse import property_records
from httk.analyse.records import AnalysisRecord, bound_values, records
from httk.analyse.summary import AnalysisSummary

#: Every record kind ``records()`` can emit, in a fixed order (the store declaration is identity-bearing).
KINDS = (
    DataRecord,
    DerivedDataRecord,
    TotalEnergyRecord,
    AverageTotalEnergyRecord,
    *core_records.RECORD_KINDS.values(),
    *property_records.RECORD_KINDS.values(),
    *property_records.DERIVED_RECORD_KINDS.values(),
)


def _store(database: Any, sql_store: Any, entry_id_scheme: Any) -> Any:
    return sql_store(
        database,
        entry_records={DataRecordEntry: KINDS, FileEntry: FileRecord, RunEntry: Run},
        entry_ids=entry_id_scheme("httk.analyse.recipe", "1"),
    )


def _result(value: dict[str, Any]) -> object | None:
    module, _, name = value["result_type"].rpartition(".")
    # Only rebuild trusted httk result types named in the JSON; never import arbitrary modules.
    if not module.startswith("httk.analyse.matsci.") or not any(f.get("definition") for f in value["fields"].values()):
        return None
    result = getattr(import_module(module), name)(**value["result"])
    try:
        records(result)
    except TypeError:  # the bindings need an explicit selection such as lag_index
        return None
    return result


def _property_records(result: object | None, product_of: tuple[RunEdge, ...]) -> dict[str, AnalysisRecord]:
    """Return the result's records keyed by binding field name (unique, unlike definition names)."""
    if result is None:
        return {}
    fields = [bound.field for bound in bound_values(result)]
    return dict(zip(fields, records(result, product_of=product_of), strict=True))


def _value(record: AnalysisRecord) -> Any:
    if isinstance(record, TotalEnergyRecord):  # The hand-written energy records name their value field.
        return record.total_energy
    if isinstance(record, AverageTotalEnergyRecord):
        return record.average_total_energy
    return record.value


def _saved_id(store: Any, field: str, record: AnalysisRecord) -> str:
    saved = store.fetch_entry(DataRecordEntry, content_id(record))
    if saved is None or saved.id is None:
        raise RuntimeError(f"property record {field} was not saved")
    return saved.id


def main() -> None:
    """Persist, reopen, query by property name and verify pinned edges."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("summary", type=Path)
    parser.add_argument("database", type=Path)
    parser.add_argument("--source-id", help="Existing workflow job identifier recorded on the analysis run")
    parser.add_argument("--upstream-run-id", help="Public run ID already stored in this SQLite database")
    args = parser.parse_args()
    summary = AnalysisSummary(args.summary.read_text(encoding="utf-8"))
    value = summary.value
    if not isinstance(value.get("sources"), list) or not value["sources"]:
        raise ValueError("summary must include at least one hashed source file")
    definition = PropertyDefinition.from_simple(
        "_httk_analysis_summary",
        fulltype="dict",
        description="A canonical analysis result and its explicit provenance.",
    )
    entry_id_scheme = import_module("httk.store").EntryIdScheme
    sql_module = import_module("httk.store.backend.sql")
    backend, sql_store = sql_module.Backend, sql_module.SqlStore
    result = _result(value)
    with backend.sqlite(args.database) as database:
        store = _store(database, sql_store, entry_id_scheme)
        files = []
        for source in value["sources"]:
            file = FileRecord(source["url"], source["name"], size=source["size"], sha256=source["sha256"])
            store.save(file)
            saved = store.fetch_entry(FileEntry, content_id(file))
            if saved is None or saved.id is None or saved.immutable_id is None:
                raise RuntimeError("source file entry was not saved")
            files.append(saved)
        if args.upstream_run_id:
            search = store.searcher()
            upstream = search.variable(Run)
            search.add(upstream.id == args.upstream_run_id)
            if len(list(search.results(record=upstream))) != 1:
                raise ValueError("upstream run ID does not identify one stored run")
        record = DataRecord.from_value(
            definition.definition_id,
            definition.name,
            value,
            product_of=tuple(RunEdge(f"source_{i}", "files", file.id) for i, file in enumerate(files)),
        )
        store.save(record)
        saved_record = store.fetch_entry(DataRecordEntry, content_id(record))
        if saved_record is None or saved_record.id is None or saved_record.immutable_id is None:
            raise RuntimeError("analysis record was not saved")
        properties = _property_records(result, record.product_of)
        for property_record in properties.values():
            store.save(property_record)
        inputs = [RunEdge(f"source_{i}", "files", file.id) for i, file in enumerate(files)]
        if args.upstream_run_id:
            inputs.append(RunEdge("upstream_run", "runs", args.upstream_run_id))
        run = Run(
            inputs=tuple(inputs),
            outputs=(
                RunEdge("analysis_summary", "records", saved_record.id),
                *(RunEdge(field, "records", _saved_id(store, field, p)) for field, p in properties.items()),
            ),
            source_id=args.source_id or f"analysis:{content_id(record)}",
        )
        store.save(run)
        saved_run = store.fetch_entry(RunEntry, content_id(run))
        if saved_run is None or saved_run.id is None or saved_run.immutable_id is None:
            raise RuntimeError("analysis run was not saved")
        record_id, run_id = saved_record.id, saved_run.id
        file_ids = [file.id for file in files]
    with backend.sqlite(args.database) as database:
        store = _store(database, sql_store, entry_id_scheme)
        search = store.searcher()
        candidate = search.variable(DataRecord)
        search.add(candidate.name == "_httk_analysis_summary")
        matches = [row[0] for row in search.results(record=candidate) if row[0].id == record_id]
        if (
            len(matches) != 1
            or matches[0].value_json != summary.canonical_json
            or matches[0].product_of != record.product_of
        ):
            raise RuntimeError("reopened summary does not match the source JSON")
        saved_run = store.fetch_entry(RunEntry, content_id(run))
        if (
            saved_run is None
            or saved_run.id != run_id
            or saved_run.outputs != run.outputs
            or saved_run.inputs != run.inputs
        ):
            raise RuntimeError("reopened run provenance differs")
        stored_properties = {}
        for field, prop in properties.items():
            fetched = store.fetch_entry(DataRecordEntry, content_id(prop))
            if fetched != prop or _value(fetched) != _value(prop):
                raise RuntimeError(f"reopened property record {field} differs")
            stored_properties[field] = _value(prop)
        # Typed records are served: filter the store's records on the first typed property's served name.
        typed = next((p for p in properties.values() if isinstance(p, TypedRecord)), None)
        query, matched = None, []
        if typed is not None:
            query = f"{load_property_definition(typed.definition_id).served_form().name} IS KNOWN"
            searchers = store.stored_property_plan(DataRecordEntry).filter_searchers(query)
            matched = sorted(row[0].id for searcher in searchers for row in searcher.results())
            if _saved_id(store, "filtered", typed) not in matched:
                raise RuntimeError(f"filter {query!r} did not match the stored record")
        for file, file_id in zip(files, file_ids):
            fetched = store.fetch_entry(FileEntry, content_id(file))
            if fetched is None or fetched.id != file_id or fetched.sha256 != file.sha256:
                raise RuntimeError("reopened source file metadata differs")
    print(
        json.dumps(
            {
                "record_id": record_id,
                "run_id": run_id,
                "file_ids": file_ids,
                "properties": stored_properties,
                "filter": {"query": query, "matched_ids": matched},
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
