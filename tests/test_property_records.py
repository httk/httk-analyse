"""The generated analyse typed records are current, registered, and round-trip every definition's shape."""

import itertools
import json
import runpy
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest
from httk.core import load_property_definition
from httk.core._typed_record_tool import check_typed_records, generate_typed_records
from httk.core.register.entries import known_entry_records, resolve_entry_record

from httk.analyse.property_records import DERIVED_RECORD_KINDS, RECORD_KINDS

TOOL = runpy.run_path(str(Path(__file__).resolve().parents[1] / "tools" / "generate_records.py"))
KINDS = sorted([*RECORD_KINDS.values(), *DERIVED_RECORD_KINDS.values()], key=lambda cls: cls.__name__)


def test_generated_module_is_current():
    source = generate_typed_records(
        TOOL["DEFINITION_IDS"],
        derived=TOOL["DERIVED"],
        module_doc=TOOL["MODULE_DOC"],
        storage_prefix=TOOL["STORAGE_PREFIX"],
        command=TOOL["COMMAND"],
    )
    assert check_typed_records(TOOL["TARGET"], source), f"stale or missing; regenerate with: {TOOL['COMMAND']}"


def test_tool_check_passes():
    tool = Path(TOOL["__file__"])
    subprocess.run([sys.executable, str(tool), "--check"], cwd=tool.parents[1], check=True)


def test_every_definition_has_one_registered_kind():
    assert len(TOOL["DEFINITION_IDS"]) == 71
    assert list(RECORD_KINDS) == TOOL["DEFINITION_IDS"]
    assert list(DERIVED_RECORD_KINDS) == [(base, derivation) for base, derivation, _ in TOOL["DERIVED"]]
    assert len(set(KINDS)) == len(RECORD_KINDS) + len(DERIVED_RECORD_KINDS)
    expected = {load_property_definition(iri).name: cls for iri, cls in RECORD_KINDS.items()} | {
        f"{load_property_definition(base).name}_{derivation.rsplit('/', 1)[1]}": cls
        for (base, derivation), cls in DERIVED_RECORD_KINDS.items()
    }
    names = {"analyse-" + name.replace("_", "-"): cls for name, cls in expected.items()}
    assert set(known_entry_records("records")) >= set(names)
    assert {n for n in known_entry_records() if n.startswith("analyse-")} == set(names)
    for name, cls in names.items():
        assert resolve_entry_record(name) is cls


def test_discovery_registers_the_tier_lazily_before_core():
    script = """
import sys
import httk.core
from httk.core import load_property_definition
from httk.core.register.entries import resolve_entry_record
order = list(sys.modules)
assert order.index("httk.registry.entries.analyse") < order.index("httk.registry.entries.core"), order
assert "httk.analyse.property_records" not in sys.modules
loaded = load_property_definition.cache_info().currsize
from httk.analyse.property_records import BulkModulusRecord
assert load_property_definition.cache_info().currsize == loaded, "importing the records loaded definitions"
assert resolve_entry_record("analyse-bulk-modulus") is BulkModulusRecord
"""
    subprocess.run([sys.executable, "-c", script], check=True)


def _value(cls: type, *, full: bool) -> Any:
    """Return a checked synthetic value: every member present (*full*), or optional members omitted and
    nullable scalar members null, with distinct sizes per variable dimension and enums respected."""
    spec = cls.__httk_typed_record__
    members = spec.definition.as_optimade().get("properties", {})
    counter = itertools.count()
    variable = iter(range(2, 100))
    sizes: dict[str, int] = {}
    out: dict[str, Any] = {}
    for item in spec.layout:
        if item.path and not full and (not item.required or (item.nullable and not item.dimensions)):
            if item.required:
                out[item.field] = None
            continue
        enum = members.get(item.field, {}).get("enum")
        dims = [size if size is not None else sizes.setdefault(name, next(variable)) for name, size in item.dimensions]

        def build(depth: int, item: Any = item, enum: Any = enum, dims: list[int] = dims) -> Any:
            if depth < len(dims):
                return [build(depth + 1) for _ in range(dims[depth])]
            if item.leaf in ("float", "integer"):
                return (1.5 if item.leaf == "float" else 2) + next(counter)
            return enum[0] if enum else ("s" if item.leaf == "string" else True)

        value = build(0)
        if item.element_nullable:
            value[0] = None
        if not item.path:
            return value
        out[item.field] = value
    return out


def test_synthetic_values_cover_every_shape():
    layouts = [item for cls in RECORD_KINDS.values() for item in cls.__httk_typed_record__.layout]
    top_level = {item.field for item in layouts if not item.path and item.dimensions}
    assert top_level == {
        "total_magnetic_moment",
        "diffusion_tensor",
        "thermal_conductivity_tensor",
        "elastic_tensor",
        "compliance_tensor",
    }
    assert {item.field for item in layouts if len(item.dimensions) == 3} == {
        "diffusion_tensors",
        "msd",
        "thermal_conductivity_tensors",
        "vacf",
    }
    variable_inner = {item.field for item in layouts if any(size is None for _, size in item.dimensions[1:])}
    assert variable_inner == {"compositions", "competing_coefficients", "real", "imaginary"}
    assert {item.field for item in layouts if item.element_nullable} == {"local_orders"}
    assert any(not item.required for item in layouts)


@pytest.mark.parametrize("full", [True, False], ids=["all-members", "optional-omitted"])
@pytest.mark.parametrize("cls", KINDS, ids=lambda cls: cls.__name__)
def test_kind_round_trips_a_checked_value(cls, full):
    value = _value(cls, full=full)
    spec = cls.__httk_typed_record__
    spec.served_definition.check(value)
    load_property_definition(spec.definition_id).check(value)
    record = cls.from_value(value)
    assert json.loads(json.dumps(record.value)) == json.loads(json.dumps(value))
    assert cls.from_value(record.value) == record
    stored_name = spec.served_name.removeprefix("_httk_")
    assert vars(cls)["__httk_storage__"].storage_name == f"analyse_{stored_name}"
    assert list(vars(cls)["__httk_property_definitions__"]) == [spec.served_name]
    assert spec.served_definition.name == spec.served_name
    assert (record.definition_id, record.derivation) == (spec.definition_id, spec.derivation)
    if spec.derivation is None:
        assert RECORD_KINDS[spec.definition_id] is cls and stored_name == spec.definition.name
    else:
        assert DERIVED_RECORD_KINDS[spec.definition_id, spec.derivation] is cls
        assert stored_name == f"{spec.definition.name}_{spec.derivation.rsplit('/', 1)[1]}"
        assert spec.served_definition.definition_id.startswith("https://schemas.httk.org/ad-hoc/")
