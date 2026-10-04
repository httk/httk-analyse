"""Turn analysis results into property-definition data records.

Bound result types know which of their fields carry which property definition
(see :class:`httk.analyse.definitions.FieldBinding`). :func:`records` emits one
record per bound value, validated against its definition: a
:class:`httk.core.DataRecord` for a property value (scalar, fixed-size or a
dictionary-valued series), and a :class:`httk.core.DerivedDataRecord` for a
statistic (``derivation``) of a base property. Selection keywords are explicit
per result type: ``lag_index`` selects the plateau of transport results
(optional for a single run, required for replica statistics).
"""

from collections.abc import Iterable, Mapping
from typing import Any

from httk.core import DataRecord, DerivedDataRecord, RunEdge, load_property_definition

from .definitions import BoundValue

__all__ = ["bound_values", "records"]


def bound_values(result: object, **selection: Any) -> tuple[BoundValue, ...]:
    r"""Return every bound value of an analysis result, including series and statistics.

    :param result: An analysis result of a bound type, such as ``BirchMurnaghanFit``.
    :param \*\*selection: Result-specific selection keywords, such as ``lag_index`` for transport results.
    :return: Bound values in binding order.
    :raises TypeError: If the result type has no bindings or a selection keyword is missing or unsupported.
    """
    method = getattr(result, "_bound_values", None)
    if method is None:
        raise TypeError(f"{type(result).__name__} has no property-definition bindings")
    return method(**selection)


def records(
    result: object, *, product_of: Iterable[RunEdge | Mapping[str, Any]] = (), **selection: Any
) -> tuple[DataRecord | DerivedDataRecord, ...]:
    r"""Build a checked data record for every bound value of an analysis result.

    Values with a ``derivation`` become derived data records of their base property.

    :param result: An analysis result of a bound type.
    :param product_of: Provenance edges attached to every record.
    :param \*\*selection: Result-specific selection keywords, such as ``lag_index`` for transport results.
    :return: Records in binding order.
    :raises TypeError: If the result type has no bindings or a selection keyword is missing or unsupported.
    :raises ValueError: If a value does not satisfy its property definition.
    """
    edges = tuple(product_of)
    out: list[DataRecord | DerivedDataRecord] = []
    for bound in bound_values(result, **selection):
        binding = bound.binding
        definition = load_property_definition(binding.definition)
        definition.check(bound.value)
        if binding.derivation is None:
            out.append(DataRecord.from_value(definition.definition_id, definition.name, bound.value, product_of=edges))
        else:
            out.append(
                DerivedDataRecord.from_value(
                    definition.definition_id, binding.derivation, definition.name, bound.value, product_of=edges
                )
            )
    return tuple(out)
