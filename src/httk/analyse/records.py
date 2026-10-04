"""Turn analysis results into property-definition data records.

Bound result types know which of their fields carry which property definition
(see :class:`httk.analyse.definitions.FieldBinding`). :func:`records` emits one
:class:`httk.core.DataRecord` per scalar or fixed-size bound value, validated
against its definition. Series over a coordinate (``axis``) and statistics
(``derivation``) are not recorded here; they belong in the analysis summary
envelope. Selection keywords are explicit per result type: transport results
require ``lag_index``, the caller's plateau choice.
"""

from collections.abc import Iterable, Mapping
from typing import Any

from httk.core import DataRecord, RunEdge, load_property_definition

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
) -> tuple[DataRecord, ...]:
    r"""Build data records for the scalar and fixed-size bound values of an analysis result.

    Values bound with an ``axis`` (series) or a ``derivation`` (statistics) are
    skipped; record them in the analysis summary envelope instead.

    :param result: An analysis result of a bound type.
    :param product_of: Provenance edges attached to every record.
    :param \*\*selection: Result-specific selection keywords, such as ``lag_index`` for transport results.
    :return: Records in binding order.
    :raises TypeError: If the result type has no bindings or a selection keyword is missing or unsupported.
    :raises ValueError: If a value does not satisfy its property definition.
    """
    edges = tuple(product_of)
    out = []
    for bound in bound_values(result, **selection):
        if bound.binding.axis is not None or bound.binding.derivation is not None:
            continue
        definition = load_property_definition(bound.binding.definition)
        definition.check(bound.value)
        out.append(DataRecord.from_value(definition.definition_id, definition.name, bound.value, product_of=edges))
    return tuple(out)
