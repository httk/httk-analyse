"""Turn analysis results into property-definition data records.

Bound result types know which of their fields carry which property definition
(see :class:`httk.analyse.definitions.FieldBinding`). :func:`records` emits one
checked record per bound value. A property value becomes a typed record of its
definition: an :mod:`httk.analyse.property_records` kind for an analyse
definition, an :mod:`httk.core.property_records` kind for a core definition such
as temperature or volume, and :class:`httk.core.TotalEnergyRecord` or
:class:`httk.core.AverageTotalEnergyRecord` for the two energies. Typed records
store their values in typed columns, so a store serves them and filters on them,
including dictionary members by nested name: ``_httk_bulk_modulus > 100`` or
``_httk_energy_prediction_errors.rmse < 0.005``. A statistic (``derivation``) of
a fixed base property becomes its generated statistic kind, served as
``_httk_<base>_<derivation>`` (for example ``_httk_total_energy_per_atom_rmse``)
under an ad-hoc definition synthesized from the base. A statistic of a
caller-chosen base (``PropertyParity``) becomes a :class:`httk.core.DerivedDataRecord`,
and a value with no typed kind (or a null value) a generic :class:`httk.core.DataRecord`;
these generic records are a catch-all that keep the value as JSON, which a store does
not serve.
Selection keywords are explicit per result type: ``lag_index`` selects the
plateau of transport results (optional for a single run, required for replica
statistics).
"""

import logging
from collections.abc import Iterable, Mapping
from typing import Any

from httk.core import (
    AverageTotalEnergyRecord,
    DataRecord,
    DerivedDataRecord,
    RunEdge,
    TotalEnergyRecord,
    TypedRecord,
    load_property_definition,
)
from httk.core import property_records as core_records
from httk.core.definition_ids import AVERAGE_TOTAL_ENERGY, TOTAL_ENERGY

from . import property_records
from .definitions import BoundValue

__all__ = ["AnalysisRecord", "bound_values", "records"]

logger = logging.getLogger(__name__)

#: A record :func:`records` emits for one bound value.
type AnalysisRecord = TypedRecord | TotalEnergyRecord | AverageTotalEnergyRecord | DataRecord | DerivedDataRecord


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
    bound = method(**selection)
    if not bound:
        logger.warning(
            "%s binds no property values (no definition applies to this result, e.g. a parity result without a "
            "property definition)",
            type(result).__name__,
        )
    return bound


def _record(bound: BoundValue, edges: tuple[RunEdge, ...]) -> AnalysisRecord:
    """Return the checked record of one bound value.

    :param bound: The bound value.
    :param edges: The provenance edges.
    :return: Its typed, derived or generic record.
    """
    binding, value = bound.binding, bound.value
    if value is not None and binding.derivation is not None:
        kind = property_records.DERIVED_RECORD_KINDS.get((binding.definition, binding.derivation))
        if kind is not None:
            return kind.from_value(value, product_of=edges)
    elif value is not None:
        kind = property_records.RECORD_KINDS.get(binding.definition) or core_records.RECORD_KINDS.get(
            binding.definition
        )
        if kind is not None:
            return kind.from_value(value, product_of=edges)
        if binding.definition == TOTAL_ENERGY:
            return TotalEnergyRecord(float(value), product_of=edges)
        if binding.definition == AVERAGE_TOTAL_ENERGY:
            return AverageTotalEnergyRecord(float(value), product_of=edges)
    definition = load_property_definition(binding.definition)
    definition.check(value)
    if binding.derivation is None:
        return DataRecord.from_value(definition.definition_id, definition.name, value, product_of=edges)
    return DerivedDataRecord.from_value(
        definition.definition_id, binding.derivation, definition.name, value, product_of=edges
    )


def records(
    result: object, *, product_of: Iterable[RunEdge | Mapping[str, Any]] = (), **selection: Any
) -> tuple[AnalysisRecord, ...]:
    r"""Build a checked record for every bound value of an analysis result.

    A property value becomes the typed record of its definition, which a store serves and filters (for
    example ``_httk_bulk_modulus > 100``, or ``_httk_energy_prediction_errors.rmse < 0.005`` on a
    dictionary member). A value with a ``derivation`` becomes its statistic kind, served as
    ``_httk_<base>_<derivation>``, or a derived data record when its base is caller-chosen; a value
    without a typed kind, or a null value, becomes a generic data record, whose value is not served.

    :param result: An analysis result of a bound type.
    :param product_of: Provenance edges attached to every record.
    :param \*\*selection: Result-specific selection keywords, such as ``lag_index`` for transport results.
    :return: Records in binding order.
    :raises TypeError: If the result type has no bindings or a selection keyword is missing or unsupported.
    :raises ValueError: If a value does not satisfy its property definition.
    """
    edges = tuple(RunEdge.from_obj(edge) for edge in product_of)
    return tuple(_record(bound, edges) for bound in bound_values(result, **selection))
