"""Identities binding analysis results to published httk property definitions.

Each constant is the IRI of a property definition vendored under
``httk.registry.schemas.analyse`` (units, shapes and conventions come from the
definition, not from this package). The derivation terms qualify a base
property so that statistics of it (mean, standard error, ...) are identified
without defining a new property. Core-level IRIs such as pressure or
temperature live in ``httk.core.definition_ids``. :class:`FieldBinding` and
:class:`BoundValue` describe how fields of analysis results bind to these
identities; :mod:`httk.analyse.records` turns them into data records.
"""

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

__all__ = [
    "BAND_GAP",
    "BIAS",
    "BULK_MODULUS",
    "BULK_MODULUS_HILL",
    "BULK_MODULUS_PRESSURE_DERIVATIVE",
    "BULK_MODULUS_REUSS",
    "BULK_MODULUS_VOIGT",
    "COMPLIANCE_TENSOR",
    "DIFFUSION_COEFFICIENT",
    "DIFFUSION_TENSOR",
    "DIRECT_BAND_GAP",
    "ELASTIC_TENSOR",
    "ENERGY_ABOVE_HULL_PER_ATOM",
    "EQUILIBRIUM_ENERGY",
    "EQUILIBRIUM_VOLUME",
    "FORMATION_ENERGY_PER_ATOM",
    "HEAT_CAPACITY_CONSTANT_PRESSURE",
    "HEAT_CAPACITY_CONSTANT_VOLUME",
    "HELMHOLTZ_FREE_ENERGY",
    "ISOTHERMAL_COMPRESSIBILITY",
    "MAE",
    "MAXIMUM_ABSOLUTE_ERROR",
    "MEAN",
    "REACTION_ENERGY",
    "RMSE",
    "SHEAR_MODULUS_HILL",
    "SHEAR_MODULUS_REUSS",
    "SHEAR_MODULUS_VOIGT",
    "SHEAR_VISCOSITY",
    "STANDARD_DEVIATION",
    "STANDARD_ERROR",
    "THERMAL_CONDUCTIVITY",
    "THERMAL_CONDUCTIVITY_TENSOR",
    "UNIVERSAL_ANISOTROPY_INDEX",
    "VIBRATIONAL_ENTROPY",
    "VIBRATIONAL_HEAT_CAPACITY",
    "VIBRATIONAL_INTERNAL_ENERGY",
    "VOLUMETRIC_THERMAL_EXPANSION",
    "ZERO_POINT_ENERGY",
    "BoundValue",
    "FieldBinding",
]

#: Property definition ``bulk_modulus``.
BULK_MODULUS = "https://schemas.httk.org/defs/v0.1/properties/mechanics/bulk_modulus"

#: Property definition ``bulk_modulus_hill``.
BULK_MODULUS_HILL = "https://schemas.httk.org/defs/v0.1/properties/mechanics/bulk_modulus_hill"

#: Property definition ``bulk_modulus_pressure_derivative``.
BULK_MODULUS_PRESSURE_DERIVATIVE = (
    "https://schemas.httk.org/defs/v0.1/properties/mechanics/bulk_modulus_pressure_derivative"
)

#: Property definition ``bulk_modulus_reuss``.
BULK_MODULUS_REUSS = "https://schemas.httk.org/defs/v0.1/properties/mechanics/bulk_modulus_reuss"

#: Property definition ``bulk_modulus_voigt``.
BULK_MODULUS_VOIGT = "https://schemas.httk.org/defs/v0.1/properties/mechanics/bulk_modulus_voigt"

#: Property definition ``compliance_tensor``.
COMPLIANCE_TENSOR = "https://schemas.httk.org/defs/v0.1/properties/mechanics/compliance_tensor"

#: Property definition ``elastic_tensor``.
ELASTIC_TENSOR = "https://schemas.httk.org/defs/v0.1/properties/mechanics/elastic_tensor"

#: Property definition ``equilibrium_energy``.
EQUILIBRIUM_ENERGY = "https://schemas.httk.org/defs/v0.1/properties/mechanics/equilibrium_energy"

#: Property definition ``equilibrium_volume``.
EQUILIBRIUM_VOLUME = "https://schemas.httk.org/defs/v0.1/properties/mechanics/equilibrium_volume"

#: Property definition ``shear_modulus_hill``.
SHEAR_MODULUS_HILL = "https://schemas.httk.org/defs/v0.1/properties/mechanics/shear_modulus_hill"

#: Property definition ``shear_modulus_reuss``.
SHEAR_MODULUS_REUSS = "https://schemas.httk.org/defs/v0.1/properties/mechanics/shear_modulus_reuss"

#: Property definition ``shear_modulus_voigt``.
SHEAR_MODULUS_VOIGT = "https://schemas.httk.org/defs/v0.1/properties/mechanics/shear_modulus_voigt"

#: Property definition ``universal_anisotropy_index``.
UNIVERSAL_ANISOTROPY_INDEX = "https://schemas.httk.org/defs/v0.1/properties/mechanics/universal_anisotropy_index"

#: Property definition ``energy_above_hull_per_atom``.
ENERGY_ABOVE_HULL_PER_ATOM = "https://schemas.httk.org/defs/v0.1/properties/energetics/energy_above_hull_per_atom"

#: Property definition ``formation_energy_per_atom``.
FORMATION_ENERGY_PER_ATOM = "https://schemas.httk.org/defs/v0.1/properties/energetics/formation_energy_per_atom"

#: Property definition ``reaction_energy``.
REACTION_ENERGY = "https://schemas.httk.org/defs/v0.1/properties/energetics/reaction_energy"

#: Property definition ``heat_capacity_constant_pressure``.
HEAT_CAPACITY_CONSTANT_PRESSURE = (
    "https://schemas.httk.org/defs/v0.1/properties/thermodynamics/heat_capacity_constant_pressure"
)

#: Property definition ``heat_capacity_constant_volume``.
HEAT_CAPACITY_CONSTANT_VOLUME = (
    "https://schemas.httk.org/defs/v0.1/properties/thermodynamics/heat_capacity_constant_volume"
)

#: Property definition ``helmholtz_free_energy``.
HELMHOLTZ_FREE_ENERGY = "https://schemas.httk.org/defs/v0.1/properties/thermodynamics/helmholtz_free_energy"

#: Property definition ``isothermal_compressibility``.
ISOTHERMAL_COMPRESSIBILITY = "https://schemas.httk.org/defs/v0.1/properties/thermodynamics/isothermal_compressibility"

#: Property definition ``vibrational_entropy``.
VIBRATIONAL_ENTROPY = "https://schemas.httk.org/defs/v0.1/properties/thermodynamics/vibrational_entropy"

#: Property definition ``vibrational_heat_capacity``.
VIBRATIONAL_HEAT_CAPACITY = "https://schemas.httk.org/defs/v0.1/properties/thermodynamics/vibrational_heat_capacity"

#: Property definition ``vibrational_internal_energy``.
VIBRATIONAL_INTERNAL_ENERGY = "https://schemas.httk.org/defs/v0.1/properties/thermodynamics/vibrational_internal_energy"

#: Property definition ``volumetric_thermal_expansion``.
VOLUMETRIC_THERMAL_EXPANSION = (
    "https://schemas.httk.org/defs/v0.1/properties/thermodynamics/volumetric_thermal_expansion"
)

#: Property definition ``zero_point_energy``.
ZERO_POINT_ENERGY = "https://schemas.httk.org/defs/v0.1/properties/thermodynamics/zero_point_energy"

#: Property definition ``diffusion_coefficient``.
DIFFUSION_COEFFICIENT = "https://schemas.httk.org/defs/v0.1/properties/transport/diffusion_coefficient"

#: Property definition ``diffusion_tensor``.
DIFFUSION_TENSOR = "https://schemas.httk.org/defs/v0.1/properties/transport/diffusion_tensor"

#: Property definition ``shear_viscosity``.
SHEAR_VISCOSITY = "https://schemas.httk.org/defs/v0.1/properties/transport/shear_viscosity"

#: Property definition ``thermal_conductivity``.
THERMAL_CONDUCTIVITY = "https://schemas.httk.org/defs/v0.1/properties/transport/thermal_conductivity"

#: Property definition ``thermal_conductivity_tensor``.
THERMAL_CONDUCTIVITY_TENSOR = "https://schemas.httk.org/defs/v0.1/properties/transport/thermal_conductivity_tensor"

#: Property definition ``band_gap``.
BAND_GAP = "https://schemas.httk.org/defs/v0.1/properties/electronic/band_gap"

#: Property definition ``direct_band_gap``.
DIRECT_BAND_GAP = "https://schemas.httk.org/defs/v0.1/properties/electronic/direct_band_gap"

#: Derivation term ``mean`` qualifying a base property for statistics.
MEAN = "https://schemas.httk.org/defs/v0.1/derivations/mean"

#: Derivation term ``standard_error`` qualifying a base property for statistics.
STANDARD_ERROR = "https://schemas.httk.org/defs/v0.1/derivations/standard_error"

#: Derivation term ``standard_deviation`` qualifying a base property for statistics.
STANDARD_DEVIATION = "https://schemas.httk.org/defs/v0.1/derivations/standard_deviation"

#: Derivation term ``rmse`` qualifying a base property for statistics.
RMSE = "https://schemas.httk.org/defs/v0.1/derivations/rmse"

#: Derivation term ``mae`` qualifying a base property for statistics.
MAE = "https://schemas.httk.org/defs/v0.1/derivations/mae"

#: Derivation term ``bias`` qualifying a base property for statistics.
BIAS = "https://schemas.httk.org/defs/v0.1/derivations/bias"

#: Derivation term ``maximum_absolute_error`` qualifying a base property for statistics.
MAXIMUM_ABSOLUTE_ERROR = "https://schemas.httk.org/defs/v0.1/derivations/maximum_absolute_error"


@dataclass(frozen=True, slots=True)
class FieldBinding:
    """The property identity of one analysis-result field.

    The field value is in the unit of ``definition``. With a ``derivation`` the
    value is that statistic of the base property (same unit). With an ``axis``
    the value is a series of the base property sampled over the coordinate
    field of that name in the same result.

    :param definition: Property-definition IRI.
    :param derivation: Derivation-term IRI qualifying the property, or ``None``.
    :param axis: Name of the coordinate field of a series value, or ``None``.
    """

    definition: str
    derivation: str | None = None
    axis: str | None = None


@dataclass(frozen=True, slots=True)
class BoundValue:
    """One result value with its property binding, as plain JSON-compatible data.

    :param field: Result field name, such as ``bulk_modulus`` or ``integrals[10]``.
    :param binding: Property binding of the value.
    :param value: Float or nested lists of floats in the unit of ``binding.definition``.
    """

    field: str
    binding: FieldBinding
    value: Any


#: Analysis property IRIs by definition name.
_PROPERTY_BY_NAME = {
    iri.rsplit("/", 1)[1]: iri for iri in (globals()[name] for name in __all__) if "/properties/" in str(iri)
}


def _plain(value: Any) -> Any:
    """Return ``value`` with nested tuples as lists and numbers as floats."""
    if isinstance(value, tuple | list):
        return [_plain(item) for item in value]
    return float(value)


def _reject_selection(result: object, selection: Mapping[str, Any]) -> None:
    """Raise when record selection keywords are given to a result that takes none."""
    if selection:
        raise TypeError(f"{type(result).__name__} takes no selection keywords, got {', '.join(sorted(selection))}")


def _bind_fields(
    result: object, selection: Mapping[str, Any], bindings: Mapping[str, FieldBinding]
) -> tuple[BoundValue, ...]:
    """Bind attributes of ``result`` named by ``bindings``, which takes no selection keywords."""
    _reject_selection(result, selection)
    return tuple(BoundValue(field, binding, _plain(getattr(result, field))) for field, binding in bindings.items())
