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
    "ACTIVATION_ENERGY",
    "ADSORPTION_ENERGY",
    "ARRHENIUS_PREFACTOR",
    "BAND_GAP",
    "BIAS",
    "BULK_MODULUS",
    "BULK_MODULUS_HILL",
    "BULK_MODULUS_PRESSURE_DERIVATIVE",
    "BULK_MODULUS_REUSS",
    "BULK_MODULUS_VOIGT",
    "CHARGED_DEFECT_FORMATION_ENERGY",
    "CHARGE_TRANSITION_LEVEL",
    "CHEMICAL_POTENTIAL_REGION",
    "COMPLIANCE_TENSOR",
    "CONVEX_HULL_PHASE_DIAGRAM",
    "DIFFUSION_COEFFICIENT",
    "DIFFUSION_PREFACTOR",
    "DIFFUSION_RUNNING_INTEGRAL",
    "DIFFUSION_TENSOR",
    "DIRECT_BAND_GAP",
    "DISTINCT_VAN_HOVE_FUNCTION",
    "ELASTIC_TENSOR",
    "ELECTRONIC_DENSITY_OF_STATES",
    "ENERGY_ABOVE_HULL_PER_ATOM",
    "ENERGY_PREDICTION_ERRORS",
    "EQUILIBRIUM_ENERGY",
    "EQUILIBRIUM_VOLUME",
    "FERMI_ENERGY",
    "FORCE_PREDICTION_ERRORS",
    "FORMATION_ENERGY_PER_ATOM",
    "HEAT_CAPACITY_CONSTANT_PRESSURE",
    "HEAT_CAPACITY_CONSTANT_VOLUME",
    "HELMHOLTZ_FREE_ENERGY",
    "HIGH_FREQUENCY_RELATIVE_PERMITTIVITY",
    "INTERMEDIATE_SCATTERING_FUNCTION",
    "ISOTHERMAL_COMPRESSIBILITY",
    "MAE",
    "MAXIMUM_ABSOLUTE_ERROR",
    "MEAN",
    "MEAN_SQUARED_DISPLACEMENT",
    "MIGRATION_BARRIER_FORWARD",
    "MIGRATION_BARRIER_REVERSE",
    "MODE_GRUNEISEN_PARAMETERS",
    "NVE_ENERGY_DRIFT",
    "QUASIHARMONIC_THERMODYNAMICS",
    "RADIAL_DISTRIBUTION_FUNCTION",
    "REACTION_ENERGY",
    "RELATIVE_EFFECTIVE_MASS",
    "RMSE",
    "SEGREGATION_ENERGY",
    "SELF_INTERMEDIATE_SCATTERING_FUNCTION",
    "SELF_VAN_HOVE_FUNCTION",
    "SHEAR_MODULUS_HILL",
    "SHEAR_MODULUS_REUSS",
    "SHEAR_MODULUS_VOIGT",
    "SHEAR_VISCOSITY",
    "SHEAR_VISCOSITY_RUNNING_INTEGRAL",
    "SPIN_CHANNEL_ELECTRONIC_DENSITY_OF_STATES",
    "STANDARD_DEVIATION",
    "STANDARD_ERROR",
    "STATIC_RELATIVE_PERMITTIVITY",
    "STEINHARDT_BOND_ORDER",
    "STRESS_PREDICTION_ERRORS",
    "SURFACE_ENERGY",
    "THERMAL_CONDUCTIVITY",
    "THERMAL_CONDUCTIVITY_RUNNING_INTEGRAL",
    "THERMAL_CONDUCTIVITY_TENSOR",
    "TOTAL_ENERGY_PER_ATOM",
    "TOTAL_MAGNETIC_MOMENT",
    "UNIVERSAL_ANISOTROPY_INDEX",
    "VELOCITY_AUTOCORRELATION",
    "VELOCITY_POWER_SPECTRUM",
    "VIBRATIONAL_ENTROPY",
    "VIBRATIONAL_HEAT_CAPACITY",
    "VIBRATIONAL_INTERNAL_ENERGY",
    "VIBRATIONAL_THERMODYNAMICS",
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

#: Property definition ``vibrational_thermodynamics``.
VIBRATIONAL_THERMODYNAMICS = "https://schemas.httk.org/defs/v0.1/properties/thermodynamics/vibrational_thermodynamics"

#: Property definition ``quasiharmonic_thermodynamics``.
QUASIHARMONIC_THERMODYNAMICS = (
    "https://schemas.httk.org/defs/v0.1/properties/thermodynamics/quasiharmonic_thermodynamics"
)

#: Property definition ``mean_squared_displacement``.
MEAN_SQUARED_DISPLACEMENT = "https://schemas.httk.org/defs/v0.1/properties/transport/mean_squared_displacement"

#: Property definition ``velocity_autocorrelation``.
VELOCITY_AUTOCORRELATION = "https://schemas.httk.org/defs/v0.1/properties/transport/velocity_autocorrelation"

#: Property definition ``diffusion_running_integral``.
DIFFUSION_RUNNING_INTEGRAL = "https://schemas.httk.org/defs/v0.1/properties/transport/diffusion_running_integral"

#: Property definition ``thermal_conductivity_running_integral``.
THERMAL_CONDUCTIVITY_RUNNING_INTEGRAL = (
    "https://schemas.httk.org/defs/v0.1/properties/transport/thermal_conductivity_running_integral"
)

#: Property definition ``shear_viscosity_running_integral``.
SHEAR_VISCOSITY_RUNNING_INTEGRAL = (
    "https://schemas.httk.org/defs/v0.1/properties/transport/shear_viscosity_running_integral"
)

#: Property definition ``radial_distribution_function``.
RADIAL_DISTRIBUTION_FUNCTION = "https://schemas.httk.org/defs/v0.1/properties/structure/radial_distribution_function"

#: Property definition ``electronic_density_of_states``.
ELECTRONIC_DENSITY_OF_STATES = "https://schemas.httk.org/defs/v0.1/properties/electronic/electronic_density_of_states"

#: Property definition ``relative_effective_mass``.
RELATIVE_EFFECTIVE_MASS = "https://schemas.httk.org/defs/v0.1/properties/electronic/relative_effective_mass"

#: Property definition ``fermi_energy``.
FERMI_ENERGY = "https://schemas.httk.org/defs/v0.1/properties/electronic/fermi_energy"

#: Property definition ``static_relative_permittivity``.
STATIC_RELATIVE_PERMITTIVITY = "https://schemas.httk.org/defs/v0.1/properties/electronic/static_relative_permittivity"

#: Property definition ``high_frequency_relative_permittivity``.
HIGH_FREQUENCY_RELATIVE_PERMITTIVITY = (
    "https://schemas.httk.org/defs/v0.1/properties/electronic/high_frequency_relative_permittivity"
)

#: Property definition ``total_magnetic_moment``.
TOTAL_MAGNETIC_MOMENT = "https://schemas.httk.org/defs/v0.1/properties/magnetism/total_magnetic_moment"

#: Property definition ``charged_defect_formation_energy``.
CHARGED_DEFECT_FORMATION_ENERGY = (
    "https://schemas.httk.org/defs/v0.1/properties/defects/charged_defect_formation_energy"
)

#: Property definition ``charge_transition_level``.
CHARGE_TRANSITION_LEVEL = "https://schemas.httk.org/defs/v0.1/properties/defects/charge_transition_level"

#: Property definition ``surface_energy``.
SURFACE_ENERGY = "https://schemas.httk.org/defs/v0.1/properties/defects/surface_energy"

#: Property definition ``adsorption_energy``.
ADSORPTION_ENERGY = "https://schemas.httk.org/defs/v0.1/properties/defects/adsorption_energy"

#: Property definition ``segregation_energy``.
SEGREGATION_ENERGY = "https://schemas.httk.org/defs/v0.1/properties/defects/segregation_energy"

#: Property definition ``migration_barrier_forward``.
MIGRATION_BARRIER_FORWARD = "https://schemas.httk.org/defs/v0.1/properties/kinetics/migration_barrier_forward"

#: Property definition ``migration_barrier_reverse``.
MIGRATION_BARRIER_REVERSE = "https://schemas.httk.org/defs/v0.1/properties/kinetics/migration_barrier_reverse"

#: Property definition ``activation_energy``.
ACTIVATION_ENERGY = "https://schemas.httk.org/defs/v0.1/properties/kinetics/activation_energy"

#: Property definition ``arrhenius_prefactor``.
ARRHENIUS_PREFACTOR = "https://schemas.httk.org/defs/v0.1/properties/kinetics/arrhenius_prefactor"

#: Property definition ``diffusion_prefactor``.
DIFFUSION_PREFACTOR = "https://schemas.httk.org/defs/v0.1/properties/transport/diffusion_prefactor"

#: Property definition ``self_intermediate_scattering_function``.
SELF_INTERMEDIATE_SCATTERING_FUNCTION = (
    "https://schemas.httk.org/defs/v0.1/properties/dynamics/self_intermediate_scattering_function"
)

#: Property definition ``intermediate_scattering_function``.
INTERMEDIATE_SCATTERING_FUNCTION = (
    "https://schemas.httk.org/defs/v0.1/properties/dynamics/intermediate_scattering_function"
)

#: Property definition ``self_van_hove_function``.
SELF_VAN_HOVE_FUNCTION = "https://schemas.httk.org/defs/v0.1/properties/dynamics/self_van_hove_function"

#: Property definition ``distinct_van_hove_function``.
DISTINCT_VAN_HOVE_FUNCTION = "https://schemas.httk.org/defs/v0.1/properties/dynamics/distinct_van_hove_function"

#: Property definition ``velocity_power_spectrum``.
VELOCITY_POWER_SPECTRUM = "https://schemas.httk.org/defs/v0.1/properties/dynamics/velocity_power_spectrum"

#: Property definition ``steinhardt_bond_order``.
STEINHARDT_BOND_ORDER = "https://schemas.httk.org/defs/v0.1/properties/structure/steinhardt_bond_order"

#: Property definition ``total_energy_per_atom``.
TOTAL_ENERGY_PER_ATOM = "https://schemas.httk.org/defs/v0.1/properties/energetics/total_energy_per_atom"

#: Property definition ``chemical_potential_region``.
CHEMICAL_POTENTIAL_REGION = "https://schemas.httk.org/defs/v0.1/properties/energetics/chemical_potential_region"

#: Property definition ``convex_hull_phase_diagram``.
CONVEX_HULL_PHASE_DIAGRAM = "https://schemas.httk.org/defs/v0.1/properties/energetics/convex_hull_phase_diagram"

#: Property definition ``energy_prediction_errors``.
ENERGY_PREDICTION_ERRORS = "https://schemas.httk.org/defs/v0.1/properties/validation/energy_prediction_errors"

#: Property definition ``force_prediction_errors``.
FORCE_PREDICTION_ERRORS = "https://schemas.httk.org/defs/v0.1/properties/validation/force_prediction_errors"

#: Property definition ``stress_prediction_errors``.
STRESS_PREDICTION_ERRORS = "https://schemas.httk.org/defs/v0.1/properties/validation/stress_prediction_errors"

#: Property definition ``nve_energy_drift``.
NVE_ENERGY_DRIFT = "https://schemas.httk.org/defs/v0.1/properties/validation/nve_energy_drift"

#: Property definition ``mode_gruneisen_parameters``.
MODE_GRUNEISEN_PARAMETERS = "https://schemas.httk.org/defs/v0.1/properties/thermodynamics/mode_gruneisen_parameters"

#: Property definition ``spin_channel_electronic_density_of_states``.
SPIN_CHANNEL_ELECTRONIC_DENSITY_OF_STATES = (
    "https://schemas.httk.org/defs/v0.1/properties/electronic/spin_channel_electronic_density_of_states"
)


@dataclass(frozen=True, slots=True)
class FieldBinding:
    """The property identity of one analysis-result field.

    The field value is in the unit of ``definition``. With a ``derivation`` the
    value is that statistic of the base property (same unit). A series is one
    value of a dictionary-typed definition whose members are equal-length lists.

    :param definition: Property-definition IRI.
    :param derivation: Derivation-term IRI qualifying the property, or ``None``.
    """

    definition: str
    derivation: str | None = None


@dataclass(frozen=True, slots=True)
class BoundValue:
    """One result value with its property binding, as plain JSON-compatible data.

    :param field: Result field name, such as ``bulk_modulus`` or ``integrals[10]``, or the
        definition name of a series.
    :param binding: Property binding of the value.
    :param value: Float, nested lists of floats, or a dictionary of member lists (a series), in the
        units of ``binding.definition``.
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


def _series(definition: str, **members: Any) -> BoundValue:
    """Bind plain-list ``members`` as one dictionary value of ``definition``, named by the definition."""
    return BoundValue(definition.rsplit("/", 1)[1], FieldBinding(definition), members)
