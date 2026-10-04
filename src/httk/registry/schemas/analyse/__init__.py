"""Register httk-analyse's vendored analysis property schemas."""

from httk.core.register import register_property_definition

register_property_definition(
    definition_id="https://schemas.httk.org/defs/v0.1/properties/mechanics/bulk_modulus",
    resource="httk.registry.schemas.analyse:bulk_modulus.json",
)

register_property_definition(
    definition_id="https://schemas.httk.org/defs/v0.1/properties/mechanics/bulk_modulus_hill",
    resource="httk.registry.schemas.analyse:bulk_modulus_hill.json",
)

register_property_definition(
    definition_id="https://schemas.httk.org/defs/v0.1/properties/mechanics/bulk_modulus_pressure_derivative",
    resource="httk.registry.schemas.analyse:bulk_modulus_pressure_derivative.json",
)

register_property_definition(
    definition_id="https://schemas.httk.org/defs/v0.1/properties/mechanics/bulk_modulus_reuss",
    resource="httk.registry.schemas.analyse:bulk_modulus_reuss.json",
)

register_property_definition(
    definition_id="https://schemas.httk.org/defs/v0.1/properties/mechanics/bulk_modulus_voigt",
    resource="httk.registry.schemas.analyse:bulk_modulus_voigt.json",
)

register_property_definition(
    definition_id="https://schemas.httk.org/defs/v0.1/properties/mechanics/compliance_tensor",
    resource="httk.registry.schemas.analyse:compliance_tensor.json",
)

register_property_definition(
    definition_id="https://schemas.httk.org/defs/v0.1/properties/mechanics/elastic_tensor",
    resource="httk.registry.schemas.analyse:elastic_tensor.json",
)

register_property_definition(
    definition_id="https://schemas.httk.org/defs/v0.1/properties/mechanics/equilibrium_energy",
    resource="httk.registry.schemas.analyse:equilibrium_energy.json",
)

register_property_definition(
    definition_id="https://schemas.httk.org/defs/v0.1/properties/mechanics/equilibrium_volume",
    resource="httk.registry.schemas.analyse:equilibrium_volume.json",
)

register_property_definition(
    definition_id="https://schemas.httk.org/defs/v0.1/properties/mechanics/shear_modulus_hill",
    resource="httk.registry.schemas.analyse:shear_modulus_hill.json",
)

register_property_definition(
    definition_id="https://schemas.httk.org/defs/v0.1/properties/mechanics/shear_modulus_reuss",
    resource="httk.registry.schemas.analyse:shear_modulus_reuss.json",
)

register_property_definition(
    definition_id="https://schemas.httk.org/defs/v0.1/properties/mechanics/shear_modulus_voigt",
    resource="httk.registry.schemas.analyse:shear_modulus_voigt.json",
)

register_property_definition(
    definition_id="https://schemas.httk.org/defs/v0.1/properties/mechanics/universal_anisotropy_index",
    resource="httk.registry.schemas.analyse:universal_anisotropy_index.json",
)

register_property_definition(
    definition_id="https://schemas.httk.org/defs/v0.1/properties/energetics/energy_above_hull_per_atom",
    resource="httk.registry.schemas.analyse:energy_above_hull_per_atom.json",
)

register_property_definition(
    definition_id="https://schemas.httk.org/defs/v0.1/properties/energetics/formation_energy_per_atom",
    resource="httk.registry.schemas.analyse:formation_energy_per_atom.json",
)

register_property_definition(
    definition_id="https://schemas.httk.org/defs/v0.1/properties/energetics/reaction_energy",
    resource="httk.registry.schemas.analyse:reaction_energy.json",
)

register_property_definition(
    definition_id="https://schemas.httk.org/defs/v0.1/properties/thermodynamics/heat_capacity_constant_pressure",
    resource="httk.registry.schemas.analyse:heat_capacity_constant_pressure.json",
)

register_property_definition(
    definition_id="https://schemas.httk.org/defs/v0.1/properties/thermodynamics/heat_capacity_constant_volume",
    resource="httk.registry.schemas.analyse:heat_capacity_constant_volume.json",
)

register_property_definition(
    definition_id="https://schemas.httk.org/defs/v0.1/properties/thermodynamics/helmholtz_free_energy",
    resource="httk.registry.schemas.analyse:helmholtz_free_energy.json",
)

register_property_definition(
    definition_id="https://schemas.httk.org/defs/v0.1/properties/thermodynamics/isothermal_compressibility",
    resource="httk.registry.schemas.analyse:isothermal_compressibility.json",
)

register_property_definition(
    definition_id="https://schemas.httk.org/defs/v0.1/properties/thermodynamics/vibrational_entropy",
    resource="httk.registry.schemas.analyse:vibrational_entropy.json",
)

register_property_definition(
    definition_id="https://schemas.httk.org/defs/v0.1/properties/thermodynamics/vibrational_heat_capacity",
    resource="httk.registry.schemas.analyse:vibrational_heat_capacity.json",
)

register_property_definition(
    definition_id="https://schemas.httk.org/defs/v0.1/properties/thermodynamics/vibrational_internal_energy",
    resource="httk.registry.schemas.analyse:vibrational_internal_energy.json",
)

register_property_definition(
    definition_id="https://schemas.httk.org/defs/v0.1/properties/thermodynamics/volumetric_thermal_expansion",
    resource="httk.registry.schemas.analyse:volumetric_thermal_expansion.json",
)

register_property_definition(
    definition_id="https://schemas.httk.org/defs/v0.1/properties/thermodynamics/zero_point_energy",
    resource="httk.registry.schemas.analyse:zero_point_energy.json",
)

register_property_definition(
    definition_id="https://schemas.httk.org/defs/v0.1/properties/transport/diffusion_coefficient",
    resource="httk.registry.schemas.analyse:diffusion_coefficient.json",
)

register_property_definition(
    definition_id="https://schemas.httk.org/defs/v0.1/properties/transport/diffusion_tensor",
    resource="httk.registry.schemas.analyse:diffusion_tensor.json",
)

register_property_definition(
    definition_id="https://schemas.httk.org/defs/v0.1/properties/transport/shear_viscosity",
    resource="httk.registry.schemas.analyse:shear_viscosity.json",
)

register_property_definition(
    definition_id="https://schemas.httk.org/defs/v0.1/properties/transport/thermal_conductivity",
    resource="httk.registry.schemas.analyse:thermal_conductivity.json",
)

register_property_definition(
    definition_id="https://schemas.httk.org/defs/v0.1/properties/transport/thermal_conductivity_tensor",
    resource="httk.registry.schemas.analyse:thermal_conductivity_tensor.json",
)

register_property_definition(
    definition_id="https://schemas.httk.org/defs/v0.1/properties/electronic/band_gap",
    resource="httk.registry.schemas.analyse:band_gap.json",
)

register_property_definition(
    definition_id="https://schemas.httk.org/defs/v0.1/properties/electronic/direct_band_gap",
    resource="httk.registry.schemas.analyse:direct_band_gap.json",
)
