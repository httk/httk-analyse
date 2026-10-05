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

register_property_definition(
    definition_id="https://schemas.httk.org/defs/v0.1/properties/thermodynamics/vibrational_thermodynamics",
    resource="httk.registry.schemas.analyse:vibrational_thermodynamics.json",
)

register_property_definition(
    definition_id="https://schemas.httk.org/defs/v0.1/properties/thermodynamics/quasiharmonic_thermodynamics",
    resource="httk.registry.schemas.analyse:quasiharmonic_thermodynamics.json",
)

register_property_definition(
    definition_id="https://schemas.httk.org/defs/v0.1/properties/transport/mean_squared_displacement",
    resource="httk.registry.schemas.analyse:mean_squared_displacement.json",
)

register_property_definition(
    definition_id="https://schemas.httk.org/defs/v0.1/properties/transport/velocity_autocorrelation",
    resource="httk.registry.schemas.analyse:velocity_autocorrelation.json",
)

register_property_definition(
    definition_id="https://schemas.httk.org/defs/v0.1/properties/transport/diffusion_running_integral",
    resource="httk.registry.schemas.analyse:diffusion_running_integral.json",
)

register_property_definition(
    definition_id="https://schemas.httk.org/defs/v0.1/properties/transport/thermal_conductivity_running_integral",
    resource="httk.registry.schemas.analyse:thermal_conductivity_running_integral.json",
)

register_property_definition(
    definition_id="https://schemas.httk.org/defs/v0.1/properties/transport/shear_viscosity_running_integral",
    resource="httk.registry.schemas.analyse:shear_viscosity_running_integral.json",
)

register_property_definition(
    definition_id="https://schemas.httk.org/defs/v0.1/properties/structure/radial_distribution_function",
    resource="httk.registry.schemas.analyse:radial_distribution_function.json",
)

register_property_definition(
    definition_id="https://schemas.httk.org/defs/v0.1/properties/electronic/electronic_density_of_states",
    resource="httk.registry.schemas.analyse:electronic_density_of_states.json",
)

register_property_definition(
    definition_id="https://schemas.httk.org/defs/v0.1/properties/electronic/relative_effective_mass",
    resource="httk.registry.schemas.analyse:relative_effective_mass.json",
)

register_property_definition(
    definition_id="https://schemas.httk.org/defs/v0.1/properties/electronic/fermi_energy",
    resource="httk.registry.schemas.analyse:fermi_energy.json",
)

register_property_definition(
    definition_id="https://schemas.httk.org/defs/v0.1/properties/electronic/static_relative_permittivity",
    resource="httk.registry.schemas.analyse:static_relative_permittivity.json",
)

register_property_definition(
    definition_id="https://schemas.httk.org/defs/v0.1/properties/electronic/high_frequency_relative_permittivity",
    resource="httk.registry.schemas.analyse:high_frequency_relative_permittivity.json",
)

register_property_definition(
    definition_id="https://schemas.httk.org/defs/v0.1/properties/magnetism/total_magnetic_moment",
    resource="httk.registry.schemas.analyse:total_magnetic_moment.json",
)

register_property_definition(
    definition_id="https://schemas.httk.org/defs/v0.1/properties/defects/charged_defect_formation_energy",
    resource="httk.registry.schemas.analyse:charged_defect_formation_energy.json",
)

register_property_definition(
    definition_id="https://schemas.httk.org/defs/v0.1/properties/defects/charge_transition_level",
    resource="httk.registry.schemas.analyse:charge_transition_level.json",
)

register_property_definition(
    definition_id="https://schemas.httk.org/defs/v0.1/properties/defects/surface_energy",
    resource="httk.registry.schemas.analyse:surface_energy.json",
)

register_property_definition(
    definition_id="https://schemas.httk.org/defs/v0.1/properties/defects/adsorption_energy",
    resource="httk.registry.schemas.analyse:adsorption_energy.json",
)

register_property_definition(
    definition_id="https://schemas.httk.org/defs/v0.1/properties/defects/segregation_energy",
    resource="httk.registry.schemas.analyse:segregation_energy.json",
)

register_property_definition(
    definition_id="https://schemas.httk.org/defs/v0.1/properties/kinetics/migration_barrier_forward",
    resource="httk.registry.schemas.analyse:migration_barrier_forward.json",
)

register_property_definition(
    definition_id="https://schemas.httk.org/defs/v0.1/properties/kinetics/migration_barrier_reverse",
    resource="httk.registry.schemas.analyse:migration_barrier_reverse.json",
)

register_property_definition(
    definition_id="https://schemas.httk.org/defs/v0.1/properties/kinetics/activation_energy",
    resource="httk.registry.schemas.analyse:activation_energy.json",
)

register_property_definition(
    definition_id="https://schemas.httk.org/defs/v0.1/properties/energetics/total_energy_per_atom",
    resource="httk.registry.schemas.analyse:total_energy_per_atom.json",
)

register_property_definition(
    definition_id="https://schemas.httk.org/defs/v0.1/properties/kinetics/arrhenius_prefactor",
    resource="httk.registry.schemas.analyse:arrhenius_prefactor.json",
)

register_property_definition(
    definition_id="https://schemas.httk.org/defs/v0.1/properties/transport/diffusion_prefactor",
    resource="httk.registry.schemas.analyse:diffusion_prefactor.json",
)

register_property_definition(
    definition_id="https://schemas.httk.org/defs/v0.1/properties/dynamics/self_intermediate_scattering_function",
    resource="httk.registry.schemas.analyse:self_intermediate_scattering_function.json",
)

register_property_definition(
    definition_id="https://schemas.httk.org/defs/v0.1/properties/dynamics/intermediate_scattering_function",
    resource="httk.registry.schemas.analyse:intermediate_scattering_function.json",
)

register_property_definition(
    definition_id="https://schemas.httk.org/defs/v0.1/properties/dynamics/self_van_hove_function",
    resource="httk.registry.schemas.analyse:self_van_hove_function.json",
)

register_property_definition(
    definition_id="https://schemas.httk.org/defs/v0.1/properties/dynamics/distinct_van_hove_function",
    resource="httk.registry.schemas.analyse:distinct_van_hove_function.json",
)

register_property_definition(
    definition_id="https://schemas.httk.org/defs/v0.1/properties/dynamics/velocity_power_spectrum",
    resource="httk.registry.schemas.analyse:velocity_power_spectrum.json",
)

register_property_definition(
    definition_id="https://schemas.httk.org/defs/v0.1/properties/structure/steinhardt_bond_order",
    resource="httk.registry.schemas.analyse:steinhardt_bond_order.json",
)

register_property_definition(
    definition_id="https://schemas.httk.org/defs/v0.1/properties/validation/energy_prediction_errors",
    resource="httk.registry.schemas.analyse:energy_prediction_errors.json",
)

register_property_definition(
    definition_id="https://schemas.httk.org/defs/v0.1/properties/validation/force_prediction_errors",
    resource="httk.registry.schemas.analyse:force_prediction_errors.json",
)

register_property_definition(
    definition_id="https://schemas.httk.org/defs/v0.1/properties/validation/stress_prediction_errors",
    resource="httk.registry.schemas.analyse:stress_prediction_errors.json",
)

register_property_definition(
    definition_id="https://schemas.httk.org/defs/v0.1/properties/validation/nve_energy_drift",
    resource="httk.registry.schemas.analyse:nve_energy_drift.json",
)

register_property_definition(
    definition_id="https://schemas.httk.org/defs/v0.1/properties/thermodynamics/mode_gruneisen_parameters",
    resource="httk.registry.schemas.analyse:mode_gruneisen_parameters.json",
)

register_property_definition(
    definition_id="https://schemas.httk.org/defs/v0.1/properties/electronic/spin_channel_electronic_density_of_states",
    resource="httk.registry.schemas.analyse:spin_channel_electronic_density_of_states.json",
)

register_property_definition(
    definition_id="https://schemas.httk.org/defs/v0.1/properties/energetics/chemical_potential_region",
    resource="httk.registry.schemas.analyse:chemical_potential_region.json",
)

register_property_definition(
    definition_id="https://schemas.httk.org/defs/v0.1/properties/energetics/convex_hull_phase_diagram",
    resource="httk.registry.schemas.analyse:convex_hull_phase_diagram.json",
)
