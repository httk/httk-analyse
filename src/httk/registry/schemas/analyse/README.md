# Vendored analysis schemas

This package contains the published httk property definitions that bind
*httk-analyse* results to units, shapes and conventions. They are registered by
`httk.registry.schemas.analyse` in httk-core's IRI registries and named by the
constants in `httk.analyse.definitions`.

They are taken byte-for-byte from the rendered output of
`httk-schemas-source` (`output/defs/v0.1/`) and published at
[schemas.httk.org](https://schemas.httk.org). Derivation terms (`mean`,
`standard_error`, ...) are not property definitions and are not vendored; only
their IRIs are used.

| Definition | `$id` | Source path | License |
| --- | --- | --- | --- |
| `bulk_modulus.json` | `https://schemas.httk.org/defs/v0.1/properties/mechanics/bulk_modulus` | `httk-schemas-source/output/defs/v0.1/properties/mechanics/bulk_modulus.json` | MIT |
| `bulk_modulus_hill.json` | `https://schemas.httk.org/defs/v0.1/properties/mechanics/bulk_modulus_hill` | `httk-schemas-source/output/defs/v0.1/properties/mechanics/bulk_modulus_hill.json` | MIT |
| `bulk_modulus_pressure_derivative.json` | `https://schemas.httk.org/defs/v0.1/properties/mechanics/bulk_modulus_pressure_derivative` | `httk-schemas-source/output/defs/v0.1/properties/mechanics/bulk_modulus_pressure_derivative.json` | MIT |
| `bulk_modulus_reuss.json` | `https://schemas.httk.org/defs/v0.1/properties/mechanics/bulk_modulus_reuss` | `httk-schemas-source/output/defs/v0.1/properties/mechanics/bulk_modulus_reuss.json` | MIT |
| `bulk_modulus_voigt.json` | `https://schemas.httk.org/defs/v0.1/properties/mechanics/bulk_modulus_voigt` | `httk-schemas-source/output/defs/v0.1/properties/mechanics/bulk_modulus_voigt.json` | MIT |
| `compliance_tensor.json` | `https://schemas.httk.org/defs/v0.1/properties/mechanics/compliance_tensor` | `httk-schemas-source/output/defs/v0.1/properties/mechanics/compliance_tensor.json` | MIT |
| `elastic_tensor.json` | `https://schemas.httk.org/defs/v0.1/properties/mechanics/elastic_tensor` | `httk-schemas-source/output/defs/v0.1/properties/mechanics/elastic_tensor.json` | MIT |
| `equilibrium_energy.json` | `https://schemas.httk.org/defs/v0.1/properties/mechanics/equilibrium_energy` | `httk-schemas-source/output/defs/v0.1/properties/mechanics/equilibrium_energy.json` | MIT |
| `equilibrium_volume.json` | `https://schemas.httk.org/defs/v0.1/properties/mechanics/equilibrium_volume` | `httk-schemas-source/output/defs/v0.1/properties/mechanics/equilibrium_volume.json` | MIT |
| `shear_modulus_hill.json` | `https://schemas.httk.org/defs/v0.1/properties/mechanics/shear_modulus_hill` | `httk-schemas-source/output/defs/v0.1/properties/mechanics/shear_modulus_hill.json` | MIT |
| `shear_modulus_reuss.json` | `https://schemas.httk.org/defs/v0.1/properties/mechanics/shear_modulus_reuss` | `httk-schemas-source/output/defs/v0.1/properties/mechanics/shear_modulus_reuss.json` | MIT |
| `shear_modulus_voigt.json` | `https://schemas.httk.org/defs/v0.1/properties/mechanics/shear_modulus_voigt` | `httk-schemas-source/output/defs/v0.1/properties/mechanics/shear_modulus_voigt.json` | MIT |
| `universal_anisotropy_index.json` | `https://schemas.httk.org/defs/v0.1/properties/mechanics/universal_anisotropy_index` | `httk-schemas-source/output/defs/v0.1/properties/mechanics/universal_anisotropy_index.json` | MIT |
| `energy_above_hull_per_atom.json` | `https://schemas.httk.org/defs/v0.1/properties/energetics/energy_above_hull_per_atom` | `httk-schemas-source/output/defs/v0.1/properties/energetics/energy_above_hull_per_atom.json` | MIT |
| `formation_energy_per_atom.json` | `https://schemas.httk.org/defs/v0.1/properties/energetics/formation_energy_per_atom` | `httk-schemas-source/output/defs/v0.1/properties/energetics/formation_energy_per_atom.json` | MIT |
| `reaction_energy.json` | `https://schemas.httk.org/defs/v0.1/properties/energetics/reaction_energy` | `httk-schemas-source/output/defs/v0.1/properties/energetics/reaction_energy.json` | MIT |
| `heat_capacity_constant_pressure.json` | `https://schemas.httk.org/defs/v0.1/properties/thermodynamics/heat_capacity_constant_pressure` | `httk-schemas-source/output/defs/v0.1/properties/thermodynamics/heat_capacity_constant_pressure.json` | MIT |
| `heat_capacity_constant_volume.json` | `https://schemas.httk.org/defs/v0.1/properties/thermodynamics/heat_capacity_constant_volume` | `httk-schemas-source/output/defs/v0.1/properties/thermodynamics/heat_capacity_constant_volume.json` | MIT |
| `helmholtz_free_energy.json` | `https://schemas.httk.org/defs/v0.1/properties/thermodynamics/helmholtz_free_energy` | `httk-schemas-source/output/defs/v0.1/properties/thermodynamics/helmholtz_free_energy.json` | MIT |
| `isothermal_compressibility.json` | `https://schemas.httk.org/defs/v0.1/properties/thermodynamics/isothermal_compressibility` | `httk-schemas-source/output/defs/v0.1/properties/thermodynamics/isothermal_compressibility.json` | MIT |
| `vibrational_entropy.json` | `https://schemas.httk.org/defs/v0.1/properties/thermodynamics/vibrational_entropy` | `httk-schemas-source/output/defs/v0.1/properties/thermodynamics/vibrational_entropy.json` | MIT |
| `vibrational_heat_capacity.json` | `https://schemas.httk.org/defs/v0.1/properties/thermodynamics/vibrational_heat_capacity` | `httk-schemas-source/output/defs/v0.1/properties/thermodynamics/vibrational_heat_capacity.json` | MIT |
| `vibrational_internal_energy.json` | `https://schemas.httk.org/defs/v0.1/properties/thermodynamics/vibrational_internal_energy` | `httk-schemas-source/output/defs/v0.1/properties/thermodynamics/vibrational_internal_energy.json` | MIT |
| `volumetric_thermal_expansion.json` | `https://schemas.httk.org/defs/v0.1/properties/thermodynamics/volumetric_thermal_expansion` | `httk-schemas-source/output/defs/v0.1/properties/thermodynamics/volumetric_thermal_expansion.json` | MIT |
| `zero_point_energy.json` | `https://schemas.httk.org/defs/v0.1/properties/thermodynamics/zero_point_energy` | `httk-schemas-source/output/defs/v0.1/properties/thermodynamics/zero_point_energy.json` | MIT |
| `diffusion_coefficient.json` | `https://schemas.httk.org/defs/v0.1/properties/transport/diffusion_coefficient` | `httk-schemas-source/output/defs/v0.1/properties/transport/diffusion_coefficient.json` | MIT |
| `diffusion_tensor.json` | `https://schemas.httk.org/defs/v0.1/properties/transport/diffusion_tensor` | `httk-schemas-source/output/defs/v0.1/properties/transport/diffusion_tensor.json` | MIT |
| `shear_viscosity.json` | `https://schemas.httk.org/defs/v0.1/properties/transport/shear_viscosity` | `httk-schemas-source/output/defs/v0.1/properties/transport/shear_viscosity.json` | MIT |
| `thermal_conductivity.json` | `https://schemas.httk.org/defs/v0.1/properties/transport/thermal_conductivity` | `httk-schemas-source/output/defs/v0.1/properties/transport/thermal_conductivity.json` | MIT |
| `thermal_conductivity_tensor.json` | `https://schemas.httk.org/defs/v0.1/properties/transport/thermal_conductivity_tensor` | `httk-schemas-source/output/defs/v0.1/properties/transport/thermal_conductivity_tensor.json` | MIT |
| `band_gap.json` | `https://schemas.httk.org/defs/v0.1/properties/electronic/band_gap` | `httk-schemas-source/output/defs/v0.1/properties/electronic/band_gap.json` | MIT |
| `direct_band_gap.json` | `https://schemas.httk.org/defs/v0.1/properties/electronic/direct_band_gap` | `httk-schemas-source/output/defs/v0.1/properties/electronic/direct_band_gap.json` | MIT |
| `vibrational_thermodynamics.json` | `https://schemas.httk.org/defs/v0.1/properties/thermodynamics/vibrational_thermodynamics` | `httk-schemas-source/output/defs/v0.1/properties/thermodynamics/vibrational_thermodynamics.json` | MIT |
| `quasiharmonic_thermodynamics.json` | `https://schemas.httk.org/defs/v0.1/properties/thermodynamics/quasiharmonic_thermodynamics` | `httk-schemas-source/output/defs/v0.1/properties/thermodynamics/quasiharmonic_thermodynamics.json` | MIT |
| `mean_squared_displacement.json` | `https://schemas.httk.org/defs/v0.1/properties/transport/mean_squared_displacement` | `httk-schemas-source/output/defs/v0.1/properties/transport/mean_squared_displacement.json` | MIT |
| `velocity_autocorrelation.json` | `https://schemas.httk.org/defs/v0.1/properties/transport/velocity_autocorrelation` | `httk-schemas-source/output/defs/v0.1/properties/transport/velocity_autocorrelation.json` | MIT |
| `diffusion_running_integral.json` | `https://schemas.httk.org/defs/v0.1/properties/transport/diffusion_running_integral` | `httk-schemas-source/output/defs/v0.1/properties/transport/diffusion_running_integral.json` | MIT |
| `thermal_conductivity_running_integral.json` | `https://schemas.httk.org/defs/v0.1/properties/transport/thermal_conductivity_running_integral` | `httk-schemas-source/output/defs/v0.1/properties/transport/thermal_conductivity_running_integral.json` | MIT |
| `shear_viscosity_running_integral.json` | `https://schemas.httk.org/defs/v0.1/properties/transport/shear_viscosity_running_integral` | `httk-schemas-source/output/defs/v0.1/properties/transport/shear_viscosity_running_integral.json` | MIT |
| `radial_distribution_function.json` | `https://schemas.httk.org/defs/v0.1/properties/structure/radial_distribution_function` | `httk-schemas-source/output/defs/v0.1/properties/structure/radial_distribution_function.json` | MIT |
| `electronic_density_of_states.json` | `https://schemas.httk.org/defs/v0.1/properties/electronic/electronic_density_of_states` | `httk-schemas-source/output/defs/v0.1/properties/electronic/electronic_density_of_states.json` | MIT |
| `relative_effective_mass.json` | `https://schemas.httk.org/defs/v0.1/properties/electronic/relative_effective_mass` | `httk-schemas-source/output/defs/v0.1/properties/electronic/relative_effective_mass.json` | MIT |
| `fermi_energy.json` | `https://schemas.httk.org/defs/v0.1/properties/electronic/fermi_energy` | `httk-schemas-source/output/defs/v0.1/properties/electronic/fermi_energy.json` | MIT |
| `static_relative_permittivity.json` | `https://schemas.httk.org/defs/v0.1/properties/electronic/static_relative_permittivity` | `httk-schemas-source/output/defs/v0.1/properties/electronic/static_relative_permittivity.json` | MIT |
| `high_frequency_relative_permittivity.json` | `https://schemas.httk.org/defs/v0.1/properties/electronic/high_frequency_relative_permittivity` | `httk-schemas-source/output/defs/v0.1/properties/electronic/high_frequency_relative_permittivity.json` | MIT |
| `total_magnetic_moment.json` | `https://schemas.httk.org/defs/v0.1/properties/magnetism/total_magnetic_moment` | `httk-schemas-source/output/defs/v0.1/properties/magnetism/total_magnetic_moment.json` | MIT |
| `charged_defect_formation_energy.json` | `https://schemas.httk.org/defs/v0.1/properties/defects/charged_defect_formation_energy` | `httk-schemas-source/output/defs/v0.1/properties/defects/charged_defect_formation_energy.json` | MIT |
| `charge_transition_level.json` | `https://schemas.httk.org/defs/v0.1/properties/defects/charge_transition_level` | `httk-schemas-source/output/defs/v0.1/properties/defects/charge_transition_level.json` | MIT |
| `surface_energy.json` | `https://schemas.httk.org/defs/v0.1/properties/defects/surface_energy` | `httk-schemas-source/output/defs/v0.1/properties/defects/surface_energy.json` | MIT |
| `adsorption_energy.json` | `https://schemas.httk.org/defs/v0.1/properties/defects/adsorption_energy` | `httk-schemas-source/output/defs/v0.1/properties/defects/adsorption_energy.json` | MIT |
| `segregation_energy.json` | `https://schemas.httk.org/defs/v0.1/properties/defects/segregation_energy` | `httk-schemas-source/output/defs/v0.1/properties/defects/segregation_energy.json` | MIT |
| `migration_barrier_forward.json` | `https://schemas.httk.org/defs/v0.1/properties/kinetics/migration_barrier_forward` | `httk-schemas-source/output/defs/v0.1/properties/kinetics/migration_barrier_forward.json` | MIT |
| `migration_barrier_reverse.json` | `https://schemas.httk.org/defs/v0.1/properties/kinetics/migration_barrier_reverse` | `httk-schemas-source/output/defs/v0.1/properties/kinetics/migration_barrier_reverse.json` | MIT |
| `activation_energy.json` | `https://schemas.httk.org/defs/v0.1/properties/kinetics/activation_energy` | `httk-schemas-source/output/defs/v0.1/properties/kinetics/activation_energy.json` | MIT |
| `total_energy_per_atom.json` | `https://schemas.httk.org/defs/v0.1/properties/energetics/total_energy_per_atom` | `httk-schemas-source/output/defs/v0.1/properties/energetics/total_energy_per_atom.json` | MIT |
| `steinhardt_bond_order.json` | `https://schemas.httk.org/defs/v0.1/properties/structure/steinhardt_bond_order` | `httk-schemas-source/output/defs/v0.1/properties/structure/steinhardt_bond_order.json` | MIT |
| `velocity_power_spectrum.json` | `https://schemas.httk.org/defs/v0.1/properties/dynamics/velocity_power_spectrum` | `httk-schemas-source/output/defs/v0.1/properties/dynamics/velocity_power_spectrum.json` | MIT |
| `distinct_van_hove_function.json` | `https://schemas.httk.org/defs/v0.1/properties/dynamics/distinct_van_hove_function` | `httk-schemas-source/output/defs/v0.1/properties/dynamics/distinct_van_hove_function.json` | MIT |
| `self_van_hove_function.json` | `https://schemas.httk.org/defs/v0.1/properties/dynamics/self_van_hove_function` | `httk-schemas-source/output/defs/v0.1/properties/dynamics/self_van_hove_function.json` | MIT |
| `intermediate_scattering_function.json` | `https://schemas.httk.org/defs/v0.1/properties/dynamics/intermediate_scattering_function` | `httk-schemas-source/output/defs/v0.1/properties/dynamics/intermediate_scattering_function.json` | MIT |
| `self_intermediate_scattering_function.json` | `https://schemas.httk.org/defs/v0.1/properties/dynamics/self_intermediate_scattering_function` | `httk-schemas-source/output/defs/v0.1/properties/dynamics/self_intermediate_scattering_function.json` | MIT |
| `diffusion_prefactor.json` | `https://schemas.httk.org/defs/v0.1/properties/transport/diffusion_prefactor` | `httk-schemas-source/output/defs/v0.1/properties/transport/diffusion_prefactor.json` | MIT |
| `arrhenius_prefactor.json` | `https://schemas.httk.org/defs/v0.1/properties/kinetics/arrhenius_prefactor` | `httk-schemas-source/output/defs/v0.1/properties/kinetics/arrhenius_prefactor.json` | MIT |

## License

The property schemas are MIT licensed; see [`LICENSE.httk`](./LICENSE.httk).
This differs from the httk source code, which is AGPL.

## Refreshing

Re-copy the listed files from the `httk-schemas-source` output (regenerate it
with that repository's generators first), then run `make test`.
