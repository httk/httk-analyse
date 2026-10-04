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
| `bulk_modulus.json` | `https://schemas.httk.org/defs/v0.1/properties/mechanics/bulk_modulus` | `httk-schemas-source/output/properties/mechanics/bulk_modulus.json` | MIT |
| `bulk_modulus_hill.json` | `https://schemas.httk.org/defs/v0.1/properties/mechanics/bulk_modulus_hill` | `httk-schemas-source/output/properties/mechanics/bulk_modulus_hill.json` | MIT |
| `bulk_modulus_pressure_derivative.json` | `https://schemas.httk.org/defs/v0.1/properties/mechanics/bulk_modulus_pressure_derivative` | `httk-schemas-source/output/properties/mechanics/bulk_modulus_pressure_derivative.json` | MIT |
| `bulk_modulus_reuss.json` | `https://schemas.httk.org/defs/v0.1/properties/mechanics/bulk_modulus_reuss` | `httk-schemas-source/output/properties/mechanics/bulk_modulus_reuss.json` | MIT |
| `bulk_modulus_voigt.json` | `https://schemas.httk.org/defs/v0.1/properties/mechanics/bulk_modulus_voigt` | `httk-schemas-source/output/properties/mechanics/bulk_modulus_voigt.json` | MIT |
| `compliance_tensor.json` | `https://schemas.httk.org/defs/v0.1/properties/mechanics/compliance_tensor` | `httk-schemas-source/output/properties/mechanics/compliance_tensor.json` | MIT |
| `elastic_tensor.json` | `https://schemas.httk.org/defs/v0.1/properties/mechanics/elastic_tensor` | `httk-schemas-source/output/properties/mechanics/elastic_tensor.json` | MIT |
| `equilibrium_energy.json` | `https://schemas.httk.org/defs/v0.1/properties/mechanics/equilibrium_energy` | `httk-schemas-source/output/properties/mechanics/equilibrium_energy.json` | MIT |
| `equilibrium_volume.json` | `https://schemas.httk.org/defs/v0.1/properties/mechanics/equilibrium_volume` | `httk-schemas-source/output/properties/mechanics/equilibrium_volume.json` | MIT |
| `shear_modulus_hill.json` | `https://schemas.httk.org/defs/v0.1/properties/mechanics/shear_modulus_hill` | `httk-schemas-source/output/properties/mechanics/shear_modulus_hill.json` | MIT |
| `shear_modulus_reuss.json` | `https://schemas.httk.org/defs/v0.1/properties/mechanics/shear_modulus_reuss` | `httk-schemas-source/output/properties/mechanics/shear_modulus_reuss.json` | MIT |
| `shear_modulus_voigt.json` | `https://schemas.httk.org/defs/v0.1/properties/mechanics/shear_modulus_voigt` | `httk-schemas-source/output/properties/mechanics/shear_modulus_voigt.json` | MIT |
| `universal_anisotropy_index.json` | `https://schemas.httk.org/defs/v0.1/properties/mechanics/universal_anisotropy_index` | `httk-schemas-source/output/properties/mechanics/universal_anisotropy_index.json` | MIT |
| `energy_above_hull_per_atom.json` | `https://schemas.httk.org/defs/v0.1/properties/energetics/energy_above_hull_per_atom` | `httk-schemas-source/output/properties/energetics/energy_above_hull_per_atom.json` | MIT |
| `formation_energy_per_atom.json` | `https://schemas.httk.org/defs/v0.1/properties/energetics/formation_energy_per_atom` | `httk-schemas-source/output/properties/energetics/formation_energy_per_atom.json` | MIT |
| `reaction_energy.json` | `https://schemas.httk.org/defs/v0.1/properties/energetics/reaction_energy` | `httk-schemas-source/output/properties/energetics/reaction_energy.json` | MIT |
| `heat_capacity_constant_pressure.json` | `https://schemas.httk.org/defs/v0.1/properties/thermodynamics/heat_capacity_constant_pressure` | `httk-schemas-source/output/properties/thermodynamics/heat_capacity_constant_pressure.json` | MIT |
| `heat_capacity_constant_volume.json` | `https://schemas.httk.org/defs/v0.1/properties/thermodynamics/heat_capacity_constant_volume` | `httk-schemas-source/output/properties/thermodynamics/heat_capacity_constant_volume.json` | MIT |
| `helmholtz_free_energy.json` | `https://schemas.httk.org/defs/v0.1/properties/thermodynamics/helmholtz_free_energy` | `httk-schemas-source/output/properties/thermodynamics/helmholtz_free_energy.json` | MIT |
| `isothermal_compressibility.json` | `https://schemas.httk.org/defs/v0.1/properties/thermodynamics/isothermal_compressibility` | `httk-schemas-source/output/properties/thermodynamics/isothermal_compressibility.json` | MIT |
| `vibrational_entropy.json` | `https://schemas.httk.org/defs/v0.1/properties/thermodynamics/vibrational_entropy` | `httk-schemas-source/output/properties/thermodynamics/vibrational_entropy.json` | MIT |
| `vibrational_heat_capacity.json` | `https://schemas.httk.org/defs/v0.1/properties/thermodynamics/vibrational_heat_capacity` | `httk-schemas-source/output/properties/thermodynamics/vibrational_heat_capacity.json` | MIT |
| `vibrational_internal_energy.json` | `https://schemas.httk.org/defs/v0.1/properties/thermodynamics/vibrational_internal_energy` | `httk-schemas-source/output/properties/thermodynamics/vibrational_internal_energy.json` | MIT |
| `volumetric_thermal_expansion.json` | `https://schemas.httk.org/defs/v0.1/properties/thermodynamics/volumetric_thermal_expansion` | `httk-schemas-source/output/properties/thermodynamics/volumetric_thermal_expansion.json` | MIT |
| `zero_point_energy.json` | `https://schemas.httk.org/defs/v0.1/properties/thermodynamics/zero_point_energy` | `httk-schemas-source/output/properties/thermodynamics/zero_point_energy.json` | MIT |
| `diffusion_coefficient.json` | `https://schemas.httk.org/defs/v0.1/properties/transport/diffusion_coefficient` | `httk-schemas-source/output/properties/transport/diffusion_coefficient.json` | MIT |
| `diffusion_tensor.json` | `https://schemas.httk.org/defs/v0.1/properties/transport/diffusion_tensor` | `httk-schemas-source/output/properties/transport/diffusion_tensor.json` | MIT |
| `shear_viscosity.json` | `https://schemas.httk.org/defs/v0.1/properties/transport/shear_viscosity` | `httk-schemas-source/output/properties/transport/shear_viscosity.json` | MIT |
| `thermal_conductivity.json` | `https://schemas.httk.org/defs/v0.1/properties/transport/thermal_conductivity` | `httk-schemas-source/output/properties/transport/thermal_conductivity.json` | MIT |
| `thermal_conductivity_tensor.json` | `https://schemas.httk.org/defs/v0.1/properties/transport/thermal_conductivity_tensor` | `httk-schemas-source/output/properties/transport/thermal_conductivity_tensor.json` | MIT |
| `band_gap.json` | `https://schemas.httk.org/defs/v0.1/properties/electronic/band_gap` | `httk-schemas-source/output/properties/electronic/band_gap.json` | MIT |
| `direct_band_gap.json` | `https://schemas.httk.org/defs/v0.1/properties/electronic/direct_band_gap` | `httk-schemas-source/output/properties/electronic/direct_band_gap.json` | MIT |

## License

The property schemas are MIT licensed; see [`LICENSE.httk`](./LICENSE.httk).
This differs from the httk source code, which is AGPL.

## Refreshing

Re-copy the listed files from the `httk-schemas-source` output (regenerate it
with that repository's generators first), then run `make test`.
