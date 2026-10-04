# Recording results

Every public value of an analysis result is in the unit of its property
definition, and values never carry unit strings. The definitions (IRIs in
`httk.analyse.definitions`, core quantities such as temperature in
`httk.core.definition_ids`) fix the name, unit, shape and meaning. Statistics of
a property, such as a mean or a standard error, keep the base definition and add
a derivation term (`MEAN`, `STANDARD_ERROR`, `RMSE`, ...) instead of defining a
new property.

## Records

`httk.analyse.records.records(result, *, product_of=(), **selection)` returns one
`httk.core.DataRecord` per scalar or fixed-size bound value of a result. Each
value is checked against its definition first:

```python
import numpy as np
from httk.core.units import default_registry
from httk.analyse.matsci import fit_birch_murnaghan
from httk.analyse.records import bound_values, records

volumes = np.linspace(8.5, 11.5, 9)
strain = (10.0 / volumes) ** (2 / 3) - 1
energies = -5.0 + 9 * 10.0 * 0.2 / 16 * (2 * strain**2 + 0.5 * strain**3)
fit = fit_birch_murnaghan(volumes, energies)
names = [record.name for record in records(fit)]
assert names == ["equilibrium_volume", "equilibrium_energy", "bulk_modulus", "bulk_modulus_pressure_derivative"]
# Input energies use eV and angstrom^3; the fitted bulk modulus is in GPa.
assert abs(fit.bulk_modulus - default_registry().convert(0.2, "angstrom^-3*eV", "GPa")) < 1e-6
```

`bound_values(result, **selection)` lists every binding as
`BoundValue(field, binding, value)`, where `binding` is a
`FieldBinding(definition, derivation=None, axis=None)`. `field` is a binding
name, not necessarily an attribute path: `isotropic` of a diffusion fit,
`integrals[4]` or `isotropic[4]` of a transport result at the selected lag, and
`standard_errors.<name>` of an equilibrium response are computed or selected
values. The EOS `rmse` is bound
to the core total energy with the `RMSE` derivation; it appears in
`bound_values` but not in `records`. Passing an object that has no bindings
raises `TypeError`.

Bound result types: `BirchMurnaghanFit`, `EOSFit`, `ElasticTensor` (stiffness,
compliance, Voigt/Reuss/Hill moduli, universal anisotropy), `FormationEnergy`,
`BandEdges` (gaps are 0.0 for a metal), `EquilibriumResponse`,
`HarmonicThermodynamics`, `QuasiHarmonicResult`, `DiffusionFit`,
`TransportResult`, `ReplicaTransport` and `PropertyParity` (statistics of its
`definition`).

## Explicit plateau selection

Running Green–Kubo integrals have no inferred plateau. Transport results
require `lag_index=`, the caller's choice of lag:

```python
from httk.analyse.matsci import thermal_conductivity

current = np.random.default_rng(0).normal(size=(40, 3))
result = thermal_conductivity(current, 0.01, temperature=300, volume=125, max_lag=6)
recorded = {record.name for record in records(result, lag_index=4)}
assert recorded == {"thermal_conductivity_tensor", "thermal_conductivity", "temperature", "volume"}
```

Omitting `lag_index` raises `TypeError`, and an index outside the lag grid
raises `ValueError`.

## Series and statistics

In this phase, only scalar and fixed-size values become records. Temperature
series (harmonic and quasi-harmonic properties, bound with
`axis="temperatures"`) and derivation-qualified values (standard errors, replica
means, parity statistics, fit RMSE) are recorded in the analysis summary
envelope instead: `analysis_summary(result, ...)` stores the same bindings, keyed by
binding name and each with its value, in its `fields` object, so the summary
and the records agree. See
{doc}`analysis-artifacts`, and {doc}`analysis-recipes` for storing both in
SQLite.
