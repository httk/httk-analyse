# Summaries and figures

`httk.analyse.summary.analysis_summary` serializes a numerical result alongside
its algorithm identifier, installed software versions, field meanings,
parameters, sample selection and scientific assumptions. Each source file is read in chunks
to record its locator, byte size and SHA-256 digest. Source frames are not
embedded. Keep source files unchanged from calculation through hashing; a hash
of a later modified file would not describe the calculation's input.

The returned `AnalysisSummary` stores canonical JSON, and `.value` returns a
fresh decoded copy. `.write(path)` writes a selected output file. Standard
frozen result dataclasses are supported; complex numbers have named `real` and
`imaginary` fields. Nonfinite values and unsupported objects raise instead of
being silently converted to strings. Installed versions identify distributions;
record an unreleased source commit in `parameters` when analyzing a development
checkout.

The envelope's `fields` object says what each result field means. For a result
type bound to property definitions (see {doc}`records`), every bound field maps
to `{"definition": <property IRI>, "derivation": <derivation IRI or null>,
"axis": <coordinate field or null>, "value": <bound value>}`, the same bindings
and values `records()` uses; selection keywords such as `lag_index=` are passed
through. The keys are binding names, not paths into `result`: `isotropic[4]`
or `standard_errors.heat_capacity_constant_pressure` name a derived or selected
value, so each entry carries its own value. Fields without a
published definition (RDF, MSD or correlation series, fit diagnostics) may get
an OPTIMADE unit expression through `units=`, validated against the vendored
OPTIMADE unit definitions and stored as `{"unit": ..., "unit_definitions":
[<unit IRIs>]}`. `dimensionless` is accepted; malformed expressions (such as
`eV/angstrom`; write `angstrom^-1*eV`) and units for fields that have a
definition raise `ValueError`.

```python
from httk.analyse.matsci.phonons import harmonic_thermodynamics
from httk.analyse.summary import analysis_summary

result = harmonic_thermodynamics([2, 3, 4], [0, 300])
summary = analysis_summary(
    result, algorithm="httk.analyse.matsci.phonons.harmonic_thermodynamics",
    units={"cutoff_frequency": "THz"},
    parameters={"frequencies_THz": [2, 3, 4]}, selection={"modes": "all"},
    sources=[], assumptions=["synthetic three-mode example"],
)
assert summary.value["result"]["retained_mode_weight"] == 3
assert summary.value["fields"]["free_energy"]["axis"] == "temperatures"
```

## Plotting

`httk.analyse.plotting` returns Matplotlib figures and axes. It never opens a
window or saves a file implicitly; callers own figure closure and choose output
paths. Helpers include `plot_eos` (data, fit and residuals),
`plot_convergence`, `plot_rdf`, `plot_msd`, `plot_transport`, `plot_phonons`
and `plot_parity`. Axes of quantities with a property definition are labelled
with the definition's title and unit, for example `Bulk modulus (GPa)`;
`plot_parity` uses the result's `definition` and omits the unit when it is
`None`. Plotting a running transport integral does not select a plateau.

`plot_chemical_potential_slice` takes a host/competitor region, two plotted
elements, one dependent element eliminated by the host equality, and fixed
potentials for every remaining element. Both plot ranges are explicit.
Feasibility shading is sampled on a declared grid; use the original region's
`contains` method for numerical membership, not pixels near a drawn boundary.

The executable materials-response notebook illustrates these routines. The
analysis recipes show file-backed summaries and storage with provenance.
