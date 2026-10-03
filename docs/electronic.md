# Electronic and magnetic quantities

`httk.analyse.matsci.electronic` analyzes supplied numerical arrays. It does
not read DFT output or infer a spin convention, energy zero, occupation limit,
or Brillouin-zone weighting.

## Density of states and chemical potential

`integrate_dos` integrates a nonnegative DOS that is linear between strictly
increasing energy samples. It clips requested bounds to the sampled interval
and interpolates the DOS at interior bounds. The DOS is in states/eV on the
spin basis chosen by its producer. `spin_degeneracy` explicitly multiplies the
integral (use 1 for a spin-resolved channel or a DOS already including all
channels; use 2 only for a one-channel spin-degenerate DOS).

`electron_count` applies a zero-temperature occupied step, with half
occupation at a grid point exactly equal to the chemical potential, or finite
temperature Fermi-Dirac factors. At positive temperature, the supplied linear
DOS times the occupation is integrated with 16-point Gauss-Legendre quadrature
on panels no wider than twice kBT around the thermal edge. Below 40 kBT from
the chemical potential the DOS integral is taken exactly; above 40 kBT the
Fermi tail is omitted, with an absolute count error bounded by
`integrated_DOS * exp(-40)` (the cutoff grows for very large integrated DOS).
At zero temperature, chemical potentials outside the sampled energy interval
return zero or the full integrated DOS.
`solve_chemical_potential` bisects the integrated count and reports an error
when the requested count is outside the integrated DOS. It does not rescale
the DOS to force a match. A zero-temperature count on a gap plateau has a
non-unique chemical potential; the bisection returns a point in that plateau.

## Bands and effective mass

`band_edges` accepts (bands, k points) energy and occupation arrays. Its
`maximum_occupation`, `occupation_tolerance`, and `energy_reference` are
explicit. Any state strictly between the empty and full tolerance limits marks
the result metallic. Indirect and direct gaps use only the supplied k points;
the direct gap is the minimum same-k separation among points containing both
occupied and empty states. Band-path data do not become a Brillouin-zone
integration through this analysis.

`fit_effective_mass` fits a full-rank local quadratic in Cartesian reciprocal
coordinates, in inverse angstroms, and energy, in eV. It returns the energy
Hessian in eV angstrom² and signed mass tensor in electron-mass units. The
conversion uses hbar²/m_e = 7.619964231073853 eV angstrom² from the 2022 CODATA
recommended constants. Negative principal mass indicates negative curvature
(a hole-like band edge); the fit does not change its sign. Rank-deficient
samples and singular Hessians raise `ValueError`; successful fits retain
condition and residual diagnostics.

`align_energies` subtracts only the reference explicitly supplied by the
caller. It does not identify a Fermi level or vacuum level.

## Moments and dielectric tensor

`magnetic_moments` sums site vectors in Bohr magnetons. Pair correlations use
the raw vector dot product in μB² unless `normalize_pairs=True`, in which case
they are dimensionless. It makes no magnetic-symmetry inference.

`summarize_dielectric` accepts a real symmetric, dimensionless static relative
dielectric tensor and returns ascending principal values, corresponding axes,
and their mean. Frequency-dependent dielectric spectra have a separate
contract and are not represented by this summary.

All result records are frozen dataclasses with copied tuple fields. Inputs
must be finite; this module does not guess missing scientific metadata.

```python
from httk.analyse.matsci.electronic import band_edges, integrate_dos

states = integrate_dos((0.0, 1.0, 2.0), (0.0, 2.0, 0.0), 0.5, 1.5, spin_degeneracy=1.0)
edges = band_edges(
    ((-1.0,), (0.5,)), ((2.0,), (0.0,)), maximum_occupation=2.0, energy_reference=0.0
)
assert states == 1.5
assert edges.indirect_gap == 1.5
```

References: [NIST 2022 CODATA recommended constants](https://physics.nist.gov/cuu/pdf/all.pdf).
