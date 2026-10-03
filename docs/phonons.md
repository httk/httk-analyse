# Harmonic phonon thermodynamics

`httk.analyse.matsci.phonons.harmonic_thermodynamics` evaluates the harmonic
oscillator free energy, internal energy, entropy and constant-volume heat
capacity from signed frequencies in THz. Energies use eV and entropies and heat
capacities use eV/K. Results are extensive in the supplied mode weights; the
routine does not infer a `3N` normalization or normalize weights.

```python
from httk.analyse.matsci.phonons import harmonic_thermodynamics

properties = harmonic_thermodynamics(
    [1.0, 2.0, 3.0], [0.0, 100.0, 300.0],
    weights=[1.0, 1.0, 1.0],
)
print(properties.free_energy, properties.heat_capacity)
```

At zero kelvin, `free_energy` and `internal_energy` equal the zero-point energy;
entropy and heat capacity are exactly zero. Frequencies at zero and below zero
raise by default. `zero_modes="omit"` and `imaginary="omit"` explicitly remove
those modes, with their summed weights reported in the result. Imaginary modes
are never converted to absolute frequencies. Weights may be zero, but must be
nonnegative with positive total. Input arrays are copied and left unchanged.

`harmonic_thermodynamics_from_dos` accepts density in states/THz and uses
trapezoid node weights, including half weights at grid endpoints. The reported
retained weight is the integrated number of states. A zero-frequency endpoint
has quadrature weight and therefore follows the same explicit zero-mode policy.

## Volume response

`mode_gruneisen` fits log frequency against log volume for modes already
matched by the caller. It uses a local quadratic with at least three volumes
and a linear fit with two. Its derivative is
`-d(log(frequency))/d(log(volume))`; no branch matching is inferred.

`quasiharmonic` combines static energies and harmonic free energies at every
sampled volume, then fits the existing Birch–Murnaghan model independently at
each temperature. Provide at least five distinct volumes that bracket a stable
minimum, a common frequency column layout, and at least three strictly
increasing temperatures. The frequencies and static energies must use the same
cell or per-atom normalization. If a mode is omitted at one volume, the same
matched modes must be omitted at every volume. `volumetric_expansion` is
`(1/V) dV/dT` from `numpy.gradient` on the nonuniform temperature grid, with
the library's second-order endpoint convention.

## Phonopy mesh

`harmonic_from_phonopy` consumes an already sampled public Phonopy mesh result.
It normalizes q-point multiplicities to sum to one and preserves each branch,
so the result is per primitive-cell mode set. It does not create displacements,
force constants, or a mesh. Phonopy's default frequency unit is THz; if the
object was configured with another frequency conversion factor, convert its
frequencies before calling the adapter.

These are harmonic and quasi-harmonic approximations. The quasi-harmonic fit
does not account for explicit anharmonicity or validate the volume dependence
of the force constants. Mesh convergence, phase stability and mode matching
remain scientific checks for the caller.

The equations and mesh normalization follow the
[Phonopy formulation](https://phonopy.github.io/phonopy/formulation.html) and
[public Python mesh/thermal API](https://phonopy.github.io/phonopy/phonopy-module.html).
The oscillator constants here use exact SI h, kB and e. Phonopy versions using
older constants can differ by a few parts per million, amplified at low
occupation. Install `httk-analyse[phonopy]` when constructing Phonopy objects;
the numerical kernels themselves require only NumPy.
