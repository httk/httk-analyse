# VASP electronic analysis adapters

`httk.analyse.integrations.vasp.dos_from_vasp` presents an exact DOSCAR result
as immutable float tuples for numerical analysis. `spin="total"` keeps a
nonmagnetic DOS unchanged and sums the two channels for a collinear DOS;
`spin="up"` and `spin="down"` select a collinear channel. The result records
the selected channel and whether its spin basis is spin-summed or resolved.
No degeneracy factor is applied to DOSCAR's nonmagnetic total.

`band_edges_from_wavefunctions` selects one zero-based spin channel from an
existing `PlaneWaveFunctions` object. WAVECAR stores eigenvalues and
occupations with axes `(spin, k point, band)`; the adapter supplies
`(band, k point)` arrays to `matsci.electronic.band_edges`. Maximum occupation,
occupation tolerance and energy reference are required explicitly. The result
describes sampled k points; it does not infer a Fermi reference or inspect
WAVECAR coefficients for weights.
