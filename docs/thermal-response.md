# Equilibrium response and transport

## Ensemble fluctuations

`httk.analyse.matsci.thermodynamics.equilibrium_response` accepts an explicit
ensemble and temperature in K. NVT uses total energy in eV for
`Cv = var(E)/(kB*T**2)`. NPT uses enthalpy in eV and volume in angstrom³ for
`Cp = var(H)/(kB*T**2)`, isothermal compressibility
`var(V)/(kB*T*mean(V))`, and volumetric expansion
`cov(V,H)/(kB*T**2*mean(V))`. Heat capacities are extensive eV/K;
compressibility is angstrom³/eV; expansion is 1/K. Inputs must share one
extensive basis and describe a stationary equilibrated ensemble. Potential
energy alone generally does not give the total heat capacity.

Moments use population normalization. An optional explicit `block_size`
returns estimates per nonoverlapping block and their standard errors. At least
two complete blocks are required. A partial tail raises unless
`remainder="drop"` is supplied; the result reports dropped samples. Block
independence and convergence with block length remain scientific checks for
the caller. Finite blocks can bias nonlinear fluctuation estimates.

## Green–Kubo transport

`httk.analyse.matsci.transport.thermal_conductivity` consumes **extensive heat
current** in eV*angstrom/ps, uniform spacing in ps, temperature in K and fixed
volume in angstrom³. It returns the full running tensor in W/(m*K):

`kappa_ij(t) = integral <J_i(0) J_j(tau)> d tau / (kB*T**2*V)`.

`viscosity` consumes symmetric tensile stress in eV/angstrom³ and returns
running xy, xz and yz shear responses in Pa*s, using `V/(kB*T)` times their
autocorrelation integrals. Their mean is the isotropic estimate. This does not
compute a complete anisotropic viscosity tensor. Correlations use all
available time origins with lag-specific counts. Both functions subtract the
temporal mean by default, retain the raw dimensionful correlations, and use
trapezoidal running integration starting at zero.

[LAMMPS compute heat/flux](https://docs.lammps.org/compute_heat_flux.html)
returns a quantity that has **not** been divided by volume. Its microscopic
stress and current definitions require special care for many-body potentials.
The toolkit requires the caller to establish that the supplied current is
physically valid for the MLIP and engine implementation. Ordinary energy and
velocity output is insufficient to reconstruct it.

No automatic plateau is selected. Compare running integrals, lag truncation,
sampling interval, trajectory duration, system size and independent replicas.
`replica_transport` returns means and standard errors for matching protocols.
Explicit block analysis uses the same API:

```python
import numpy as np
from httk.analyse.matsci.transport import thermal_conductivity, replica_transport

current = np.random.default_rng(71).normal(size=(400, 3))
blocks = [thermal_conductivity(block, .001, temperature=300, volume=1000,
                              max_lag=20)
          for block in np.split(current, 4)]
summary = replica_transport(blocks)
assert summary.replicas == 4
```

This example checks the API; random currents are not a materials prediction.
Blocks must be long enough to support the desired correlation time and
sufficiently independent for the reported standard error to be meaningful.
