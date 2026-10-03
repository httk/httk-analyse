# Static equation-of-state fitting

Use {py:func}`httk.analyse.matsci.eos.fit_birch_murnaghan` for a third-order
Birch–Murnaghan fit of static energies against volume. At least five distinct
positive volumes are required. The result is an immutable
{py:class}`httk.analyse.matsci.eos.BirchMurnaghanFit`.

```python
from httk.analyse.matsci import fit_birch_murnaghan

# volumes: angstrom³; energies: eV, both per cell or both per atom.
fit = fit_birch_murnaghan(volumes, energies)
print(fit.equilibrium_volume, fit.equilibrium_energy)
print(fit.bulk_modulus_gpa, fit.bulk_modulus_derivative)
print(fit.rmse, fit.residuals)
pressure_ev_per_a3 = fit.pressure(fit.equilibrium_volume)
```

## Prepare a consistent volume scan

Use one composition and phase branch with consistent electronic settings,
energy definition, and relaxation protocol. Fixed-shape scaling and relaxation
of internal coordinates or cell shape at fixed volume need not give the same
response. Document the protocol alongside the result. A DFT scan must converge
energies with respect to basis size and Brillouin-zone sampling before its
curvature can be trusted.

The inputs determine normalization; the fitter cannot detect a total-energy
array paired with per-atom volumes. Scale both together. The modulus remains
unchanged when both extensive quantities are divided by the same atom count.

Mean total or potential energies from finite-temperature MD do not define an
isothermal pressure EOS. That requires the volume derivative of Helmholtz free
energy or a separately specified pressure-volume analysis. This routine does
not perform that analysis.

## Model and diagnostics

For $q=(V_0/V)^{2/3}-1$, the model is

$$E(V)=E_0+\frac{9B_0V_0}{16}\left[2q^2+(B'_0-4)q^3\right].$$

The implementation fits a cubic in $(V_\mathrm{ref}/V)^{2/3}$ using centered,
scaled linear least squares, then derives the equilibrium parameters. It
requires a stable minimum strictly inside the sampled volume interval. Invalid
or rank-deficient inputs and absent interior minima raise `ValueError`.

- `bulk_modulus` and `pressure(volume)` use eV/angstrom³. Pressure is positive
  under compression and equals $-dE/dV$.
- `bulk_modulus_gpa` multiplies by 160.2176634. `bulk_modulus_derivative` is
  dimensionless, $dB/dP$ at equilibrium.
- `residuals` are observed minus predicted energies in original input order;
  `rmse` is their root mean square in eV.
- `condition_number` describes the **scaled polynomial design**. It measures
  the linear solve, not how well the physical parameters are constrained.

A small residual and good design condition do not establish accurate $B_0$ or
$B'_0$. Inspect residuals, vary the fit window, and increase volume coverage.
The pressure derivative is especially sensitive to noise in a narrow window.
No parameter confidence intervals or fit-quality certification are returned.
`energy(volume)` and `pressure(volume)` allow extrapolation to positive finite
volumes; physical accuracy beyond the sampled interval is not established.

See the executed {doc}`notebooks/materials-toolbox` for a synthetic scan and
the [ASE EOS documentation](https://ase.gitlab.io/ase/ase/eos.html) for other
standard EOS families. Only BM3 is implemented here.
