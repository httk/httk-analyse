"""Run a synthetic static volume scan and stationary correlated-series analysis."""

import numpy as np
from httk.core.units import default_registry

from httk.analyse.generic import autocorrelation, block_average
from httk.analyse.matsci import fit_birch_murnaghan


def main() -> None:
    """Verify EOS recovery and print block-size sensitivity for a seeded series."""
    # Synthetic single-branch static energies; no finite-temperature averaging.
    # A real scan must pin its energy definition and fixed-volume relaxation protocol.
    volumes = np.linspace(14.0, 18.0, 13)
    v0, e0, b0, bp = 16.0, -10.0, 0.6, 4.5
    strain = (v0 / volumes) ** (2.0 / 3.0) - 1.0
    energies = e0 + 9.0 * b0 * v0 / 16.0 * (2.0 * strain**2 + (bp - 4.0) * strain**3)
    fit = fit_birch_murnaghan(volumes.tolist(), energies.tolist())
    assert abs(fit.equilibrium_volume - v0) < 1e-8
    assert abs(fit.bulk_modulus - default_registry().convert(b0, "angstrom^-3*eV", "GPa")) < 1e-6
    assert abs(fit.bulk_modulus_derivative - bp) < 1e-7
    assert abs(fit.pressure(v0)) < 1e-9
    print(f"Equilibrium volume: {fit.equilibrium_volume:.6f} angstrom^3")
    print(f"Bulk modulus: {fit.bulk_modulus:.6f} GPa")
    print(f"EOS RMSE: {fit.rmse:.3g} eV")

    # AR(1) initialized from its stationary distribution, with unit variance.
    # This is a statistics example, not a synthetic claim about a real material.
    rng = np.random.default_rng(20261003)
    series = np.empty(8192)
    series[0] = rng.normal()
    coefficient = 0.9
    for index in range(1, len(series)):
        series[index] = coefficient * series[index - 1] + np.sqrt(1.0 - coefficient**2) * rng.normal()
    values = series.tolist()
    correlation = autocorrelation(values, max_lag=20)
    assert correlation[0] == 1.0
    assert 0.8 < correlation[1] < 0.98
    print(f"Lag-one autocorrelation: {correlation[1]:.4f}")
    for size in (1, 8, 32, 128):
        summary = block_average(values, block_size=size)
        assert summary.used_samples == len(values)
        print(f"Block size {size:3d}: {len(summary.block_means):4d} blocks, mean SE {summary.standard_error:.5f}")
    print("Compare block sizes and retain enough blocks; these estimates do not certify independence.")


if __name__ == "__main__":
    main()
