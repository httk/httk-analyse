"""Independent analytic EOS fits and derivatives."""

import numpy as np
import pytest

pytest.importorskip("scipy")
from httk.analyse._constants import GPA_PER_EV_PER_A3
from httk.analyse.matsci.eos_models import EOSFit, fit_eos


def _model(name, bp):
    return EOSFit(name, 16, -5, 0.7 * GPA_PER_EV_PER_A3, bp, (), (), (), (), 0, 0, 0)


@pytest.mark.parametrize("model", ["birch-murnaghan", "murnaghan", "vinet"])
@pytest.mark.parametrize("bp", [0.0, 1.0, 4.5])
def test_recover_parameters_and_pressure_derivative(model, bp):
    source = _model(model, bp)
    volumes = np.linspace(13, 19, 17)
    energies = [source.energy(v) for v in volumes]
    fit = fit_eos(volumes, energies, model=model)
    assert fit.equilibrium_volume == pytest.approx(16, abs=1e-7)
    assert fit.equilibrium_energy == pytest.approx(-5, abs=1e-8)
    assert fit.bulk_modulus == pytest.approx(0.7 * GPA_PER_EV_PER_A3, abs=2e-5)
    assert fit.bulk_modulus_derivative == pytest.approx(bp, abs=2e-6)
    for volume in [14.0, 16.0, 18.0]:
        h = 1e-4
        derivative = (fit.energy(volume + h) - fit.energy(volume - h)) / (2 * h)
        assert fit.pressure(volume) == pytest.approx(-derivative * GPA_PER_EV_PER_A3, abs=2e-7)
    assert fit.pressure(fit.equilibrium_volume) == pytest.approx(0.0, abs=1e-6)


@pytest.mark.parametrize("model", ["murnaghan", "vinet", "birch-murnaghan"])
def test_independent_ase_energy(model):
    ase = pytest.importorskip("ase.eos")
    function = getattr(ase, model.replace("-", ""))
    source = _model(model, 4.5)
    for volume in np.linspace(12, 21, 15):
        assert source.energy(volume) == pytest.approx(function(volume, -5, 0.7, 4.5, 16), abs=1e-12)


def test_weighted_fit_suppresses_known_outlier_and_preserves_data():
    v = np.linspace(13, 19, 17)
    e = np.asarray([_model("vinet", 4.5).energy(x) for x in v])
    e[3] += 0.03
    weights = np.ones(17)
    weights[3] = 1e-8
    original = e.copy()
    equal = fit_eos(v, e, model="vinet")
    weighted = fit_eos(v, e, model="vinet", weights=weights)
    assert abs(weighted.equilibrium_volume - 16) < abs(equal.equilibrium_volume - 16) * 0.01
    np.testing.assert_array_equal(e, original)
    assert weighted.weighted_rmse < weighted.rmse


@pytest.mark.parametrize("weights", [[1], [1] * 16 + [0], [1] * 16 + [-1], [1] * 16 + [float('nan')]])
def test_bad_weights(weights):
    v = np.linspace(13, 19, 17)
    with pytest.raises(ValueError):
        fit_eos(v, [_model("vinet", 4.5).energy(x) for x in v], model="vinet", weights=weights)


@pytest.mark.parametrize('name', ['murnaghan', 'vinet', 'birch-murnaghan'])
def test_noisy_weighted_fit_matches_independent_ase_scipy_fit(name):
    ase = pytest.importorskip('ase.eos')
    from scipy.optimize import curve_fit

    function = getattr(ase, name.replace('-', ''))
    rng = np.random.default_rng(282)
    volumes = np.linspace(12.5, 20, 23)
    sigma = np.linspace(0.0001, 0.002, len(volumes))
    energies = function(volumes, -5, 0.7, 4.5, 16) + rng.normal(size=len(volumes)) * sigma
    reference, _ = curve_fit(
        function,
        volumes,
        energies,
        p0=(-5, 0.7, 4.5, 16),
        sigma=sigma,
        absolute_sigma=True,
        maxfev=10000,
        ftol=1e-12,
        xtol=1e-12,
    )
    actual = fit_eos(volumes, energies, model=name, weights=1 / sigma**2)
    np.testing.assert_allclose(
        [
            actual.equilibrium_energy,
            actual.bulk_modulus / GPA_PER_EV_PER_A3,
            actual.bulk_modulus_derivative,
            actual.equilibrium_volume,
        ],
        reference,
        rtol=2e-6,
        atol=2e-7,
    )
    np.testing.assert_allclose(actual.weights, 1 / sigma**2)


def test_direct_eos_result_rejects_mutable_numeric_entries():
    from dataclasses import replace

    with pytest.raises(ValueError):
        replace(_model('vinet', 4.5), volumes=[[]])
    with pytest.raises(ValueError):
        replace(_model('vinet', 4.5), rmse=float('inf'))
