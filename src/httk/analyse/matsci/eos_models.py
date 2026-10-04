"""Weighted static EOS fits using optional SciPy nonlinear least squares."""

import math
from collections.abc import Sequence
from dataclasses import dataclass
from importlib import import_module
from typing import Any, Literal

import numpy as np

from .._constants import GPA_PER_EV_PER_A3
from ..definitions import BoundValue, _bind_fields
from .eos import _EOS_BINDINGS, _finite_sequence, _positive_scalar, fit_birch_murnaghan

__all__ = ["EOSFit", "EOSModel", "fit_eos"]

type EOSModel = Literal["birch-murnaghan", "murnaghan", "vinet"]


@dataclass(frozen=True, slots=True)
class EOSFit:
    """An approximate static EOS fit with explicit relative weighting.

    The weighted Jacobian condition measures numerical conditioning, not
    confidence in physical parameters. Inputs must use one static branch,
    composition and electronic/relaxation protocol. Residuals are observed minus
    predicted. Pressure is positive in compression.

    :param model: EOS family.
    :param equilibrium_volume: Minimum volume in angstrom³.
    :param equilibrium_energy: Minimum energy in eV.
    :param bulk_modulus: Equilibrium bulk modulus in GPa.
    :param bulk_modulus_derivative: Dimensionless equilibrium pressure derivative.
    :param volumes: Input volumes in caller order.
    :param energies: Input energies in caller order.
    :param weights: Positive relative least-squares weights in caller order.
    :param residuals: Energy residuals in eV.
    :param rmse: Unweighted energy root mean squared error in eV.
    :param weighted_rmse: Weighted energy root mean squared error in eV.
    :param condition_number: Condition number of the scaled weighted parameter Jacobian.
    """

    model: EOSModel
    equilibrium_volume: float
    equilibrium_energy: float
    bulk_modulus: float
    bulk_modulus_derivative: float
    volumes: tuple[float, ...]
    energies: tuple[float, ...]
    weights: tuple[float, ...]
    residuals: tuple[float, ...]
    rmse: float
    weighted_rmse: float
    condition_number: float

    def __post_init__(self) -> None:
        """Copy sequence fields into immutable tuples."""
        if self.model not in ("birch-murnaghan", "murnaghan", "vinet"):
            raise ValueError("unknown EOS model")
        for name in ("equilibrium_volume", "bulk_modulus"):
            object.__setattr__(self, name, _positive_scalar(getattr(self, name), name))
        for name in ("equilibrium_energy", "bulk_modulus_derivative", "rmse", "weighted_rmse", "condition_number"):
            object.__setattr__(self, name, _finite_sequence((getattr(self, name),), name)[0])
        for name in ("volumes", "energies", "weights", "residuals"):
            object.__setattr__(self, name, _finite_sequence(getattr(self, name), name))

    def _bound_values(self, **selection: Any) -> tuple[BoundValue, ...]:
        """Bind the fitted parameters and the unweighted energy RMSE to property definitions."""
        return _bind_fields(self, selection, _EOS_BINDINGS)

    def energy(self, volume: float) -> float:
        """Evaluate energy, allowing explicit model extrapolation.

        :param volume: Positive volume in angstrom³ on the fit's extensive basis.
        :return: Model energy in eV.
        :raises ValueError: If the query or predicted energy is not representable.
        """
        ratio = _positive_scalar(volume, "volume") / self.equilibrium_volume
        value = (
            self.equilibrium_energy
            + self.bulk_modulus
            / GPA_PER_EV_PER_A3
            * self.equilibrium_volume
            * _energy_ratio(ratio, self.bulk_modulus_derivative, self.model)
        )
        return _finite(value)

    def pressure(self, volume: float) -> float:
        """Evaluate positive-compression pressure.

        :param volume: Positive volume in angstrom³.
        :return: Model pressure in GPa.
        :raises ValueError: If the query or predicted pressure is not representable.
        """
        ratio = _positive_scalar(volume, "volume") / self.equilibrium_volume
        return _finite(self.bulk_modulus * _pressure_ratio(ratio, self.bulk_modulus_derivative, self.model))


def fit_eos(
    volumes: Sequence[float], energies: Sequence[float], *, model: EOSModel, weights: Sequence[float] | None = None
) -> EOSFit:
    """Fit BM3, Murnaghan or Vinet static energies with optional relative weights.

    Requires ``httk-analyse[scipy]``. Weights multiply squared residuals and need
    not sum to one. The initial NumPy BM3 fit must have an interior stable
    minimum; the nonlinear fit also requires an interior positive-modulus
    minimum and full Jacobian rank. Results contain no inferred error bars.
    Fit-window sensitivity and comparing physical protocols remain necessary.

    :param volumes: At least five distinct positive volumes in angstrom³.
    :param energies: Matching static energies in eV on the same extensive basis.
    :param model: Explicit EOS family to fit.
    :param weights: Positive finite relative weights, or equal weighting.
    :return: Immutable fit with input-order residuals and numerical diagnostics.
    :raises ImportError: If SciPy is unavailable.
    :raises ValueError: If data, weights, minimum, convergence or rank are invalid.
    """
    if model not in ("birch-murnaghan", "murnaghan", "vinet"):
        raise ValueError("unknown EOS model")
    try:
        least_squares = import_module("scipy.optimize").least_squares
    except ImportError as exc:
        raise ImportError("nonlinear EOS fits require httk-analyse[scipy]") from exc
    initial = fit_birch_murnaghan(volumes, energies)
    v = np.asarray(initial.volumes)
    e = np.asarray(initial.energies)
    w = np.ones(len(v)) if weights is None else np.asarray(_finite_sequence(weights, "weights"))
    if w.shape != v.shape or np.any(w <= 0):
        raise ValueError("weights must match volumes and be strictly positive")
    input_weights = tuple(float(value) for value in w)
    w = w / np.max(w)
    if np.any(w == 0):
        raise ValueError("relative weight range is not representable")
    # Normalize physical units before optimization; a change of extensive basis
    # should not dictate finite-difference steps or convergence tolerances.
    vscale = initial.equilibrium_volume
    escale = float(np.ptp(e))
    eoffset = float(np.min(e))
    target = (e - eoffset) / escale
    x = v / vscale
    root_w = np.sqrt(w)

    def predictions(parameters: np.ndarray) -> np.ndarray:
        v0, e0, b0, bp = parameters
        return np.asarray([e0 + b0 * v0 * _energy_ratio(float(value / v0), float(bp), model) for value in x])

    def residual(parameters: np.ndarray) -> np.ndarray:
        return (predictions(parameters) - target) * root_w

    start = np.asarray(
        [
            1.0,
            (initial.equilibrium_energy - eoffset) / escale,
            initial.bulk_modulus / GPA_PER_EV_PER_A3 * vscale / escale,
            initial.bulk_modulus_derivative,
        ]
    )
    bounds = ([float(min(x)), -np.inf, 0.0, -np.inf], [float(max(x)), np.inf, np.inf, np.inf])
    try:
        fitted = least_squares(
            residual, start, bounds=bounds, x_scale="jac", max_nfev=5000, ftol=1e-12, xtol=1e-12, gtol=1e-12
        )
    except (OverflowError, FloatingPointError, ValueError) as exc:
        raise ValueError("EOS optimization failed to produce finite parameters") from exc
    if not fitted.success or not np.isfinite(fitted.x).all() or np.any(fitted.active_mask):
        raise ValueError("EOS fit did not converge to an interior positive-modulus minimum")
    singular = np.linalg.svd(fitted.jac, compute_uv=False)
    if singular[-1] <= np.finfo(float).eps * max(fitted.jac.shape) * singular[0]:
        raise ValueError("EOS weighted parameter Jacobian is rank deficient")
    v0, e0, b0, bp = (float(value) for value in fitted.x)
    residuals = (target - predictions(fitted.x)) * escale
    rmse = float(np.linalg.norm(residuals / math.sqrt(len(v))))
    weighted_rmse = float(np.linalg.norm(residuals * np.sqrt(w / np.sum(w))))
    parameters = (v0 * vscale, e0 * escale + eoffset, b0 * escale / vscale * GPA_PER_EV_PER_A3, bp)
    if not all(math.isfinite(value) for value in parameters) or not np.isfinite(residuals).all():
        raise ValueError("EOS fit results are not finite")
    return EOSFit(
        model,
        *parameters,
        tuple(v),
        tuple(e),
        input_weights,
        tuple(float(value) for value in residuals),
        rmse,
        weighted_rmse,
        float(singular[0] / singular[-1]),
    )


def _finite(value: float) -> float:
    if not math.isfinite(value):
        raise ValueError("EOS prediction is not finite")
    return value


def _pressure_ratio(ratio: float, bp: float, model: EOSModel) -> float:
    try:
        logarithm = math.log(ratio)
        if model == "murnaghan":
            # expm1(t)/t has a removable singularity at B'=0.
            arg = -bp * logarithm
            factor = 1.0 if arg == 0 else math.expm1(arg) / arg
            value = -logarithm * factor
        elif model == "vinet":
            strain = -math.expm1(logarithm / 3)
            value = 3 * strain * math.exp(-2 * logarithm / 3 + 1.5 * (bp - 1) * strain)
        else:
            eta = math.expm1(-2 * logarithm / 3)
            value = 1.5 * eta * math.exp(-5 * logarithm / 3) * (1 + 0.75 * (bp - 4) * eta)
    except (OverflowError, ValueError) as exc:
        raise ValueError("EOS pressure cannot be represented at this volume") from exc
    return _finite(value)


def _energy_ratio(ratio: float, bp: float, model: EOSModel) -> float:
    # Integrating the dimensionless pressure avoids removable singularities in
    # the closed energy expressions at Murnaghan B'=0/1 and Vinet B'=1.
    quad = import_module("scipy.integrate").quad
    integral, error = quad(lambda value: -_pressure_ratio(value, bp, model), 1.0, ratio, epsabs=1e-13, epsrel=1e-12)
    if not math.isfinite(integral) or error > 1e-10 * max(1.0, abs(integral)):
        raise ValueError("EOS energy integration did not converge")
    return float(integral)
