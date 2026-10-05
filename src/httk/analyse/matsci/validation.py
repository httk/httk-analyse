"""Property, conservation and derivative checks for interatomic models."""

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from .. import definitions as defs
from ..definitions import BoundValue, FieldBinding, _reject_selection, _series
from .dynamics import _array
from .mlip import ErrorStatistics, _statistics

__all__ = [
    "CommitteeSpread",
    "EnergyDrift",
    "PropertyParity",
    "committee_spread",
    "energy_drift",
    "force_energy_consistency",
    "property_parity",
]


@dataclass(frozen=True, slots=True)
class PropertyParity:
    """Paired scalar predictions on an explicitly matched property basis.

    :param labels: Label for each paired scalar.
    :param definition: Property-definition IRI fixing the common unit, or ``None`` when no definition is published.
    :param reference: Reference scalars.
    :param predicted: Predicted scalars.
    :param residuals: Prediction minus reference.
    :param statistics: Equally weighted residual summary.
    """

    labels: tuple[str, ...]
    definition: str | None
    reference: tuple[float, ...]
    predicted: tuple[float, ...]
    residuals: tuple[float, ...]
    statistics: ErrorStatistics

    def __post_init__(self) -> None:
        """Copy labels and numerical values into immutable tuples."""
        object.__setattr__(self, "labels", tuple(self.labels))
        for name in ("reference", "predicted", "residuals"):
            object.__setattr__(self, name, tuple(float(v) for v in getattr(self, name)))

    def _bound_values(self, **selection: Any) -> tuple[BoundValue, ...]:
        """Bind the error statistics as derivations of ``definition``; nothing when it is ``None``."""
        _reject_selection(self, selection)
        if self.definition is None:
            return ()
        return tuple(
            BoundValue(
                f"statistics.{name}", FieldBinding(self.definition, derivation), float(getattr(self.statistics, name))
            )
            for name, derivation in (
                ("rmse", defs.RMSE),
                ("mae", defs.MAE),
                ("bias", defs.BIAS),
                ("maximum_absolute_error", defs.MAXIMUM_ABSOLUTE_ERROR),
            )
        )


@dataclass(frozen=True, slots=True)
class EnergyDrift:
    """Linear total-energy conservation check on a selected NVE interval.

    :param slope: Fitted slope in eV/atom/ps.
    :param intercept: Energy in eV/atom at the first selected time.
    :param residual_rms: Residual RMS in eV/atom.
    :param endpoint_change: Last minus first observed energy in eV/atom.
    :param start: First time in ps.
    :param stop: Last time in ps.
    :param samples: Number of selected samples.
    """

    slope: float
    intercept: float
    residual_rms: float
    endpoint_change: float
    start: float
    stop: float
    samples: int

    def _bound_values(self, **selection: Any) -> tuple[BoundValue, ...]:
        """Bind the drift fit as one dictionary value."""
        _reject_selection(self, selection)
        return (
            _series(
                defs.NVE_ENERGY_DRIFT,
                slope=float(self.slope),
                intercept=float(self.intercept),
                residual_rms=float(self.residual_rms),
                endpoint_change=float(self.endpoint_change),
                start=float(self.start),
                stop=float(self.stop),
                samples=int(self.samples),
            ),
        )


@dataclass(frozen=True, slots=True)
class CommitteeSpread:
    """Model spread on corresponding scalar predictions, without calibration.

    :param mean: Per-entry arithmetic mean across models.
    :param standard_deviation: Per-entry model standard deviation.
    :param models: Number of committee members.
    :param ddof: Degrees of freedom removed from the variance denominator.
    :param definition: Property-definition IRI fixing the common prediction unit.
    """

    mean: tuple[float, ...]
    standard_deviation: tuple[float, ...]
    models: int
    ddof: int
    definition: str

    def __post_init__(self) -> None:
        """Copy numerical summaries into immutable tuples."""
        for name in ("mean", "standard_deviation"):
            object.__setattr__(self, name, tuple(float(v) for v in getattr(self, name)))


def property_parity(reference: Any, predicted: Any, *, labels: Sequence[str], definition: str | None) -> PropertyParity:
    """Compare already matched scalar properties with equal sample weights.

    :param reference: Finite one-dimensional reference values.
    :param predicted: Corresponding finite predictions in the unit of ``definition``.
    :param labels: One nonempty label per pair; repeated labels are allowed.
    :param definition: Nonempty property-definition IRI (not loaded here), or ``None`` when the quantity has no
        published definition and the producing routine documents its unit.
    :return: Paired data, residuals and their statistics.
    :raises ValueError: If inputs are empty, mismatched, nonfinite or lack labels or a valid definition.
    """
    ref, pred = _array(reference, "reference"), _array(predicted, "predicted")
    names = tuple(labels) if not isinstance(labels, (str, bytes)) else ()
    if ref.ndim != 1 or not len(ref) or pred.shape != ref.shape:
        raise ValueError("reference and prediction must be matching nonempty vectors")
    if len(names) != len(ref) or any(not isinstance(v, str) or not v.strip() for v in names):
        raise ValueError("one nonempty label per scalar pair is required")
    if definition is not None:
        _definition(definition)
    residual = pred - ref
    stats = _statistics(residual, np.full(len(ref), 1 / len(ref)))
    return PropertyParity(names, definition, tuple(ref), tuple(pred), tuple(residual), stats)


def energy_drift(times: Any, energies: Any, *, atom_count: int, ensemble: str) -> EnergyDrift:
    """Fit total energy per atom against time for an explicitly NVE trajectory.

    :param times: Strictly increasing selected physical times in ps.
    :param energies: Total energies in eV on a constant atom-count basis.
    :param atom_count: Positive integer number of atoms.
    :param ensemble: Must explicitly be 'NVE'.
    :return: Slope, first-time intercept, residual RMS and observed endpoint change.
    :raises ValueError: If the ensemble, inputs or derived fit is invalid.
    """
    if ensemble != "NVE":
        raise ValueError("energy conservation drift requires explicit NVE ensemble")
    if isinstance(atom_count, bool) or not isinstance(atom_count, int) or atom_count <= 0:
        raise ValueError("atom_count must be a positive integer")
    t, energy = _array(times, "times"), _array(energies, "energies") / atom_count
    if t.ndim != 1 or len(t) < 3 or energy.shape != t.shape or np.any(np.diff(t) <= 0):
        raise ValueError("at least three increasing times and matching energies are required")
    span = t[-1] - t[0]
    design = np.column_stack((np.ones(len(t)), (t - t[0]) / span))
    offset = energy[0]
    coefficients, _, rank, _ = np.linalg.lstsq(design, energy - offset, rcond=None)
    residual = energy - offset - design @ coefficients
    output = (coefficients[1] / span, coefficients[0] + offset, np.sqrt(np.mean(residual**2)), energy[-1] - energy[0])
    if rank != 2 or not np.isfinite(output).all():
        raise ValueError("energy drift fit is nonfinite or rank deficient")
    return EnergyDrift(
        float(output[0]), float(output[1]), float(output[2]), float(output[3]), float(t[0]), float(t[-1]), len(t)
    )


def force_energy_consistency(
    energy: Callable[[np.ndarray], float], positions: Any, forces: Any, *, displacement: float
) -> PropertyParity:
    """Compare supplied forces to central finite differences of a total energy.

    Executes 6*N energy calls on independent copied positions; it does not mutate
    the input. Convergence with displacement must be tested by the caller.

    :param energy: Callable mapping Cartesian positions in angstrom to total energy in eV.
    :param positions: Finite (atoms,3) Cartesian coordinates.
    :param forces: Corresponding Cartesian forces in eV/angstrom.
    :param displacement: Positive explicit central-difference step in angstrom.
    :return: Supplied-minus-numerical force component residuals, with atom/axis labels
        in eV/angstrom (no published force definition, so ``definition`` is ``None``).
    :raises ValueError: If geometry, step or callable results are invalid.
    """
    xyz, force = _array(positions, "positions"), _array(forces, "forces")
    step = _array(displacement, "displacement")
    if xyz.ndim != 2 or xyz.shape[1] != 3 or len(xyz) == 0 or force.shape != xyz.shape:
        raise ValueError("matching nonempty (atoms,3) positions and forces are required")
    if step.ndim != 0 or float(step) <= 0:
        raise ValueError("displacement must be a positive finite scalar")
    delta = float(step)
    numerical = np.empty_like(xyz)
    for i in range(len(xyz)):
        for j in range(3):
            plus, minus = xyz.copy(), xyz.copy()
            plus[i, j] += delta
            minus[i, j] -= delta
            if plus[i, j] == xyz[i, j] or minus[i, j] == xyz[i, j]:
                raise ValueError("displacement is not representable at the coordinate magnitude")
            ep, em = _array(energy(plus), "energy result"), _array(energy(minus), "energy result")
            if ep.ndim or em.ndim:
                raise ValueError("energy callable must return a scalar")
            numerical[i, j] = -(float(ep) - float(em)) / (plus[i, j] - minus[i, j])
    return property_parity(
        numerical.ravel(),
        force.ravel(),
        labels=[f"{i}:{axis}" for i in range(len(xyz)) for axis in "xyz"],
        definition=None,
    )


def committee_spread(predictions: Any, *, definition: str, ddof: int = 1) -> CommitteeSpread:
    """Summarize spread across models on a common sequence of scalar entries.

    Spread is a disagreement diagnostic, not calibrated uncertainty or a claim
    that a configuration lies inside the training distribution.

    :param predictions: Finite (models,entries) predictions with at least two models.
    :param definition: Property-definition IRI (not loaded here) fixing the common unit of the predictions.
    :param ddof: Nonnegative integer variance correction smaller than model count.
    :return: Per-entry mean and model standard deviation.
    :raises ValueError: If shape, definition, ddof or derived statistics are invalid.
    """
    values = _array(predictions, "predictions")
    _definition(definition)
    if values.ndim != 2 or values.shape[0] < 2 or values.shape[1] == 0:
        raise ValueError("predictions require (at least two models, nonempty entries)")
    if isinstance(ddof, bool) or not isinstance(ddof, int) or not 0 <= ddof < len(values):
        raise ValueError("ddof must be an integer from zero through model count minus one")
    mean, spread = values.mean(axis=0), values.std(axis=0, ddof=ddof)
    if not np.isfinite(mean).all() or not np.isfinite(spread).all():
        raise ValueError("committee statistics are not finite")
    return CommitteeSpread(tuple(mean), tuple(spread), len(values), ddof, definition)


def _definition(value: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError("a nonempty property-definition IRI is required")
