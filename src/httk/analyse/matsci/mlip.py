"""Residual metrics for machine-learning interatomic potentials."""

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
from httk.core.definition_ids import ATOMIC_FORCE, STRESS_TENSOR

from .. import definitions as defs
from ..definitions import BoundValue, FieldBinding, _reject_selection

__all__ = [
    "EnergyErrors",
    "ErrorStatistics",
    "ForceErrors",
    "StressErrors",
    "energy_errors",
    "force_errors",
    "stress_errors",
]

_STATISTIC_DERIVATIONS = (
    ("rmse", defs.RMSE),
    ("mae", defs.MAE),
    ("bias", defs.BIAS),
    ("maximum_absolute_error", defs.MAXIMUM_ABSOLUTE_ERROR),
)
_STATISTIC_MEMBERS = ("bias", "mae", "rmse", "maximum_absolute_error", "percentile95_absolute_error")
_STRESS_COMPONENTS = ((0, 0), (1, 1), (2, 2), (1, 2), (0, 2), (0, 1))


def _energy_summary(
    field: str,
    weighting: str,
    offset: float | None,
    statistics: "ErrorStatistics",
    residuals: tuple[float, ...],
) -> BoundValue:
    """Build one ``energy_prediction_errors`` dictionary value with exact member types."""
    value: dict[str, Any] = {
        "weighting": str(weighting),
        "offset_per_atom": None if offset is None else float(offset),
        "count": int(statistics.count),
    }
    value.update({name: float(getattr(statistics, name)) for name in _STATISTIC_MEMBERS})
    value["residuals"] = [float(r) for r in residuals]
    return BoundValue(field, FieldBinding(defs.ENERGY_PREDICTION_ERRORS), value)


@dataclass(frozen=True, slots=True)
class ErrorStatistics:
    """Summarize a set of prediction-minus-reference residuals.

    :param count: Number of residual samples before weighting.
    :param bias: Weighted mean signed residual.
    :param mae: Weighted mean absolute residual.
    :param rmse: Weighted root mean square residual.
    :param maximum_absolute_error: Largest absolute residual.
    :param percentile95_absolute_error: Unweighted 95th percentile absolute residual using NumPy's linear interpolation.
    """

    count: int
    bias: float
    mae: float
    rmse: float
    maximum_absolute_error: float
    percentile95_absolute_error: float


@dataclass(frozen=True, slots=True)
class EnergyErrors:
    """Hold raw and optional offset-corrected per-atom energy errors.

    :param residuals: Raw predicted-minus-reference residuals in eV/atom, in configuration order.
    :param statistics: Raw energy error statistics.
    :param offset_per_atom: Explicit offset subtracted from predicted energies, or ``None`` when no correction was requested.
    :param corrected_residuals: Corrected residuals in eV/atom, or ``None`` when no correction was requested.
    :param corrected_statistics: Corrected energy error statistics, or ``None`` when no correction was requested.
    :param weighting: Weighting used for mean and RMS statistics.
    """

    residuals: tuple[float, ...]
    statistics: ErrorStatistics
    offset_per_atom: float | None
    corrected_residuals: tuple[float, ...] | None
    corrected_statistics: ErrorStatistics | None
    weighting: Literal["configuration", "atom"]

    def __post_init__(self) -> None:
        """Copy residual sequences into immutable tuples."""
        object.__setattr__(self, "residuals", tuple(float(value) for value in self.residuals))
        if self.corrected_residuals is not None:
            object.__setattr__(self, "corrected_residuals", tuple(float(value) for value in self.corrected_residuals))

    def _bound_values(self, **selection: Any) -> tuple[BoundValue, ...]:
        """Bind the raw statistics as ``total_energy_per_atom`` derivations (configuration weighting only), plus summaries.

        A statistic's population is part of its meaning and derivation terms do not carry weighting, so
        other weightings are bound only through the ``energy_prediction_errors`` dictionary value (always
        bound; raw comparison). When an offset was applied the corrected comparison is also bound, as
        ``corrected_energy_prediction_errors``.
        """
        _reject_selection(self, selection)
        out: list[BoundValue] = []
        if self.weighting == "configuration":
            binding = defs.TOTAL_ENERGY_PER_ATOM
            out.extend(
                BoundValue(
                    f"statistics.{name}", FieldBinding(binding, derivation), float(getattr(self.statistics, name))
                )
                for name, derivation in _STATISTIC_DERIVATIONS
            )
        out.append(_energy_summary("energy_prediction_errors", self.weighting, None, self.statistics, self.residuals))
        if self.corrected_statistics is not None and self.corrected_residuals is not None:
            out.append(
                _energy_summary(
                    "corrected_energy_prediction_errors",
                    self.weighting,
                    self.offset_per_atom,
                    self.corrected_statistics,
                    self.corrected_residuals,
                )
            )
        return tuple(out)


@dataclass(frozen=True, slots=True)
class ForceErrors:
    """Hold force residuals and aggregate, configuration, and species metrics.

    :param residuals: Per-atom ``(x, y, z)`` residual tuples in configuration and atom order, in eV/angstrom.
    :param weighting: Weighting policy used by aggregate and grouped summaries.
    :param component_statistics: Aggregate residual statistics in x, y, z order.
    :param mean_vector_error: Weighted mean Euclidean norm of atom force residuals.
    :param rms_vector_error: Weighted root mean square Euclidean norm of atom force residuals.
    :param per_configuration_component_statistics: Component statistics for each configuration.
    :param per_configuration_mean_vector_error: Mean vector error for each configuration.
    :param per_configuration_rms_vector_error: RMS vector error for each configuration.
    :param per_species_component_statistics: Species labels and their conditional component statistics, sorted by label.
    :param species: Species labels in configuration and atom order, matching residuals.
    """

    residuals: tuple[tuple[tuple[float, float, float], ...], ...]
    weighting: Literal["atom", "configuration"]
    component_statistics: tuple[ErrorStatistics, ErrorStatistics, ErrorStatistics]
    mean_vector_error: float
    rms_vector_error: float
    per_configuration_component_statistics: tuple[tuple[ErrorStatistics, ErrorStatistics, ErrorStatistics], ...]
    per_configuration_mean_vector_error: tuple[float, ...]
    per_configuration_rms_vector_error: tuple[float, ...]
    per_species_component_statistics: tuple[tuple[str, tuple[ErrorStatistics, ErrorStatistics, ErrorStatistics]], ...]
    species: tuple[tuple[str, ...], ...]

    def __post_init__(self) -> None:
        """Copy all nested result sequences into immutable tuples."""
        object.__setattr__(
            self,
            "residuals",
            tuple(tuple(tuple(float(value) for value in row) for row in config) for config in self.residuals),
        )
        object.__setattr__(self, "component_statistics", tuple(self.component_statistics))
        object.__setattr__(self, "species", tuple(tuple(row) for row in self.species))
        object.__setattr__(
            self,
            "per_configuration_component_statistics",
            tuple(tuple(stats) for stats in self.per_configuration_component_statistics),
        )
        object.__setattr__(self, "per_configuration_mean_vector_error", tuple(self.per_configuration_mean_vector_error))
        object.__setattr__(self, "per_configuration_rms_vector_error", tuple(self.per_configuration_rms_vector_error))
        object.__setattr__(
            self,
            "per_species_component_statistics",
            tuple((label, tuple(stats)) for label, stats in self.per_species_component_statistics),
        )

    def _bound_values(self, **selection: Any) -> tuple[BoundValue, ...]:
        """Bind ``force_prediction_errors`` always, plus ``atomic_force`` derivations for atom weighting.

        The derivation values are the (x, y, z) vectors of the aggregate component statistics; other
        weightings yield only the dictionary value, which carries ``weighting`` as a member.
        """
        _reject_selection(self, selection)
        out: list[BoundValue] = []
        if self.weighting == "atom":
            out.extend(
                BoundValue(
                    f"component_statistics.{name}",
                    FieldBinding(ATOMIC_FORCE, derivation),
                    [float(getattr(stats, name)) for stats in self.component_statistics],
                )
                for name, derivation in _STATISTIC_DERIVATIONS
            )
        species = sorted(self.per_species_component_statistics, key=lambda item: item[0])
        value: dict[str, Any] = {
            "weighting": str(self.weighting),
            "count": int(self.component_statistics[0].count),
        }
        for name in _STATISTIC_MEMBERS:
            value[f"component_{name}"] = [float(getattr(stats, name)) for stats in self.component_statistics]
        value["mean_vector_error"] = float(self.mean_vector_error)
        value["rms_vector_error"] = float(self.rms_vector_error)
        value["per_configuration_mean_vector_errors"] = [float(v) for v in self.per_configuration_mean_vector_error]
        value["per_configuration_rms_vector_errors"] = [float(v) for v in self.per_configuration_rms_vector_error]
        value["per_configuration_component_rmse"] = [
            [float(stats.rmse) for stats in config] for config in self.per_configuration_component_statistics
        ]
        value["species_labels"] = [str(label) for label, _ in species]
        value["per_species_count"] = [int(stats[0].count) for _, stats in species]
        for name in _STATISTIC_MEMBERS:
            value[f"per_species_component_{name}"] = [
                [float(getattr(stat, name)) for stat in stats] for _, stats in species
            ]
        out.append(BoundValue("force_prediction_errors", FieldBinding(defs.FORCE_PREDICTION_ERRORS), value))
        return tuple(out)


@dataclass(frozen=True, slots=True)
class StressErrors:
    """Hold stress residuals and aggregate and per-configuration metrics.

    :param residuals: Six component residual tuples per configuration in ``xx, yy, zz, yz, xz, xy`` order, in GPa.
    :param component_statistics: Aggregate component statistics in ``xx, yy, zz, yz, xz, xy`` order.
    :param per_configuration_component_statistics: Component statistics for each configuration.
    """

    residuals: tuple[tuple[float, float, float, float, float, float], ...]
    component_statistics: tuple[
        ErrorStatistics, ErrorStatistics, ErrorStatistics, ErrorStatistics, ErrorStatistics, ErrorStatistics
    ]
    per_configuration_component_statistics: tuple[
        tuple[ErrorStatistics, ErrorStatistics, ErrorStatistics, ErrorStatistics, ErrorStatistics, ErrorStatistics], ...
    ]

    def __post_init__(self) -> None:
        """Copy component sequences into immutable tuples."""
        object.__setattr__(self, "residuals", tuple(tuple(float(value) for value in row) for row in self.residuals))
        object.__setattr__(self, "component_statistics", tuple(self.component_statistics))
        object.__setattr__(
            self,
            "per_configuration_component_statistics",
            tuple(tuple(stats) for stats in self.per_configuration_component_statistics),
        )

    def _bound_values(self, **selection: Any) -> tuple[BoundValue, ...]:
        """Bind the Voigt-ordered ``stress_tensor`` derivations and the ``stress_prediction_errors`` dictionary value."""
        _reject_selection(self, selection)
        out = [
            BoundValue(
                f"component_statistics.{name}",
                FieldBinding(STRESS_TENSOR, derivation),
                [float(getattr(stats, name)) for stats in self.component_statistics],
            )
            for name, derivation in _STATISTIC_DERIVATIONS
        ]
        value: dict[str, Any] = {"count": int(self.component_statistics[0].count)}
        for name in _STATISTIC_MEMBERS:
            value[f"component_{name}"] = [float(getattr(stats, name)) for stats in self.component_statistics]
        value["residuals"] = [[float(v) for v in row] for row in self.residuals]
        out.append(BoundValue("stress_prediction_errors", FieldBinding(defs.STRESS_PREDICTION_ERRORS), value))
        return tuple(out)


def energy_errors(
    reference: Sequence[float],
    predicted: Sequence[float],
    *,
    atom_counts: Sequence[int],
    offset_per_atom: float | None = None,
    weighting: Literal["configuration", "atom"] = "configuration",
) -> EnergyErrors:
    """Compare total energies and report per-atom prediction residuals.

    Raw residuals are always reported. An explicit offset is subtracted from
    each predicted energy per atom; no energy shift is inferred from the data.
    Mean and RMS statistics weight each configuration equally by default.
    Choose ``weighting="atom"`` to weight each atom equally instead. The
    reported 95th percentile is unweighted under either policy.

    :param reference: Reference total energies in eV.
    :param predicted: Predicted total energies in eV.
    :param atom_counts: Positive integer atom count for each configuration.
    :param offset_per_atom: Optional explicit energy offset in eV/atom to subtract from each prediction.
    :param weighting: Weight configurations equally or weight each atom equally in means and RMS.
    :return: Immutable raw and optionally corrected per-atom residual summaries.
    :raises ValueError: If inputs have incompatible shapes, invalid counts, non-finite values, or unrepresentable residuals.
    """
    refs = _real_array(reference, "reference", ndim=1)
    preds = _real_array(predicted, "predicted", ndim=1)
    counts = _atom_counts(atom_counts)
    if refs.size == 0 or refs.shape != preds.shape or refs.size != len(counts):
        raise ValueError("reference, predicted, and atom_counts must have the same non-zero length")
    if weighting not in ("configuration", "atom"):
        raise ValueError("weighting must be 'configuration' or 'atom'")
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        differences = preds - refs
        if np.isfinite(differences).all():
            residuals = differences / counts
        else:
            residuals = preds / counts - refs / counts
    if not np.isfinite(residuals).all():
        raise ValueError("per-atom energy residuals are not finite")
    weights = counts.astype(np.float64) if weighting == "atom" else np.ones(len(counts), dtype=np.float64)
    raw = tuple(float(value) for value in residuals)
    corrected: tuple[float, ...] | None = None
    corrected_statistics: ErrorStatistics | None = None
    offset: float | None = None
    if offset_per_atom is not None:
        offset = _real_scalar(offset_per_atom, "offset_per_atom")
        with np.errstate(over="ignore", invalid="ignore"):
            values = residuals - offset
        if not np.isfinite(values).all():
            raise ValueError("corrected per-atom energy residuals are not finite")
        corrected = tuple(float(value) for value in values)
        corrected_statistics = _statistics(values, weights)
    return EnergyErrors(raw, _statistics(residuals, weights), offset, corrected, corrected_statistics, weighting)


def force_errors(
    reference: Sequence[Sequence[Sequence[float]]],
    predicted: Sequence[Sequence[Sequence[float]]],
    *,
    species: Sequence[Sequence[str]],
    weighting: Literal["atom", "configuration"] = "atom",
) -> ForceErrors:
    """Compare paired ragged force arrays in eV/angstrom.

    Atom order and species labels must identify the same physical atoms in each
    reference/prediction pair. This array API cannot detect atom permutations or
    dataset leakage. ``atom`` weights every atom equally; ``configuration``
    gives each configuration equal total weight. Species summaries condition
    and renormalize those same weights on atoms of that species. The 95th
    percentile remains unweighted in every summary. The pooled per-component
    RMSE common in the literature equals ``rms_vector_error / sqrt(3)`` of the
    result with ``weighting="atom"``; ``rms_vector_error`` is the Euclidean
    vector RMSE and is not directly comparable to it.

    :param reference: One finite ``(atoms, 3)`` force array per configuration.
    :param predicted: Matching predicted force arrays in eV/angstrom.
    :param species: One chemical-symbol label per atom in matching atom order.
    :param weighting: Weight every atom equally or every configuration equally.
    :return: Immutable component, vector, configuration, and species metrics.
    :raises ValueError: If shapes, species labels, weighting, or finite numeric values are invalid.
    """
    refs = _force_configurations(reference, "reference")
    preds = _force_configurations(predicted, "predicted")
    labels = _species_labels(species)
    if not refs or len(refs) != len(preds) or len(refs) != len(labels):
        raise ValueError("reference, predicted, and species must contain the same non-zero number of configurations")
    if weighting not in ("atom", "configuration"):
        raise ValueError("weighting must be 'atom' or 'configuration'")
    deltas: list[np.ndarray] = []
    for index, (ref, pred, config_species) in enumerate(zip(refs, preds, labels, strict=True)):
        if ref.shape != pred.shape or ref.shape[0] != len(config_species):
            raise ValueError(f"force arrays and species must match in configuration {index}")
        with np.errstate(over="ignore", invalid="ignore"):
            delta = pred - ref
        if not np.isfinite(delta).all():
            raise ValueError(f"force residuals are not finite in configuration {index}")
        deltas.append(delta)

    residuals = tuple(tuple((float(row[0]), float(row[1]), float(row[2])) for row in config) for config in deltas)
    flat = np.concatenate(deltas, axis=0)
    weights = _force_weights(deltas, weighting)
    component_stats = (
        _statistics(flat[:, 0], weights),
        _statistics(flat[:, 1], weights),
        _statistics(flat[:, 2], weights),
    )
    norms = np.hypot(np.hypot(flat[:, 0], flat[:, 1]), flat[:, 2])
    _finite_result(norms, "force vector errors")
    mean_vector = _weighted_mean(norms, weights)
    rms_vector = _weighted_rms(norms, weights)
    per_config_stats: list[tuple[ErrorStatistics, ErrorStatistics, ErrorStatistics]] = []
    per_config_mean: list[float] = []
    per_config_rms: list[float] = []
    species_values: dict[str, list[np.ndarray]] = {}
    species_weights: dict[str, list[np.ndarray]] = {}
    for delta, config_species in zip(deltas, labels, strict=True):
        local_weights = (
            np.ones(len(delta), dtype=np.float64)
            if weighting == "atom"
            else np.full(len(delta), 1.0 / len(delta), dtype=np.float64)
        )
        per_config_stats.append(
            (
                _statistics(delta[:, 0], np.ones(len(delta))),
                _statistics(delta[:, 1], np.ones(len(delta))),
                _statistics(delta[:, 2], np.ones(len(delta))),
            )
        )
        local_norms = np.hypot(np.hypot(delta[:, 0], delta[:, 1]), delta[:, 2])
        _finite_result(local_norms, "force vector errors")
        local_mean_weights = np.full(len(delta), 1.0 / len(delta), dtype=np.float64)
        per_config_mean.append(_weighted_mean(local_norms, local_mean_weights))
        per_config_rms.append(_weighted_rms(local_norms, local_mean_weights))
        for label in set(config_species):
            mask = np.fromiter((value == label for value in config_species), dtype=bool, count=len(config_species))
            species_values.setdefault(label, []).append(delta[mask])
            species_weights.setdefault(label, []).append(local_weights[mask])
    per_species = tuple(
        (
            label,
            (
                _statistics(
                    np.concatenate([values[:, 0] for values in species_values[label]]),
                    np.concatenate(species_weights[label]),
                ),
                _statistics(
                    np.concatenate([values[:, 1] for values in species_values[label]]),
                    np.concatenate(species_weights[label]),
                ),
                _statistics(
                    np.concatenate([values[:, 2] for values in species_values[label]]),
                    np.concatenate(species_weights[label]),
                ),
            ),
        )
        for label in sorted(species_values)
    )
    return ForceErrors(
        residuals,
        weighting,
        component_stats,
        mean_vector,
        rms_vector,
        tuple(per_config_stats),
        tuple(per_config_mean),
        tuple(per_config_rms),
        per_species,
        labels,
    )


def stress_errors(
    reference: Sequence[Sequence[Sequence[float]]], predicted: Sequence[Sequence[Sequence[float]]]
) -> StressErrors:
    """Compare symmetric tensile-positive stress tensors in GPa.

    Tensor pairs are compared without symmetrizing. Components use the order
    ``xx, yy, zz, yz, xz, xy`` and take the upper-triangle value for each shear
    component. The symmetry check permits absolute or relative differences up
    to ``1e-12``; convert virials and source sign conventions before calling.
    Typical float32 model outputs and virial (``-sum(r (x) F)``) stresses
    carry asymmetry of order ``1e-8`` GPa and are rejected: inspect
    the asymmetry and symmetrize explicitly (``(S + S.T)/2``) first.
    Aggregate component metrics weight each configuration equally. The 95th
    percentile is unweighted in every summary.

    :param reference: Finite stress tensors with shape ``(configurations, 3, 3)``.
    :param predicted: Matching predicted stress tensors in GPa.
    :return: Immutable six-component residuals and statistics.
    :raises ValueError: If shapes, symmetry, or finite numeric values are invalid.
    """
    refs = _real_array(reference, "reference", ndim=3)
    preds = _real_array(predicted, "predicted", ndim=3)
    if refs.shape != preds.shape or refs.shape[0] == 0 or refs.shape[1:] != (3, 3):
        raise ValueError("reference and predicted must have matching shape (configurations, 3, 3)")
    for name, values in (("reference", refs), ("predicted", preds)):
        if not np.allclose(values, np.swapaxes(values, 1, 2), rtol=1e-12, atol=1e-12):
            raise ValueError(f"{name} stress tensors must be symmetric")
    with np.errstate(over="ignore", invalid="ignore"):
        delta = preds - refs
    if not np.isfinite(delta).all():
        raise ValueError("stress residuals are not finite")
    components = np.stack([delta[:, i, j] for i, j in _STRESS_COMPONENTS], axis=1)
    residuals = tuple(
        (float(row[0]), float(row[1]), float(row[2]), float(row[3]), float(row[4]), float(row[5])) for row in components
    )
    overall = (
        _statistics(components[:, 0], np.ones(len(components))),
        _statistics(components[:, 1], np.ones(len(components))),
        _statistics(components[:, 2], np.ones(len(components))),
        _statistics(components[:, 3], np.ones(len(components))),
        _statistics(components[:, 4], np.ones(len(components))),
        _statistics(components[:, 5], np.ones(len(components))),
    )
    per_config = tuple(
        (
            _statistics(components[index : index + 1, 0], np.ones(1)),
            _statistics(components[index : index + 1, 1], np.ones(1)),
            _statistics(components[index : index + 1, 2], np.ones(1)),
            _statistics(components[index : index + 1, 3], np.ones(1)),
            _statistics(components[index : index + 1, 4], np.ones(1)),
            _statistics(components[index : index + 1, 5], np.ones(1)),
        )
        for index in range(len(components))
    )
    return StressErrors(residuals, overall, per_config)


def _real_array(values: object, name: str, *, ndim: int) -> np.ndarray:
    """Copy a finite real input to a float64 array of the requested rank."""
    try:
        raw = np.asarray(values)
        if (
            np.iscomplexobj(raw)
            or raw.dtype.kind in "SU"
            or (
                raw.dtype.kind == "O"
                and any(isinstance(value, (str, bytes, complex, np.complexfloating)) for value in raw.flat)
            )
        ):
            raise ValueError(f"{name} must contain real numeric values")
        array = np.asarray(values, dtype=np.float64)
    except (TypeError, OverflowError) as exc:
        raise ValueError(f"{name} must contain finite real float64 values") from exc
    if array.ndim != ndim or not np.isfinite(array).all():
        raise ValueError(f"{name} must be a finite {ndim}-dimensional real array")
    return array.copy()


def _real_scalar(value: float, name: str) -> float:
    """Return one finite real scalar as a float."""
    if isinstance(value, (str, bytes, complex, np.complexfloating)):
        raise ValueError(f"{name} must be a finite real scalar")
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be a finite real scalar") from exc
    if not math.isfinite(result):
        raise ValueError(f"{name} must be a finite real scalar")
    return result


def _atom_counts(values: Sequence[int]) -> np.ndarray:
    """Copy and validate positive integer atom counts."""
    try:
        raw = tuple(values)
    except TypeError as exc:
        raise ValueError("atom_counts must be a sequence of positive integers") from exc
    if any(
        isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value <= 0 for value in raw
    ):
        raise ValueError("atom_counts must contain positive integers")
    try:
        counts = np.asarray(raw, dtype=np.float64)
    except (OverflowError, TypeError, ValueError) as exc:
        raise ValueError("atom_counts must be representable as positive float64 weights") from exc
    if not np.isfinite(counts).all():
        raise ValueError("atom_counts must be representable as positive float64 weights")
    return counts


def _force_configurations(values: Sequence[Sequence[Sequence[float]]], name: str) -> tuple[np.ndarray, ...]:
    """Copy non-empty ragged force configurations with three components."""
    try:
        rows = tuple(values)
    except TypeError as exc:
        raise ValueError(f"{name} must contain force configurations") from exc
    configurations = tuple(_real_array(row, f"{name}[{index}]", ndim=2) for index, row in enumerate(rows))
    if any(config.shape[0] == 0 or config.shape[1] != 3 for config in configurations):
        raise ValueError(f"{name} configurations must have shape (atoms, 3) with at least one atom")
    return configurations


def _species_labels(values: Sequence[Sequence[str]]) -> tuple[tuple[str, ...], ...]:
    """Copy non-empty chemical-symbol labels into nested tuples."""
    try:
        if isinstance(values, (str, bytes)) or any(isinstance(row, (str, bytes)) for row in values):
            raise ValueError("species must contain one sequence of labels per configuration")
        rows = tuple(tuple(row) for row in values)
    except TypeError as exc:
        raise ValueError("species must contain one sequence of labels per configuration") from exc
    if any(not row or any(not isinstance(label, str) or not label for label in row) for row in rows):
        raise ValueError("species labels must be non-empty strings")
    return rows


def _force_weights(configurations: Sequence[np.ndarray], weighting: str) -> np.ndarray:
    """Return flattened force-atom weights normalized to one."""
    if weighting == "atom":
        weights = np.ones(sum(len(config) for config in configurations), dtype=np.float64)
    else:
        weights = np.concatenate([np.full(len(config), 1.0 / len(config)) for config in configurations])
    return weights / np.sum(weights)


def _statistics(values: np.ndarray, weights: np.ndarray) -> ErrorStatistics:
    """Compute weighted mean and RMS plus unweighted absolute-error quantiles."""
    values = np.asarray(values, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    if (
        values.ndim != 1
        or not len(values)
        or values.shape != weights.shape
        or not np.isfinite(values).all()
        or not np.isfinite(weights).all()
        or np.any(weights < 0.0)
        or np.max(weights) <= 0.0
    ):
        raise ValueError("residual statistics require matching finite one-dimensional values and weights")
    scaled_weights = weights / np.max(weights)
    normalized_weights = scaled_weights / np.sum(scaled_weights)
    bias = _weighted_mean(values, normalized_weights)
    absolute = np.abs(values)
    mae = _weighted_mean(absolute, normalized_weights)
    rmse = _weighted_rms(values, normalized_weights)
    percentile = float(np.quantile(absolute, 0.95, method="linear"))
    maximum = float(np.max(absolute))
    _finite_result(np.asarray((bias, mae, rmse, percentile, maximum)), "error statistics")
    return ErrorStatistics(len(values), bias, mae, rmse, maximum, percentile)


def _weighted_mean(values: np.ndarray, weights: np.ndarray) -> float:
    """Compute a weighted mean after scaling to avoid needless overflow."""
    scale = float(np.max(np.abs(values)))
    if scale == 0.0:
        return 0.0
    result = float(np.sum((values / scale) * weights) * scale)
    _finite_result(np.asarray((result,)), "weighted mean")
    return result


def _weighted_rms(values: np.ndarray, weights: np.ndarray) -> float:
    """Compute a weighted RMS after scaling to avoid square overflow."""
    scale = float(np.max(np.abs(values)))
    if scale == 0.0:
        return 0.0
    result = float(math.sqrt(float(np.sum((values / scale) ** 2 * weights))) * scale)
    _finite_result(np.asarray((result,)), "weighted RMS")
    return result


def _finite_result(values: np.ndarray, name: str) -> None:
    """Reject a non-finite derived result."""
    if not np.isfinite(values).all():
        raise ValueError(f"{name} are not finite representable float64 values")
