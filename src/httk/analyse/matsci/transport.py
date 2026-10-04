"""Dimensionful Green–Kubo running integrals with explicit flux conventions."""

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
from httk.core import definition_ids

from .. import definitions as defs
from .._constants import GPA_PER_EV_PER_A3, KB_EV_PER_K
from ..definitions import BoundValue, FieldBinding
from .dynamics import _array

__all__ = ["ReplicaTransport", "TransportResult", "replica_transport", "thermal_conductivity", "viscosity"]


@dataclass(frozen=True, slots=True)
class TransportResult:
    """Running transport coefficients, without inferred plateau selection.

    :param times: Lag times in ps.
    :param components: Tensor/component names in row order.
    :param correlations: Dimensionful correlations before physical prefactors.
    :param integrals: Running transport coefficients in the unit of the ``quantity`` definition.
    :param counts: Number of time origins at each lag.
    :param quantity: ``thermal_conductivity`` (integrals in W/(m K)) or ``viscosity`` (shear viscosity, integrals in Pa s).
    :param temperature: Equilibrium temperature in K.
    :param volume: Fixed cell volume in angstrom³.
    :param remove_mean: Whether the time mean of each input component was removed.
    """

    times: tuple[float, ...]
    components: tuple[str, ...]
    correlations: tuple[tuple[float, ...], ...]
    integrals: tuple[tuple[float, ...], ...]
    counts: tuple[int, ...]
    quantity: Literal["thermal_conductivity", "viscosity"]
    temperature: float
    volume: float
    remove_mean: bool

    def __post_init__(self) -> None:
        """Copy numerical arrays and component labels into immutable tuples."""
        object.__setattr__(self, "times", tuple(float(v) for v in self.times))
        object.__setattr__(self, "components", tuple(self.components))
        object.__setattr__(self, "counts", tuple(int(v) for v in self.counts))
        for name in ("correlations", "integrals"):
            data = _array(getattr(self, name), name)
            object.__setattr__(self, name, tuple(tuple(float(v) for v in row) for row in data))

    @property
    def isotropic(self) -> tuple[float, ...]:
        """Return the mean diagonal conductivity or mean three shear viscosities."""
        indices = (0, 4, 8) if self.quantity == "thermal_conductivity" else (0, 1, 2)
        return tuple(sum(row[i] for i in indices) / 3 for row in self.integrals)

    def _bound_values(self, *, lag_index: int, **selection: Any) -> tuple[BoundValue, ...]:
        r"""Bind the coefficients at the caller-chosen plateau lag, plus temperature and volume.

        :param lag_index: Explicit plateau lag index into ``integrals``; no plateau is inferred.
        :param \*\*selection: Unsupported further keywords, rejected.
        :return: The bound values.
        :raises TypeError: If unsupported keywords are given.
        :raises ValueError: If ``lag_index`` is not a valid lag index.
        """
        return (
            *_lag_values("integrals", self.quantity, self.integrals, lag_index, None, selection),
            BoundValue("temperature", FieldBinding(definition_ids.TEMPERATURE), self.temperature),
            BoundValue("volume", FieldBinding(definition_ids.VOLUME), self.volume),
        )


@dataclass(frozen=True, slots=True)
class ReplicaTransport:
    """Mean and standard error across independent equal-protocol replicas.

    :param times: Common lag times in ps.
    :param components: Common component names.
    :param mean: Replica-mean running coefficients.
    :param standard_error: Sample standard deviation divided by sqrt(replica count).
    :param replicas: Number of independent replicas.
    """

    times: tuple[float, ...]
    components: tuple[str, ...]
    mean: tuple[tuple[float, ...], ...]
    standard_error: tuple[tuple[float, ...], ...]
    replicas: int

    def __post_init__(self) -> None:
        """Copy nested replica summaries into immutable tuples."""
        object.__setattr__(self, "times", tuple(float(v) for v in self.times))
        object.__setattr__(self, "components", tuple(self.components))
        for name in ("mean", "standard_error"):
            values = _array(getattr(self, name), name)
            object.__setattr__(self, name, tuple(tuple(float(v) for v in row) for row in values))

    def _bound_values(self, *, lag_index: int, **selection: Any) -> tuple[BoundValue, ...]:
        r"""Bind the replica mean and standard error at the caller-chosen plateau lag.

        The quantity follows from ``components`` (nine Cartesian pairs for
        thermal conductivity, three shear pairs for viscosity).

        :param lag_index: Explicit plateau lag index; no plateau is inferred.
        :param \*\*selection: Unsupported further keywords, rejected.
        :return: The bound values, all qualified by a derivation.
        :raises TypeError: If unsupported keywords are given.
        :raises ValueError: If ``lag_index`` is not a valid lag index.
        """
        quantity: Literal["thermal_conductivity", "viscosity"] = (
            "thermal_conductivity" if len(self.components) == 9 else "viscosity"
        )
        return (
            *_lag_values("mean", quantity, self.mean, lag_index, defs.MEAN, selection),
            *_lag_values("standard_error", quantity, self.standard_error, lag_index, defs.STANDARD_ERROR, selection),
        )


def thermal_conductivity(
    heat_current: Any,
    timestep: float,
    *,
    temperature: float,
    volume: float,
    max_lag: int,
    remove_mean: bool = True,
) -> TransportResult:
    """Integrate total heat-current correlations into a conductivity tensor.

    Supply the EXTENSIVE total heat current J of the whole cell in eV*angstrom/ps,
    not flux density J/V. LAMMPS ``compute heat/flux`` in ``metal`` units is
    directly usable; in ``real`` units (kcal/mol*angstrom/fs) multiply by
    43.3641 (0.0433641 eV per kcal/mol times 1000 fs/ps). The tensor is integral <J_i(0)J_j(t)>/(kB*T²*V). Results retain
    off-diagonal asymmetry of finite sampling. Equilibrium stationarity and the
    physical current definition must be established upstream, especially for
    many-body MLIPs; energies and velocities alone do not define that current.

    :param heat_current: Finite (samples,3) extensive heat-current vectors.
    :param timestep: Uniform sample spacing in ps.
    :param temperature: Positive equilibrium temperature in K.
    :param volume: Positive fixed cell volume in angstrom³.
    :param max_lag: Largest inclusive lag, smaller than the sample count.
    :param remove_mean: Subtract each current component's temporal mean.
    :return: Running conductivity tensor in W/(m*K), starting at zero.
    :raises ValueError: If input shapes, units metadata, lags or derived values are invalid.
    """
    data, dt, t, v = _inputs(heat_current, timestep, temperature, volume, max_lag)
    if remove_mean:
        data -= data.mean(axis=0)
    correlation = np.asarray([data[: len(data) - lag].T @ data[lag:] / (len(data) - lag) for lag in range(max_lag + 1)])
    correlation = correlation.reshape(max_lag + 1, 9)
    # eV/(angstrom*ps*K) -> J/(metre*second*K).
    factor = 1.602176634e3 / (KB_EV_PER_K * t * t * v)
    return _result(correlation, len(data), dt, t, v, remove_mean, factor, "thermal_conductivity")


def viscosity(
    stresses: Any,
    timestep: float,
    *,
    temperature: float,
    volume: float,
    max_lag: int,
    remove_mean: bool = True,
) -> TransportResult:
    """Integrate off-diagonal stress fluctuations into three shear viscosities.

    Uses xy, xz and yz auto-correlations times V/(kB*T), with a fixed volume and
    equilibrium temperature. The isotropic result averages only these three
    shear components; the five-component traceless (Daivis-Evans) estimator is
    not implemented, and the result is not a complete anisotropic fourth-rank
    viscosity tensor. Stress must be tensile-positive in GPa. The LAMMPS
    pressure tensor is compressive-positive and in bar: negate it and multiply by
    1e-4 to get GPa.

    :param stresses: Symmetric tensile-positive (samples,3,3) stress in GPa.
    :param timestep: Uniform sample spacing in ps.
    :param temperature: Positive equilibrium temperature in K.
    :param volume: Positive fixed cell volume in angstrom³.
    :param max_lag: Largest inclusive lag.
    :param remove_mean: Subtract each shear component's temporal mean.
    :return: Running xy, xz, yz shear viscosities in Pa*s.
    :raises ValueError: If shapes, symmetry, metadata, lags or results are invalid.
    """
    tensor = _array(stresses, "stresses")
    if (
        tensor.ndim != 3
        or tensor.shape[1:] != (3, 3)
        or not np.allclose(tensor, tensor.swapaxes(1, 2), rtol=1e-12, atol=1e-12)
    ):
        raise ValueError("stresses require symmetric (samples,3,3) tensors")
    tensor = tensor / GPA_PER_EV_PER_A3
    data = np.stack((tensor[:, 0, 1], tensor[:, 0, 2], tensor[:, 1, 2]), axis=1)
    data, dt, t, v = _inputs(data, timestep, temperature, volume, max_lag)
    if remove_mean:
        data -= data.mean(axis=0)
    correlation = np.asarray([np.mean(data[: len(data) - lag] * data[lag:], axis=0) for lag in range(max_lag + 1)])
    # eV*ps/angstrom^3 -> Pa*s.
    factor = 0.1602176634 * v / (KB_EV_PER_K * t)
    return _result(correlation, len(data), dt, t, v, remove_mean, factor, "viscosity")


def replica_transport(results: Sequence[TransportResult]) -> ReplicaTransport:
    """Aggregate independent runs or caller-selected decorrelated blocks.

    Equal protocol means the same quantity, T, V, lag grid, components and
    centering. Counts may differ. This estimates between-replica standard error;
    it does not prove replica independence or remove truncation/finite-size bias.
    To use blocks, run the estimator separately on each equal-length block with
    a lag smaller than the block and compare results across block lengths.

    :param results: At least two independent equal-protocol transport results.
    :return: Per-component means and standard errors at every lag.
    :raises ValueError: If fewer than two results or mismatched protocols are supplied.
    """
    if len(results) < 2:
        raise ValueError("at least two independent replicas are required")
    first = results[0]
    for result in results[1:]:
        if any(
            getattr(result, name) != getattr(first, name)
            for name in ("times", "components", "quantity", "temperature", "volume", "remove_mean")
        ):
            raise ValueError("replica protocols and lag grids must match")
    values = _array([result.integrals for result in results], "replica integrals")
    mean = values.mean(axis=0)
    error = values.std(axis=0, ddof=1) / math.sqrt(len(results))
    if not np.isfinite(mean).all() or not np.isfinite(error).all():
        raise ValueError("replica statistics are not finite")
    return ReplicaTransport(
        first.times, first.components, tuple(map(tuple, mean)), tuple(map(tuple, error)), len(results)
    )


def _lag_values(
    field: str,
    quantity: Literal["thermal_conductivity", "viscosity"],
    rows: tuple[tuple[float, ...], ...],
    lag_index: int,
    derivation: str | None,
    selection: dict[str, Any],
) -> tuple[BoundValue, ...]:
    if selection:
        raise TypeError(f"transport results take only lag_index, got {', '.join(sorted(selection))}")
    if isinstance(lag_index, bool) or not isinstance(lag_index, int) or not 0 <= lag_index < len(rows):
        raise ValueError(f"lag_index must be an integer from 0 through {len(rows) - 1}")
    row = rows[lag_index]
    name = f"{field}[{lag_index}]"
    isotropic_name = f"isotropic[{lag_index}]" if field == "integrals" else f"{field}.isotropic[{lag_index}]"
    bound = []
    if quantity == "thermal_conductivity":
        tensor = [list(row[i : i + 3]) for i in (0, 3, 6)]
        bound.append(BoundValue(name, FieldBinding(defs.THERMAL_CONDUCTIVITY_TENSOR, derivation), tensor))
    # Per-component standard errors do not determine the error of their average (covariances are not kept).
    if derivation != defs.STANDARD_ERROR:
        indices = (0, 4, 8) if quantity == "thermal_conductivity" else (0, 1, 2)
        definition = defs.THERMAL_CONDUCTIVITY if quantity == "thermal_conductivity" else defs.SHEAR_VISCOSITY
        bound.append(BoundValue(isotropic_name, FieldBinding(definition, derivation), sum(row[i] for i in indices) / 3))
    return tuple(bound)


def _inputs(
    values: Any, timestep: float, temperature: float, volume: float, lag: int
) -> tuple[np.ndarray, float, float, float]:
    data = _array(values, "current")
    if data.ndim != 2 or data.shape[1] != 3 or len(data) < 2:
        raise ValueError("current requires shape (at least two samples,3)")
    scalars = []
    for name, value in (("timestep", timestep), ("temperature", temperature), ("volume", volume)):
        scalar = _array(value, name)
        if scalar.ndim != 0 or float(scalar) <= 0:
            raise ValueError(f"{name} must be a positive finite scalar")
        scalars.append(float(scalar))
    if isinstance(lag, bool) or not isinstance(lag, int) or lag < 1 or lag >= len(data):
        raise ValueError("max_lag must be an integer from one through samples minus one")
    return data, scalars[0], scalars[1], scalars[2]


def _result(
    correlation: np.ndarray,
    count: int,
    dt: float,
    temp: float,
    volume: float,
    center: bool,
    factor: float,
    quantity: Literal["thermal_conductivity", "viscosity"],
) -> TransportResult:
    running = np.zeros_like(correlation)
    running[1:] = np.cumsum(0.5 * (correlation[1:] + correlation[:-1]) * dt * factor, axis=0)
    if not np.isfinite(correlation).all() or not np.isfinite(running).all():
        raise ValueError("transport correlation or running integral is not finite")
    components = (
        tuple(i + j for i in "xyz" for j in "xyz") if quantity == "thermal_conductivity" else ("xy", "xz", "yz")
    )
    return TransportResult(
        tuple(i * dt for i in range(len(correlation))),
        components,
        tuple(map(tuple, correlation)),
        tuple(map(tuple, running)),
        tuple(count - i for i in range(len(correlation))),
        quantity,
        temp,
        volume,
        center,
    )
