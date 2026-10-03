"""Dimensionful Green–Kubo running integrals with explicit flux conventions."""

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np

from .dynamics import _array

__all__ = ["ReplicaTransport", "TransportResult", "replica_transport", "thermal_conductivity", "viscosity"]

_KB_EV = 1.380649e-23 / 1.602176634e-19


@dataclass(frozen=True, slots=True)
class TransportResult:
    """Running transport coefficients, without inferred plateau selection.

    :param times: Lag times in ps.
    :param components: Tensor/component names in row order.
    :param correlations: Dimensionful correlations before physical prefactors.
    :param integrals: Running transport coefficients in the declared unit.
    :param counts: Number of time origins at each lag.
    :param quantity: Thermal conductivity or shear viscosity.
    :param unit: W/(m*K) or Pa*s.
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
    unit: str
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


@dataclass(frozen=True, slots=True)
class ReplicaTransport:
    """Mean and standard error across independent equal-protocol replicas.

    :param times: Common lag times in ps.
    :param components: Common component names.
    :param mean: Replica-mean running coefficients.
    :param standard_error: Sample standard deviation divided by sqrt(replica count).
    :param replicas: Number of independent replicas.
    :param unit: Coefficient unit.
    """

    times: tuple[float, ...]
    components: tuple[str, ...]
    mean: tuple[tuple[float, ...], ...]
    standard_error: tuple[tuple[float, ...], ...]
    replicas: int
    unit: str

    def __post_init__(self) -> None:
        """Copy nested replica summaries into immutable tuples."""
        object.__setattr__(self, "times", tuple(float(v) for v in self.times))
        object.__setattr__(self, "components", tuple(self.components))
        for name in ("mean", "standard_error"):
            values = _array(getattr(self, name), name)
            object.__setattr__(self, name, tuple(tuple(float(v) for v in row) for row in values))


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

    Supply the EXTENSIVE microscopic heat current J in eV*angstrom/ps, not flux
    density J/V. The tensor is integral <J_i(0)J_j(t)>/(kB*T²*V). Results retain
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
    factor = 1.602176634e3 / (_KB_EV * t * t * v)
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
    equilibrium temperature. The isotropic result averages these three shear
    responses. It is not a complete anisotropic fourth-rank viscosity tensor.

    :param stresses: Symmetric tensile-positive (samples,3,3) stress in eV/angstrom³.
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
    data = np.stack((tensor[:, 0, 1], tensor[:, 0, 2], tensor[:, 1, 2]), axis=1)
    data, dt, t, v = _inputs(data, timestep, temperature, volume, max_lag)
    if remove_mean:
        data -= data.mean(axis=0)
    correlation = np.asarray([np.mean(data[: len(data) - lag] * data[lag:], axis=0) for lag in range(max_lag + 1)])
    # eV*ps/angstrom^3 -> Pa*s.
    factor = 0.1602176634 * v / (_KB_EV * t)
    return _result(correlation, len(data), dt, t, v, remove_mean, factor, "viscosity")


def replica_transport(results: Sequence[TransportResult]) -> ReplicaTransport:
    """Aggregate independent runs or caller-selected decorrelated blocks.

    Equal protocol means the same quantity, units, T, V, lag grid, components and
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
            for name in ("times", "components", "quantity", "unit", "temperature", "volume", "remove_mean")
        ):
            raise ValueError("replica protocols and lag grids must match")
    values = _array([result.integrals for result in results], "replica integrals")
    mean = values.mean(axis=0)
    error = values.std(axis=0, ddof=1) / math.sqrt(len(results))
    if not np.isfinite(mean).all() or not np.isfinite(error).all():
        raise ValueError("replica statistics are not finite")
    return ReplicaTransport(
        first.times, first.components, tuple(map(tuple, mean)), tuple(map(tuple, error)), len(results), first.unit
    )


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
    unit = "W/(m*K)" if quantity == "thermal_conductivity" else "Pa*s"
    return TransportResult(
        tuple(i * dt for i in range(len(correlation))),
        components,
        tuple(map(tuple, correlation)),
        tuple(map(tuple, running)),
        tuple(count - i for i in range(len(correlation))),
        quantity,
        unit,
        temp,
        volume,
        center,
    )
