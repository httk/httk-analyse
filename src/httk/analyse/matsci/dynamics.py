"""Multiple-origin fixed-frame dynamics with explicit sampling conventions."""

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np

from .. import definitions as defs
from .._constants import M2_PER_S_PER_A2_PER_PS
from ..definitions import BoundValue, FieldBinding, _bind_fields, _plain, _reject_selection
from ..definitions import _series as _bind_series

__all__ = [
    "DiffusionFit",
    "RadialDynamics",
    "ScatteringSeries",
    "TensorSeries",
    "VelocitySpectrum",
    "diffusion_from_msd",
    "integrate_vacf",
    "intermediate_scattering",
    "mean_squared_displacement",
    "van_hove_distinct",
    "van_hove_self",
    "velocity_autocorrelation",
    "velocity_spectrum",
]


@dataclass(frozen=True, slots=True)
class TensorSeries:
    """Immutable lag times, tensor values and time-origin counts.

    Tensor units follow ``kind``: angstrom² for ``msd``, (angstrom/ps)² for
    ``vacf`` and angstrom²/ps for ``vacf_integral``.

    :param times: Lag times in ps.
    :param tensors: One 3x3 Cartesian tensor per lag.
    :param counts: Number of time origins contributing at each lag.
    :param kind: Estimator: ``msd``, ``vacf`` or ``vacf_integral``.
    :param remove_com: Whether center-of-mass motion was removed (MSD), else ``None``.
    :param remove_mean: Whether per-atom temporal means were removed (VACF and its integral), else ``None``.
    :raises ValueError: If ``kind`` is unknown.
    """

    times: tuple[float, ...]
    tensors: tuple[tuple[tuple[float, ...], ...], ...]
    counts: tuple[int, ...]
    kind: Literal["msd", "vacf", "vacf_integral"]
    remove_com: bool | None = None
    remove_mean: bool | None = None

    def __post_init__(self) -> None:
        """Copy nested numeric sequences to immutable tuples."""
        if self.kind not in ("msd", "vacf", "vacf_integral"):
            raise ValueError("kind must be 'msd', 'vacf' or 'vacf_integral'")
        object.__setattr__(self, "times", tuple(float(v) for v in self.times))
        object.__setattr__(self, "tensors", _tensors(self.tensors))
        object.__setattr__(self, "counts", tuple(int(v) for v in self.counts))

    @property
    def trace(self) -> tuple[float, ...]:
        """Return tensor traces in lag order."""
        return tuple(sum(row[i][i] for i in range(3)) for row in self.tensors)

    def _bound_values(self, **selection: Any) -> tuple[BoundValue, ...]:
        """Bind the series of ``kind``; the VACF integral is converted to m²/s and carries no origin counts."""
        _reject_selection(self, selection)
        times, tensors, counts = list(self.times), _plain(self.tensors), list(self.counts)
        if self.kind == "msd":
            return (_bind_series(defs.MEAN_SQUARED_DISPLACEMENT, lag_times=times, msd=tensors, origin_counts=counts),)
        if self.kind == "vacf":
            return (_bind_series(defs.VELOCITY_AUTOCORRELATION, lag_times=times, vacf=tensors, origin_counts=counts),)
        diffusion = (np.asarray(self.tensors) * M2_PER_S_PER_A2_PER_PS).tolist()
        return (_bind_series(defs.DIFFUSION_RUNNING_INTEGRAL, lag_times=times, diffusion_tensors=diffusion),)


@dataclass(frozen=True, slots=True)
class DiffusionFit:
    """A user-window linear MSD fit, without an inferred diffusive regime.

    :param tensor: Diffusion tensor in m²/s (the unit of the ``diffusion_tensor`` definition).
    :param intercept: MSD intercept tensor in angstrom² (a fit diagnostic of the MSD series).
    :param rmse: RMS residual per tensor component in angstrom² (MSD units).
    :param window: Inclusive start/stop lag indices.
    :param condition_number: Condition of the centered/scaled time design.
    """

    tensor: tuple[tuple[float, ...], ...]
    intercept: tuple[tuple[float, ...], ...]
    rmse: tuple[tuple[float, ...], ...]
    window: tuple[int, int]
    condition_number: float

    def __post_init__(self) -> None:
        """Copy fitted matrices and the selected window."""
        for name in ("tensor", "intercept", "rmse"):
            object.__setattr__(self, name, _tensors([getattr(self, name)])[0])
        object.__setattr__(self, "window", tuple(self.window))

    def _bound_values(self, **selection: Any) -> tuple[BoundValue, ...]:
        """Bind the diffusion tensor and its isotropic coefficient."""
        return _bind_fields(
            self,
            selection,
            {"tensor": FieldBinding(defs.DIFFUSION_TENSOR), "isotropic": FieldBinding(defs.DIFFUSION_COEFFICIENT)},
        )

    @property
    def isotropic(self) -> float:
        """Return one third of the diffusion tensor trace in m²/s."""
        return sum(self.tensor[i][i] for i in range(3)) / 3


@dataclass(frozen=True, slots=True)
class RadialDynamics:
    """Self probability or distinct number density in spherical shells.

    :param lag: Selected integer lag.
    :param time: Lag time in ps.
    :param edges: Radial shell edges in angstrom.
    :param density: Self probability or distinct neighbor density per angstrom³, without range renormalization.
    :param counts: Number of displacement samples in each shell.
    :param samples: Number of central-atom/time-origin samples, including those outside the bins.
    :param kind: Self or distinct radial correlation.
    """

    lag: int
    time: float
    edges: tuple[float, ...]
    density: tuple[float, ...]
    counts: tuple[int, ...]
    samples: int
    kind: Literal["self", "distinct"] = "self"

    def __post_init__(self) -> None:
        """Copy numeric shell fields into tuples."""
        for name in ("edges", "density"):
            object.__setattr__(self, name, tuple(float(v) for v in _array(getattr(self, name), name)))
        object.__setattr__(self, "counts", tuple(int(v) for v in self.counts))


@dataclass(frozen=True, slots=True)
class ScatteringSeries:
    """Complex intermediate scattering on explicitly supplied wavevectors.

    :param times: Lag times in ps.
    :param wavevectors: Cartesian wavevectors in inverse angstrom.
    :param values: Complex correlations indexed by lag then wavevector.
    :param counts: Time-origin counts per lag.
    :param kind: Self or coherent density correlation.
    """

    times: tuple[float, ...]
    wavevectors: tuple[tuple[float, ...], ...]
    values: tuple[tuple[complex, ...], ...]
    counts: tuple[int, ...]
    kind: Literal["self", "coherent"]

    def __post_init__(self) -> None:
        """Copy complex correlations and wavevectors into tuples."""
        object.__setattr__(self, "times", tuple(float(v) for v in self.times))
        object.__setattr__(self, "wavevectors", tuple(tuple(float(v) for v in row) for row in self.wavevectors))
        object.__setattr__(self, "values", tuple(tuple(complex(v) for v in row) for row in self.values))
        object.__setattr__(self, "counts", tuple(int(v) for v in self.counts))


@dataclass(frozen=True, slots=True)
class VelocitySpectrum:
    """One-sided velocity power spectral density averaged over atoms and axes.

    :param frequencies: Frequencies in THz (inverse ps).
    :param power: Power per THz, in (angstrom/ps)²/THz.
    :param window: Applied window.
    :param remove_mean: Whether each atom/component time mean was removed.
    :param frequency_spacing: Uniform frequency increment in THz.
    :param mean_square: Window-weighted mean-square velocity matching the discrete spectral integral.
    """

    frequencies: tuple[float, ...]
    power: tuple[float, ...]
    window: Literal["none", "hann"]
    remove_mean: bool
    frequency_spacing: float
    mean_square: float

    def __post_init__(self) -> None:
        """Copy spectral arrays into finite scalar tuples."""
        for name in ("frequencies", "power"):
            object.__setattr__(self, name, tuple(float(v) for v in _array(getattr(self, name), name)))


def mean_squared_displacement(
    unwrapped_positions: Any,
    timestep: float,
    *,
    max_lag: int | None = None,
    remove_com: bool = False,
    masses: Sequence[float] | None = None,
) -> TensorSeries:
    """Average displacement outer products over every time origin and atom.

    Positions must use persistent atom order and one fixed laboratory/material
    frame. Wrapped coordinates and changing-cell frame transformations must be
    resolved upstream. Masses are required for optional center-of-mass removal.

    :param unwrapped_positions: Finite Cartesian array (frames, atoms, 3), in angstrom.
    :param timestep: Uniform positive time between frames in ps.
    :param max_lag: Largest lag, inclusive; default frames minus one.
    :param remove_com: Subtract each frame's mass-weighted center before displacements.
    :param masses: Positive atom masses, required only when removing COM motion.
    :return: Symmetric MSD tensors in angstrom² on lag times in ps, with time-origin counts.
    :raises ValueError: If data, lag, masses or derived quantities are invalid.
    """
    positions, dt, lag = _trajectory(unwrapped_positions, timestep, max_lag)
    if remove_com:
        weight = _array(masses, "masses")
        if weight.shape != (positions.shape[1],) or np.any(weight <= 0):
            raise ValueError("COM removal requires one positive mass per atom")
        weight = weight / np.max(weight)
        weight /= np.sum(weight)
        positions -= np.einsum("n,tni->ti", weight, positions)[:, None, :]
    elif masses is not None:
        raise ValueError("masses require remove_com=True")
    values = []
    for k in range(lag + 1):
        delta = positions[k:] - positions[: len(positions) - k]
        values.append(np.einsum("tni,tnj->ij", delta, delta) / (delta.shape[0] * delta.shape[1]))
    return _series(values, len(positions), lag, dt, "msd", remove_com=bool(remove_com))


def velocity_autocorrelation(
    velocities: Any,
    timestep: float,
    *,
    max_lag: int | None = None,
    remove_mean: bool = False,
) -> TensorSeries:
    """Compute dimensionful velocity correlation tensors over all origins.

    C_ij(k)=mean[v_i(t)*v_j(t+k)] over origins and atoms. The lagged tensor need
    not be symmetric. This pair-count estimator is not a power spectral density.

    Removing each atom's own time mean also removes Δr/T, the diffusive signal. It biases the
    integral estimate of D low by about t_plateau/T and drives the full-length integral to zero,
    so use ``remove_mean=True`` only for non-diffusive (solid) systems.

    :param velocities: Array (frames, atoms, 3) in angstrom/ps.
    :param timestep: Uniform frame spacing in ps.
    :param max_lag: Largest inclusive lag.
    :param remove_mean: Subtract each atom/component time mean before correlation; only for non-diffusive systems.
    :return: Correlation tensors in (angstrom/ps)² on lag times in ps.
    :raises ValueError: If data, lag or results are invalid.
    """
    velocity, dt, lag = _trajectory(velocities, timestep, max_lag)
    if remove_mean:
        velocity -= velocity.mean(axis=0, keepdims=True)
    tensors = [
        np.einsum("tni,tnj->ij", velocity[: len(velocity) - k], velocity[k:])
        / ((len(velocity) - k) * velocity.shape[1])
        for k in range(lag + 1)
    ]
    return _series(tensors, len(velocity), lag, dt, "vacf", remove_mean=bool(remove_mean))


def diffusion_from_msd(msd: TensorSeries, *, window: tuple[int, int]) -> DiffusionFit:
    """Fit MSD_ij=2*D_ij*t+intercept on an explicit inclusive lag window.

    A linear fit alone cannot establish diffusive behavior. Inspect window
    sensitivity and residuals; negative noisy fitted eigenvalues are retained.

    :param msd: MSD tensor series in angstrom² and ps.
    :param window: Inclusive (start, stop) lag indices with at least three points.
    :return: Diffusion tensor, intercepts and component residual RMS.
    :raises ValueError: If the window, times or tensor data are invalid.
    """
    times = _array(msd.times, "times")
    tensors = _array(msd.tensors, "MSD")
    if len(window) != 2 or any(isinstance(i, bool) or not isinstance(i, int) for i in window):
        raise ValueError("window must be two integer lag indices")
    start, stop = window
    if start < 0 or stop >= len(times) or stop - start < 2 or tensors.shape != (len(times), 3, 3):
        raise ValueError("fit requires an in-range window containing at least three lags")
    t = times[start : stop + 1]
    if np.any(np.diff(t) <= 0):
        raise ValueError("lag times must increase strictly")
    center, scale = float(t.mean()), float(np.ptp(t))
    design = np.column_stack((np.ones(len(t)), (t - center) / scale))
    y = tensors[start : stop + 1].reshape(len(t), 9)
    coeff, _, rank, singular = np.linalg.lstsq(design, y, rcond=None)
    if rank != 2:
        raise ValueError("diffusion time design is rank deficient")
    slope = coeff[1] / scale
    intercept = coeff[0] - slope * center
    rms = np.sqrt(np.mean((y - design @ coeff) ** 2, axis=0))
    rows = _tensors(np.asarray((slope / 2 * M2_PER_S_PER_A2_PER_PS, intercept, rms)).reshape(3, 3, 3))
    return DiffusionFit(rows[0], rows[1], rows[2], (start, stop), float(singular[0] / singular[-1]))


def integrate_vacf(vacf: TensorSeries) -> TensorSeries:
    """Trapezoid-integrate a velocity correlation into a running diffusion tensor.

    The result is a running series kept in angstrom²/ps; the diffusion coefficient
    as a property (m²/s) is reported by :func:`diffusion_from_msd`.

    :param vacf: Dimensionful VACF in (angstrom/ps)² on increasing ps lags.
    :return: Running integral in angstrom²/ps, starting at zero.
    :raises ValueError: If lag times or tensor values are invalid.
    """
    times = _array(vacf.times, "times")
    values = _array(vacf.tensors, "VACF")
    if times.ndim != 1 or len(times) < 2 or times[0] != 0 or np.any(np.diff(times) <= 0):
        raise ValueError("VACF times must start at zero and increase")
    if values.shape != (len(times), 3, 3):
        raise ValueError("VACF tensors and times must match")
    integrated = np.zeros_like(values)
    integrated[1:] = np.cumsum(0.5 * (values[1:] + values[:-1]) * np.diff(times)[:, None, None], axis=0)
    return TensorSeries(tuple(times), _tensors(integrated), vacf.counts, "vacf_integral", remove_mean=vacf.remove_mean)


def van_hove_self(unwrapped_positions: Any, timestep: float, *, lag: int, bins: Sequence[float]) -> RadialDynamics:
    """Histogram self displacements without renormalizing a truncated radial range.

    :param unwrapped_positions: Persistent-atom Cartesian positions (frames, atoms, 3) in angstrom.
    :param timestep: Uniform frame spacing in ps.
    :param lag: Nonnegative integer lag.
    :param bins: Strictly increasing nonnegative spherical-shell edges in angstrom.
    :return: Probability per shell volume and raw sample counts.
    :raises ValueError: If inputs or shell edges are invalid.
    """
    positions, dt, _ = _trajectory(unwrapped_positions, timestep, lag)
    edges = _array(bins, "bins")
    if edges.ndim != 1 or len(edges) < 2 or edges[0] < 0 or np.any(np.diff(edges) <= 0):
        raise ValueError("bins must be increasing nonnegative radial edges")
    displacement = positions[lag:] - positions[: len(positions) - lag]
    radii = np.linalg.norm(displacement, axis=-1).ravel()
    counts, _ = np.histogram(radii, bins=edges)
    density = counts / (len(radii) * (4 * math.pi / 3) * np.diff(edges**3))
    _finite(density)
    return RadialDynamics(
        lag, lag * dt, tuple(edges), tuple(float(v) for v in density), tuple(int(v) for v in counts), len(radii)
    )


def van_hove_distinct(
    positions: Any,
    timestep: float,
    *,
    lag: int,
    bins: Sequence[float],
    cell: Any,
) -> RadialDynamics:
    """Compute the bulk distinct van Hove density with persistent atom identity.

    Counts displacements from atom i at each origin to atom j at the lag, with
    i != j. Minimum images use one fixed periodic 3D cell. The integral over a
    complete spatial domain is N-1; a finite spherical range is not renormalized.

    :param positions: Cartesian array (frames, atoms, 3) in angstrom, with persistent atom order.
    :param timestep: Uniform frame spacing in ps.
    :param lag: Nonnegative integer lag.
    :param bins: Increasing nonnegative radial edges, within the unique-image radius.
    :param cell: Fixed fully periodic 3x3 row-vector cell in angstrom.
    :return: Distinct neighbor density per angstrom³, directed counts and center-origin count.
    :raises ValueError: If data, cell or radial range is invalid.
    """
    from .structure import _cell, _check_unique_cutoff, _minimum_image_rows

    data, dt, _ = _trajectory(positions, timestep, lag)
    edges = _array(bins, "bins")
    if edges.ndim != 1 or len(edges) < 2 or edges[0] < 0 or np.any(np.diff(edges) <= 0):
        raise ValueError("bins must be increasing nonnegative radial edges")
    basis, inverse = _cell(cell)
    _check_unique_cutoff(float(edges[-1]), basis, inverse, (True, True, True))
    histogram = np.zeros(len(edges) - 1, dtype=np.int64)
    atom_count = data.shape[1]
    for origin in range(len(data) - lag):
        for atom in range(atom_count):
            displacements = data[origin + lag] - data[origin, atom]
            vectors = _minimum_image_rows(displacements, basis, inverse, (True, True, True))
            distances = np.linalg.norm(vectors, axis=1)
            histogram += np.histogram(distances[np.arange(atom_count) != atom], bins=edges)[0]
    samples = atom_count * (len(data) - lag)
    density = histogram / (samples * (4 * math.pi / 3) * np.diff(edges**3))
    _finite(density)
    return RadialDynamics(
        lag,
        lag * dt,
        tuple(edges),
        tuple(float(v) for v in density),
        tuple(int(v) for v in histogram),
        samples,
        "distinct",
    )


def intermediate_scattering(
    positions: Any,
    timestep: float,
    wavevectors: Any,
    *,
    kind: Literal["self", "coherent"],
    max_lag: int | None = None,
) -> ScatteringSeries:
    """Compute self or coherent scattering with exp(+i q·displacement).

    Self scattering averages over atoms and origins. Coherent scattering is
    mean[conj(rho(q,t))*rho(q,t+lag)]/N, retaining the density peak at q=0.
    For arbitrary q use unwrapped positions; periodic wrapped coordinates are
    phase equivalent only for wavevectors commensurate with the fixed cell.

    :param positions: Cartesian array (frames, atoms, 3) in angstrom.
    :param timestep: Uniform frame spacing in ps.
    :param wavevectors: Finite Cartesian (wavevectors, 3) array in inverse angstrom.
    :param kind: Explicit self or coherent definition.
    :param max_lag: Largest inclusive lag.
    :return: Complex correlations in lag and wavevector order.
    :raises ValueError: If shapes, kind or numeric values are invalid.
    """
    data, dt, lag = _trajectory(positions, timestep, max_lag)
    q = _array(wavevectors, "wavevectors")
    if q.ndim != 2 or q.shape[1] != 3 or not len(q) or kind not in ("self", "coherent"):
        raise ValueError("wavevectors require shape (Q,3), and kind must be self or coherent")
    phases = np.exp(1j * np.einsum("tni,qi->tnq", data, q))
    rho = phases.sum(axis=1)
    values = []
    for k in range(lag + 1):
        if kind == "self":
            corr = (phases[: len(data) - k].conj() * phases[k:]).mean(axis=(0, 1))
        else:
            corr = (rho[: len(data) - k].conj() * rho[k:]).mean(axis=0) / data.shape[1]
        if not np.isfinite(corr).all():
            raise ValueError("scattering correlation is not finite")
        values.append(tuple(complex(value) for value in corr))
    return ScatteringSeries(
        tuple(k * dt for k in range(lag + 1)),
        tuple(tuple(float(v) for v in row) for row in q),
        tuple(values),
        tuple(len(data) - k for k in range(lag + 1)),
        kind,
    )


def velocity_spectrum(
    velocities: Any,
    timestep: float,
    *,
    window: Literal["none", "hann"] = "hann",
    remove_mean: bool = False,
) -> VelocitySpectrum:
    """Compute a one-sided periodogram whose discrete integral obeys Parseval.

    This is a velocity power spectrum, not a normalized phonon density of states.
    It averages equally over atoms and Cartesian components. Multiplying power
    by frequency spacing and summing gives the window-weighted mean square.

    The default keeps each atom's time-averaged velocity. For a diffusing atom that mean is
    Δr/T, the diffusive signal: removing it zeroes the rectangular-window ω=0 bin, which is
    proportional to the diffusion coefficient. Per-atom mean removal is appropriate only for
    non-diffusive (solid) systems.

    :param velocities: Array (frames, atoms, 3) in angstrom/ps.
    :param timestep: Uniform positive frame spacing in ps.
    :param window: Rectangular (none) or Hann window.
    :param remove_mean: Remove each atom/component time mean before windowing; only for non-diffusive systems.
    :return: THz frequencies and dimensionful velocity power density.
    :raises ValueError: If input, window or spectral values are invalid.
    """
    data, dt, _ = _trajectory(velocities, timestep, 0)
    if window not in ("none", "hann") or (window == "hann" and len(data) < 3):
        raise ValueError("window must be none or hann; Hann requires at least three frames")
    if remove_mean:
        data -= data.mean(axis=0, keepdims=True)
    weight = np.ones(len(data)) if window == "none" else np.hanning(len(data))
    transformed = np.fft.rfft(data * weight[:, None, None], axis=0)
    power = dt / np.sum(weight**2) * np.mean(abs(transformed) ** 2, axis=(1, 2))
    if len(data) % 2 == 0:
        power[1:-1] *= 2
    else:
        power[1:] *= 2
    _finite(power)
    frequencies = np.fft.rfftfreq(len(data), d=dt)
    mean_square = float(np.mean(np.sum((data * weight[:, None, None]) ** 2, axis=0) / np.sum(weight**2)))
    return VelocitySpectrum(
        tuple(frequencies), tuple(float(v) for v in power), window, remove_mean, 1 / (len(data) * dt), mean_square
    )


def _array(values: Any, name: str) -> np.ndarray:
    try:
        raw = np.asarray(values)
        if np.iscomplexobj(raw) or raw.dtype.kind in "SU":
            raise ValueError(f"{name} must contain real numeric data")
        result = np.array(values, dtype=float, copy=True)
    except (TypeError, OverflowError) as exc:
        raise ValueError(f"{name} must contain finite real numbers") from exc
    if not np.isfinite(result).all():
        raise ValueError(f"{name} must contain finite real numbers")
    return result


def _finite(values: np.ndarray) -> None:
    if not np.isfinite(values).all():
        raise ValueError("derived dynamics values are not finite")


def _trajectory(values: Any, timestep: float, max_lag: int | None) -> tuple[np.ndarray, float, int]:
    data = _array(values, "trajectory")
    dt = _array(timestep, "timestep")
    if data.ndim != 3 or data.shape[-1] != 3 or data.shape[0] < 2 or data.shape[1] < 1:
        raise ValueError("trajectory requires shape (at least 2 frames, at least 1 atom,3)")
    if dt.ndim != 0 or float(dt) <= 0:
        raise ValueError("timestep must be a positive scalar")
    lag = len(data) - 1 if max_lag is None else max_lag
    if isinstance(lag, bool) or not isinstance(lag, int) or lag < 0 or lag >= len(data):
        raise ValueError("max_lag must be an integer between zero and frames minus one")
    return data, float(dt), lag


def _tensors(values: Any) -> tuple[tuple[tuple[float, ...], ...], ...]:
    data = _array(values, "tensors")
    if data.ndim != 3 or data.shape[1:] != (3, 3):
        raise ValueError("tensors must have shape (lags,3,3)")
    return tuple(tuple(tuple(float(v) for v in row) for row in tensor) for tensor in data)


def _series(
    values: Any, frames: int, lag: int, dt: float, kind: Literal["msd", "vacf"], **options: bool
) -> TensorSeries:
    return TensorSeries(
        tuple(k * dt for k in range(lag + 1)),
        _tensors(values),
        tuple(frames - k for k in range(lag + 1)),
        kind,
        **options,
    )
