"""Explicit approximate analyses of existing exact trajectory interfaces."""

from collections.abc import Sequence
from itertools import tee
from typing import Any

import numpy as np
from httk.atomistic.models.trajectory.api import TrajectoryAPI

from httk.analyse.matsci.dynamics import TensorSeries, mean_squared_displacement, velocity_autocorrelation
from httk.analyse.matsci.structure import RadialDistribution, radial_distribution

__all__ = ["msd_from_trajectory", "rdf_from_trajectory", "vacf_from_trajectory"]


def rdf_from_trajectory(
    trajectory: TrajectoryAPI,
    bins: Sequence[float],
    *,
    pair: tuple[str, str] | None = None,
) -> RadialDistribution:
    """Stream a bulk RDF with each frame's own cell and fixed species labels.

    :param trajectory: Exact trajectory whose structural lengths are angstrom.
    :param bins: Radial bin edges in angstrom.
    :param pair: Optional ordered species-name pair for partial RDF.
    :return: Radial distribution using an explicit float approximation of structures.
    :raises ValueError: If geometry, periodicity or labels violate bulk RDF requirements.
    """
    source = trajectory.frames()

    def checked_frames() -> Any:
        for frame in source:
            if tuple(frame.cell.periodicity) != (True, True, True):
                raise ValueError("bulk trajectory RDF requires three periodic directions")
            yield frame

    positions, cells = tee(checked_frames())
    try:
        return radial_distribution(
            (frame.cartesian_sites().to_floats() for frame in positions),
            (frame.cell.basis.to_floats() for frame in cells),
            bins,
            species=trajectory.species_at_sites if pair is not None else None,
            pair=pair,
        )
    finally:
        close = getattr(source, "close", None)
        if close is not None:
            close()


def msd_from_trajectory(
    trajectory: TrajectoryAPI,
    *,
    max_lag: int | None = None,
    remove_com: bool = False,
    masses: Sequence[float] | None = None,
) -> TensorSeries:
    """Materialize known unwrapped angstrom positions and compute a fixed-cell MSD.

    Requires ``time`` (ps), ``atom_ids`` and ``unwrapped_positions`` canonical
    observables. Never unwrap sparse wrapped positions or infer a timestep.
    Memory is O(frames*atoms); select source segments before calling.

    :param trajectory: Trajectory with canonical synchronized observables.
    :param max_lag: Largest inclusive lag.
    :param remove_com: Remove mass-weighted center-of-mass motion.
    :param masses: Positive masses required for COM removal.
    :return: Multiple-origin MSD tensors in angstrom².
    :raises ValueError: If atom order, cell or time spacing changes.
    :raises KeyError: If required observables are absent.
    """
    positions, timestep = _dynamic_data(trajectory, "unwrapped_positions")
    return mean_squared_displacement(positions, timestep, max_lag=max_lag, remove_com=remove_com, masses=masses)


def vacf_from_trajectory(
    trajectory: TrajectoryAPI,
    *,
    max_lag: int | None = None,
    remove_mean: bool = False,
) -> TensorSeries:
    """Materialize canonical velocities and compute a fixed-cell VACF.

    :param trajectory: Trajectory with ps time, atom_ids and angstrom/ps velocities.
    :param max_lag: Largest inclusive lag.
    :param remove_mean: Remove each atom/component time mean.
    :return: Velocity correlations in (angstrom/ps)².
    :raises ValueError: If atom order, cell or time spacing changes.
    :raises KeyError: If required observables are absent.
    """
    velocities, timestep = _dynamic_data(trajectory, "velocities")
    return velocity_autocorrelation(velocities, timestep, max_lag=max_lag, remove_mean=remove_mean)


def _dynamic_data(trajectory: TrajectoryAPI, observable: str) -> tuple[list[Any], float]:
    samples = trajectory.samples("time", "atom_ids", observable)
    data: list[Any] = []
    times: list[float] = []
    identity = None
    first_cell = None
    first_periodicity = None
    try:
        for frame, values in samples:
            time, ids, quantities = values
            current = tuple(ids)
            atom_count = len(frame.species_at_sites)
            if (
                len(current) != atom_count
                or any(
                    isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) for value in current
                )
                or len(set(current)) != atom_count
            ):
                raise ValueError("atom_ids must contain one unique integer ID per frame atom")
            if np.shape(quantities) != (atom_count, 3):
                raise ValueError("dynamic observable must contain one Cartesian vector per frame atom")
            if identity is None:
                identity = current
                first_cell = frame.cell.basis
                first_periodicity = frame.cell.periodicity
            if current != identity:
                raise ValueError("persistent atom order changes within trajectory")
            if frame.cell.basis != first_cell or frame.cell.periodicity != first_periodicity:
                raise ValueError("dynamics adapter requires a fixed cell and coordinate frame")
            times.append(float(time))
            data.append(quantities)
    finally:
        close = getattr(samples, "close", None)
        if close is not None:
            close()
    if len(times) < 2 or not np.isfinite(times).all():
        raise ValueError("dynamics requires at least two finite frame times")
    intervals = np.diff(times)
    if not np.all(intervals > 0) or not np.allclose(intervals, intervals[0], rtol=1e-10, atol=0):
        raise ValueError("dynamics requires uniformly increasing time in ps")
    return data, float(intervals[0])
