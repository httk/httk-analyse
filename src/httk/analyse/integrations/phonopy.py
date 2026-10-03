"""Adapters from Phonopy objects to harmonic phonon thermodynamics."""

from collections.abc import Sequence
from typing import Any, Literal

import numpy as np

from httk.analyse.matsci.phonons import HarmonicThermodynamics, _matrix, _vector, harmonic_thermodynamics

__all__ = ["harmonic_from_phonopy"]


def harmonic_from_phonopy(
    phonon: Any,
    temperatures: Sequence[float],
    *,
    zero_modes: Literal["raise", "omit"] = "raise",
    imaginary: Literal["raise", "omit"] = "raise",
    cutoff_frequency: float = 0.0,
) -> HarmonicThermodynamics:
    """Calculate harmonic properties from a Phonopy mesh result.

    Uses the public ``mesh`` result. Q-point multiplicities are normalized to one;
    every branch is retained. Phonopy frequencies are interpreted as THz.

    :param phonon: A Phonopy object after mesh sampling has run.
    :param temperatures: Nonnegative temperatures in K.
    :param zero_modes: Explicit zero-mode policy.
    :param imaginary: Explicit negative-frequency policy.
    :param cutoff_frequency: Nonnegative THz zero-mode cutoff, as Phonopy's ``cutoff_frequency``.
    :return: Harmonic properties normalized per primitive-cell mode set.
    :raises ValueError: If the mesh is missing or malformed.
    """
    mesh = getattr(phonon, "mesh", None)
    frequencies = getattr(mesh, "frequencies", None)
    q_weights = getattr(mesh, "weights", None)
    if frequencies is None or q_weights is None:
        raise ValueError("phonon must expose mesh frequencies and q-point weights")
    freq, qweight = _matrix(frequencies, "phonopy mesh frequencies"), _vector(q_weights, "phonopy mesh weights")
    if freq.shape[0] != len(qweight) or np.any(qweight < 0) or not np.any(qweight > 0):
        raise ValueError("phonopy mesh weights must be nonnegative and match q-points")
    mode_weights = np.repeat(qweight / qweight.sum(), freq.shape[1])
    return harmonic_thermodynamics(
        tuple(map(float, freq.ravel())),
        temperatures,
        tuple(map(float, mode_weights)),
        zero_modes=zero_modes,
        imaginary=imaginary,
        cutoff_frequency=cutoff_frequency,
    )
