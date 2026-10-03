"""Focused checks for the Phonopy analysis adapter."""

import numpy as np
import pytest

from httk.analyse.integrations.phonopy import harmonic_from_phonopy


def test_phonopy_adapter_normalizes_q_weights_and_forwards_cutoff():
    class Mesh:
        frequencies = np.asarray([[1.0, 2.0, 3.0], [1.0, 2.0, 3.0]])
        weights = np.asarray([1, 3])

    result = harmonic_from_phonopy(type("Phonon", (), {"mesh": Mesh()})(), [0.0])
    assert result.retained_mode_weight == pytest.approx(3.0)
    assert result.cutoff_frequency == 0.0

    class NoisyMesh:
        frequencies = np.asarray([[1e-6, -1e-6, 1e-6], [1.0, 2.0, 3.0]])
        weights = np.asarray([1, 3])

    noisy = type("Phonon", (), {"mesh": NoisyMesh()})()
    with pytest.raises(ValueError, match="negative frequencies"):
        harmonic_from_phonopy(noisy, [300.0], zero_modes="omit")
    forwarded = harmonic_from_phonopy(noisy, [300.0], zero_modes="omit", cutoff_frequency=1e-3)
    assert forwarded.cutoff_frequency == 1e-3
    assert forwarded.excluded_zero_weight == pytest.approx(0.75)
    assert forwarded.retained_mode_weight == pytest.approx(2.25)
