"""File-family analysis adapters validate synchronized physical metadata."""

from fractions import Fraction

import numpy as np
import pytest

from httk.atomistic import Trajectory, UnitcellStructure
from httk.analyse.integrations.trajectory import msd_from_trajectory, rdf_from_trajectory, vacf_from_trajectory


def _trajectory(*, times=(0, 1, 2), ids=((1, 2),) * 3, variable_cell=False):
    frames = [
        UnitcellStructure(
            np.eye(3) * (8 + i if variable_cell else 8),
            [[0, 0, 0], [Fraction(1, 8), 0, 0]],
            species_at_sites=['Ar', 'Ar'],
        )
        for i in range(3)
    ]
    return Trajectory(
        frames,
        observables={
            'time': times,
            'atom_ids': ids,
            'unwrapped_positions': tuple(((i, 0, 0), (i + 1, 0, 0)) for i in range(3)),
            'velocities': (((1, 0, 0), (1, 0, 0)),) * 3,
        },
    )


def test_rdf_and_dynamic_adapters_use_existing_trajectory_family():
    trajectory = _trajectory()
    rdf = rdf_from_trajectory(trajectory, [0, 0.5, 1.5, 3])
    assert rdf.frame_count == 3
    assert rdf.hist_counts == (0, 6, 0)
    assert msd_from_trajectory(trajectory).trace == (0, 1, 4)
    assert vacf_from_trajectory(trajectory).trace == (1, 1, 1)


@pytest.mark.parametrize(
    'kwargs',
    [
        {'times': (0, 1, 3)},
        {'times': (0, 1e-13, 0)},
        {'times': (0, 1e-13, 3e-13)},
        {'ids': ((1,),) * 3},
        {'ids': ((1, 1),) * 3},
        {'ids': ((1, 2), (2, 1), (1, 2))},
        {'variable_cell': True},
    ],
)
def test_dynamic_adapter_rejects_ambiguous_motion(kwargs):
    with pytest.raises(ValueError):
        msd_from_trajectory(_trajectory(**kwargs))


def test_no_inferred_unwrapping_or_time():
    original = _trajectory()
    trajectory = Trajectory(tuple(original.frames()))
    with pytest.raises(KeyError):
        msd_from_trajectory(trajectory)
