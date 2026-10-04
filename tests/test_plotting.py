"""Plot axes are labelled from property definitions."""

import pytest

pytest.importorskip("matplotlib")
import matplotlib

matplotlib.use("Agg")

from httk.analyse import definitions as defs
from httk.analyse.matsci import (
    harmonic_thermodynamics,
    mean_squared_displacement,
    property_parity,
    velocity_autocorrelation,
    viscosity,
)
from httk.analyse.plotting import plot_msd, plot_parity, plot_phonons, plot_transport


def test_axis_labels_come_from_definitions():
    _, axes = plot_phonons(harmonic_thermodynamics([2.0], [0.0, 300.0]))
    assert axes[0].get_ylabel() == "Helmholtz free energy (eV)"
    assert axes[-1].get_xlabel() == "Temperature (K)"
    stress = [[[0, s, 0], [s, 0, 0], [0, 0, 0]] for s in (1.0, -1.0, 0.5, 0.0)]
    _, axis = plot_transport(viscosity(stress, 0.1, temperature=100, volume=10, max_lag=2))
    assert axis.get_ylabel() == "Shear viscosity (Pa*s)"
    _, axis = plot_parity(property_parity([1.0, 2.0], [1.5, 2.5], labels=["a", "b"], definition=defs.BULK_MODULUS))
    assert axis.get_xlabel() == "Reference: Bulk modulus (GPa)"
    _, axis = plot_parity(property_parity([1.0, 2.0], [1.5, 2.5], labels=["a", "b"], definition=None))
    assert (axis.get_xlabel(), axis.get_ylabel()) == ("Reference", "Predicted")


def test_plot_msd_dispatches_on_kind():
    motion = [[[0.1 * t, 0.0, 0.0]] for t in range(5)]
    plot_msd(mean_squared_displacement(motion, 0.1))
    with pytest.raises(ValueError):
        plot_msd(velocity_autocorrelation(motion, 0.1))
