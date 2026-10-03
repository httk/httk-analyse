"""Materials-science analysis built on generic numerical primitives."""

from .eos import BirchMurnaghanFit, fit_birch_murnaghan
from .mlip import EnergyErrors, ErrorStatistics, ForceErrors, StressErrors, energy_errors, force_errors, stress_errors
from .phase_diagrams import PhaseDiagram, PhaseDiagramBuilder

__all__ = [
    "BirchMurnaghanFit",
    "EnergyErrors",
    "ErrorStatistics",
    "ForceErrors",
    "PhaseDiagram",
    "PhaseDiagramBuilder",
    "StressErrors",
    "energy_errors",
    "fit_birch_murnaghan",
    "force_errors",
    "stress_errors",
]
