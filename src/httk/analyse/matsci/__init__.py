"""Materials-science analysis built on generic numerical primitives."""

from .energetics import (
    ChemicalPotentialRegion,
    ConvergenceTable,
    FormationEnergy,
    chemical_potential_region,
    convergence_table,
    enthalpy,
    formation_energy,
    reaction_energy,
)
from .eos import BirchMurnaghanFit, fit_birch_murnaghan
from .eos_models import EOSFit, EOSModel, fit_eos
from .mlip import EnergyErrors, ErrorStatistics, ForceErrors, StressErrors, energy_errors, force_errors, stress_errors
from .phase_diagrams import PhaseDiagram, PhaseDiagramBuilder

__all__ = [
    "BirchMurnaghanFit",
    "ChemicalPotentialRegion",
    "ConvergenceTable",
    "EOSFit",
    "EOSModel",
    "EnergyErrors",
    "ErrorStatistics",
    "ForceErrors",
    "FormationEnergy",
    "PhaseDiagram",
    "PhaseDiagramBuilder",
    "StressErrors",
    "chemical_potential_region",
    "convergence_table",
    "energy_errors",
    "enthalpy",
    "fit_birch_murnaghan",
    "fit_eos",
    "force_errors",
    "formation_energy",
    "reaction_energy",
    "stress_errors",
]
