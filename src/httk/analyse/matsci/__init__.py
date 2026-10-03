"""Materials-science analysis built on generic numerical primitives."""

from .eos import BirchMurnaghanFit, fit_birch_murnaghan
from .phase_diagrams import PhaseDiagram, PhaseDiagramBuilder

__all__ = ["BirchMurnaghanFit", "PhaseDiagram", "PhaseDiagramBuilder", "fit_birch_murnaghan"]
