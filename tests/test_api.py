"""Tests for the deliberate public import surface."""

from httk import analyse
from httk.analyse import generic, matsci
from httk.analyse.generic import LowerConvexHull
from httk.analyse.matsci import PhaseDiagram, PhaseDiagramBuilder


def test_root_exposes_only_analysis_submodules() -> None:
    assert analyse.generic is generic
    assert analyse.matsci is matsci
    assert not hasattr(analyse, "LowerConvexHull")
    assert not hasattr(analyse, "PhaseDiagram")


def test_submodules_export_their_canonical_classes() -> None:
    assert generic.LowerConvexHull is LowerConvexHull
    assert matsci.PhaseDiagram is PhaseDiagram
    assert matsci.PhaseDiagramBuilder is PhaseDiagramBuilder
