"""Tests for the deliberate public import surface."""

import importlib
import pkgutil
import subprocess
import sys

from httk import analyse
from httk.analyse import generic, matsci
from httk.analyse.generic import LowerConvexHull
from httk.analyse.matsci import PhaseDiagram, PhaseDiagramBuilder


def test_root_exposes_only_analysis_submodules() -> None:
    assert analyse.generic is generic
    assert analyse.matsci is matsci
    assert not hasattr(analyse, "LowerConvexHull")
    assert not hasattr(analyse, "PhaseDiagram")


def test_crysviz_submodule_is_imported_without_optional_dependency() -> None:
    subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import sys; import httk.analyse; assert httk.analyse.crysviz.__name__ == 'httk.analyse.crysviz'; "
                "assert 'crysviz' not in sys.modules"
            ),
        ],
        check=True,
    )


def test_explicit_simplex_does_not_import_highspy() -> None:
    subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import sys; from httk.analyse.generic import LowerConvexHull; "
                "h = LowerConvexHull([(0.0,), (1.0,)], [0.0, 0.0], solver='simplex'); "
                "assert h.supported_segments == ((0, 1),); "
                "assert h.solver == 'simplex'; assert 'highspy' not in sys.modules"
            ),
        ],
        check=True,
    )


def test_submodules_export_their_canonical_classes() -> None:
    assert generic.LowerConvexHull is LowerConvexHull
    assert matsci.PhaseDiagram is PhaseDiagram
    assert matsci.PhaseDiagramBuilder is PhaseDiagramBuilder


def test_toolbox_exports_use_canonical_defining_modules() -> None:
    from httk.analyse.generic.timeseries import BlockAverage, autocorrelation, block_average
    from httk.analyse.matsci.eos import BirchMurnaghanFit, fit_birch_murnaghan

    assert generic.BlockAverage is BlockAverage
    assert generic.autocorrelation is autocorrelation
    assert generic.block_average is block_average
    assert matsci.BirchMurnaghanFit is BirchMurnaghanFit
    assert matsci.fit_birch_murnaghan is fit_birch_murnaghan
    assert not hasattr(analyse, "fit_birch_murnaghan")
    assert not hasattr(analyse, "block_average")


def test_export_policy_is_pinned() -> None:
    from httk.analyse import integrations

    assert analyse.__all__ == ["crysviz", "generic", "integrations", "matsci", "plotting", "summary"]
    assert integrations.__all__ == ["phonopy", "trajectory", "vasp"]
    expected = {}
    for info in pkgutil.iter_modules(matsci.__path__):
        if info.name.startswith("_"):
            continue
        module = importlib.import_module(f"httk.analyse.matsci.{info.name}")
        for name in module.__all__:
            assert name not in expected
            expected[name] = getattr(module, name)
    assert matsci.__all__ == sorted(expected)
    assert all(getattr(matsci, name) is obj for name, obj in expected.items())
