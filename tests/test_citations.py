"""Tests for import-tracked citation credits."""

import subprocess
import sys
from textwrap import dedent


def test_analysis_and_plotting_citations() -> None:
    subprocess.run(
        [
            sys.executable,
            "-c",
            dedent(
                """\
                import httk.analyse
                import httk.core

                assert "Numerical analysis uses NumPy" in httk.core.credits.entries()
                assert "Phase-diagram plotting uses Matplotlib" not in httk.core.credits.entries()

                import matplotlib

                matplotlib.use("Agg")
                diagram = httk.analyse.matsci.PhaseDiagram.from_compositions(
                    [{"A": 1}, {"B": 1}, {"A": 1, "B": 1}, {"A": 1, "B": 3}],
                    [0.0, 0.0, -2.0, -1.0],
                )
                diagram.plot()
                assert "Phase-diagram plotting uses Matplotlib" in httk.core.credits.entries()
                """
            ),
        ],
        check=True,
    )
