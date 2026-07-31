# httk-analyse

*httk-analyse* is a [*httk₂*](https://github.com/httk/httk2) module for analysis
algorithms: generic lower-convex-hull construction and materials-science phase
diagrams. Its Python package is `httk.analyse`.

## Usage

```python
from httk.analyse.generic import LowerConvexHull
from httk.analyse.matsci import PhaseDiagram
```

`LowerConvexHull` provides the generic geometric construction. `PhaseDiagram`
applies it to compositions and energies for materials-science phase-diagram
analysis. See [the example](examples/example.py) for a deterministic,
headless invocation.

The module depends on *httk-core*, *httk-atomistic*, NumPy, and Matplotlib.
