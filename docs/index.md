# *httk-analyse*

This site documents specifically the *httk-analyse* module. For the full
documentation of *httk₂* as a whole, see [docs.httk.org](https://docs.httk.org).

*httk-analyse* provides numerical and materials-science analysis for *httk₂*:
convex hulls, phase diagrams, static equations of state, and scalar time-series
statistics. Analysis explicitly computes approximate float64 results while
preserving its inputs.

```{admonition} Quick links
:class: tip

- **API reference**: {doc}`reference/index`
- **Generic lower hulls**: {doc}`generic-hulls`
- **Materials phase diagrams**: {doc}`phase-diagrams`
- **Equation-of-state fitting**: {doc}`equations-of-state`
- **Simulation time series**: {doc}`time-series`
- **MLIP validation**: {doc}`mlip-validation`
- **CrysViz structure viewer**: {doc}`crysviz`
- **Examples notebook**: {doc}`notebooks/examples`
````

## Install

Preferably work in a Python virtual environment, then do:
```bash
git clone https://github.com/httk/httk-analyse
cd httk-analyse
python -m pip install -e .
```

## Usage example

```python
from httk.analyse.generic import LowerConvexHull

hull = LowerConvexHull([(0.0,), (0.5,), (1.0,)], [0.0, -1.0, 0.0])
assert tuple(hull.hull_indices) == (0, 1, 2)
```

```{toctree}
:maxdepth: 2
:caption: Documentation

generic-hulls
phase-diagrams
equations-of-state
energetics
time-series
mlip-validation
crysviz
reference/index
notebooks/examples
notebooks/materials-toolbox
```
