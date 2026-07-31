# Materials phase diagrams

`httk.analyse.matsci.PhaseDiagram` builds a lower convex hull in composition
space. It normalizes compositions and evaluates energies per atom, so entries
with differently sized formula units can be compared directly. Inputs may be
composition mappings or compatible `StructureLike` objects from
*httk-atomistic*.

## Build a binary diagram

Use `from_compositions` when the compositions are already known. The optional
IDs label entries in decompositions and plots:

```python
from httk.analyse.matsci import PhaseDiagram

diagram = PhaseDiagram.from_compositions(
    [{"A": 1}, {"B": 1}, {"A": 1, "B": 1}],
    [0.0, 0.0, -2.0],
    ids=["A", "B", "AB"],
)

assert tuple(diagram.hull_indices) == (0, 1, 2)
assert diagram.energy_above_hull[2] == 0.0
```

`energy_above_hull` is the per-atom energy above the stable lower envelope.
`hull_indices` retains the order of the supplied entries. Phase boundaries are
exposed directly as `phase_lines`, which delegates to the generic hull's
`supported_segments`.

## Plotting

Binary diagrams plot as a line diagram. Higher-dimensional composition spaces
plot their supported polygonal regions when possible. `plot()` returns a
Matplotlib axes object, making it straightforward to add labels or incorporate
the diagram into an existing figure:

```python
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ax = diagram.plot()
plt.close(ax.figure)
```

The plotting API is a presentation layer: use `energy_above_hull` and
`phase_lines` for programmatic analysis. For hulls over non-materials
coordinates, see {doc}`generic-hulls`.
