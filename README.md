# httk-analyse

![Status: Early beta](https://img.shields.io/badge/status-early--beta-orange)

> **⚠️ EARLY BETA**
>
> This is an early beta release of *httk₂*. The organization of the packages
> and their APIs should not yet be regarded as stable, and may change between
> releases.

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

## Performance benchmarks

The opt-in phase-diagram benchmark compares httk with ASE while sweeping both
the number of phases and the number of species. The species sweep reaches twelve
species by default so that it captures the high-dimensional regime where
explicit convex-hull facet enumeration can become expensive.

```console
python -m pip install -e '.[benchmark]'
make benchmark
```

httk eagerly computes hull membership, energy above hull, decompositions, and
supported phase lines. ASE's constructor computes only its lower simplices, so
the benchmark reports both ASE construction alone and ASE construction followed
by a decomposition query for every unstable input phase. Use `--help` to change
either sweep, the repeat count, seed, or write full results and environment
metadata to JSON, for example:

```console
python benchmarks/run_phase_diagram_benchmarks.py \
  --species-counts 2 4 6 8 10 12 \
  --species-sweep-phases 40 \
  --repeats 5 \
  --json phase-diagram-benchmark.json
```
