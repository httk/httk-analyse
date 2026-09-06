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
Optional `httk-analyse[highs]` adds HiGHS acceleration: pass `solver="highs"`
to `LowerConvexHull`, either `PhaseDiagram` factory, or `PhaseDiagramBuilder`.
The default stays `solver="simplex"`. The accelerated route refines HiGHS bases
with the built-in numerical checks and falls back when a basis is unsuitable.

## Performance benchmarks

See the [optional HiGHS measurements](benchmarks/HIGHS_REPORT.md) for a
guarded comparison of both solver routes, including few-element cases.

The opt-in benchmark compares httk with ASE across phase count and species
count. It requires Linux with `/proc`, GNU `timeout`, and the benchmark extra.
Each timed repetition runs in a fresh process under `httk memguard`, with a
default 1 GiB sampled group RSS budget, a hard 2 GiB per-process address-space
limit, and a 60-second timeout. No measurements run concurrently.

```console
python -m pip install -e '.[benchmark]'
make benchmark
```

To measure the optional solver, install `.[benchmark,highs]` and pass
`--httk-solver highs` with a new `--json` path. Run the same cases with
`--httk-solver simplex` for the built-in baseline. Solver choice and the
HiGHS package version are recorded; ASE measurements are unchanged. HiGHS
uses one solver thread independently of the BLAS `--threads` setting.

Start with a small pilot:

```console
python benchmarks/run_phase_diagram_benchmarks.py \
  --phase-counts 20 40 --species-counts 2 3 4 \
  --species-sweep-phases 20 --repeats 3 \
  --json benchmark-results/pilot.json
```

The JSON checkpoint is atomically replaced after **each measurement**. A Markdown
report is written alongside it when the runner stops, including on interruption.
Repeat the same command with `--resume` to run only missing trials; parameters,
package versions, source fingerprints and recorded environment must match.
Existing results are never overwritten by a new run. A failed trial is retained
and stops further repetitions of that case/metric, while other measurements
continue. Use a new output path to retry failures under different budgets.

Render a report again without running workers:

```console
python benchmarks/run_phase_diagram_benchmarks.py \
  --json benchmark-results/pilot.json --report-only
```

The default `make benchmark` retains the original sweeps: 20/40/80 phases at
three species and 2/3/4/6/8/10/12 species at thirty phases, with three repetitions.
It writes `benchmark-results/phase-diagram.json` and `.md`. Pass arguments with
`make benchmark BENCHMARK_ARGS="--resume"`, or invoke the script directly.
Extend one axis at a time and stop expanding a regime once limits are reached;
high-dimensional facet enumeration can be expensive even with few phases.

### What is measured

- `httk`: construction, including membership, energy above hull, and unstable
  decompositions. It does **not** extract the lazy phase lines.
- `ase_construct`: ASE's constructor, which builds lower simplices only.
- `ase_full`: construction plus one decomposition query per unstable phase;
  this is the closer comparison with httk's eager analysis.
- `httk_phase_lines`: optional (`--phase-lines`) first phase-line access on a
  newly constructed diagram, with construction excluded from the timer.

Timings exclude imports, dataset generation and result validation. By contrast,
the timeout and reported per-worker peak RSS include all of these, and any
`--warmups`; phase-line RSS also includes its prerequisite construction. Medians
use successful trials only, explicitly showing partial completion. RSS is the
largest successful worker peak, not an incremental allocation measurement.
Failed workers can lack a reliable peak; their exit status and bounded diagnostic
log are retained instead. Timeout and memory-limit outcomes are not timings.

The report compares stable memberships and per-atom hull energies, and checks
composition/energy reconstruction, weight sums, and nonnegative weights. It
does not require identical decomposition indices: degenerate solutions can be
non-unique. Membership is reported separately because coplanar membership
conventions can differ. Numerical comparisons use absolute tolerance `1e-7`;
httk runs at its default `1e-8`, ASE at its own defaults. A numerical mismatch or
failed measurement gives exit code 1; setup errors give 2 and interruption 130.
The script also records stable/facet/line counts and separate LP-count estimates.

### Limits and reproducibility

`--max-rss-gb`, `--as-gb`, and `--timeout` set per-worker limits; `--threads`
sets OMP/OpenBLAS/MKL/BLIS threads (default one). The watchdog samples every
0.1 seconds and can overshoot its RSS budget. The address-space limit is a hard
kernel limit, but measures virtual rather than resident memory and can reject
large reservations. A timeout includes setup and has a three-second kill grace.
For an additional hard aggregate memory ceiling, run inside an externally
configured cgroup/container; the benchmark does not manage cgroups.

Inputs retain the original seeded generator: unique integer compositions,
pure-element endpoints, and random energies. These synthetic inputs do not
represent a real-material workload. Use several seeds and representative data
before drawing general conclusions. Run serious measurements on an otherwise
idle machine in frozen checkouts/environments. The runner records CPU, Python,
packages, Git heads, tracked dirty flags, and Python-source hashes; live source
changes stop the run and invalidate an in-flight measurement. Non-Python data
and same-version dependency binary replacements are not fingerprinted. Atomic
checkpoints protect against interrupted writes, not power-loss durability.
