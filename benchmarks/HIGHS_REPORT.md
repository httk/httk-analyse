# Optional HiGHS backend: exploratory measurements

Measured September 6, 2026. The optional `solver="highs"` route improves the
few-element cases tested here, but does not eliminate ASE's binary/ternary
advantage. Keep `solver="simplex"` as the default; HiGHS is an opt-in accelerator,
not a replacement for the existing numerical contract.

## Few elements: three fresh workers per timing

Seconds, medians of three trials. Construction includes hull membership,
energy above hull and unstable decompositions; lazy phase-line extraction is
not included. The ASE column is its construction plus unstable decompositions,
not its much cheaper constructor alone. ASE medians below are from the
matching baseline run.

| Elements | Phases | Built-in simplex | Checked HiGHS | Speedup | ASE full |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 2 | 320 | 1.704 | 0.728 | 2.34x | 0.059 |
| 3 | 20 | 0.0611 | 0.0279 | 2.19x | 0.00457 |
| 3 | 320 | 1.739 | 0.800 | 2.17x | 0.234 |
| 4 | 320 | 2.886 | 0.897 | 3.22x | 1.021 |

The four-element crossover is modest and workload-dependent. These results
do not support claiming that HiGHS is faster than ASE for binary or ternary
diagrams. A specialized low-dimensional algorithm remains a possible next
step if those cases need further improvement.

## Second seed: scaling and lazy phase lines

Seed 20260732, 80 phases. These are **single trials**, not three-run medians.
They corroborate the primary trend but are weaker timing evidence. Phase-line
times are additional first-access work, excluding prerequisite construction.

| Elements | Built-in construction | HiGHS construction | Built-in lines | HiGHS lines | ASE full |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 2 | 0.269 | 0.123 | 0.0184 | 0.0143 | 0.0118 |
| 3 | 0.351 | 0.132 | 0.137 | 0.0573 | 0.0309 |
| 4 | 0.479 | 0.142 | 0.701 | 0.230 | 0.0998 |
| 10 | 0.710 | 0.207 | 14.177 | 2.106 | 30.454 |

At ten elements, both LP routes outperform ASE's full calculation on this
dataset; HiGHS further improves construction and first-access phase lines.
Phase lines remain a substantial separate cost. These results do not establish
asymptotic scaling or a universal crossover dimension.

## Validation and memory

All 104 fresh-worker measurements completed successfully; no timeouts or
memory kills occurred. Every case matched ASE's stable membership and passed
the benchmark's numerical checks (absolute threshold `1e-7`). Across the two
httk routes, stable memberships agreed and the largest hull-energy difference
was `1.07e-14` eV/atom. The largest observed composition reconstruction error
was `3.33e-16`, energy reconstruction error `6.66e-16` eV/atom, and weight-sum
error `2.22e-16`. Both routes returned the same phase-line counts on the
second-seed cases; the benchmark does not retain complete edge lists, so that
is not an exact edge-set comparison.

The largest successful worker peak RSS, including ASE and all setup, was
128.9 MiB. This is not an incremental solver allocation or a worst-case bound.

## What was accelerated

The previous SciPy experiment rebuilt a solver on every mixture request.
Direct `highspy` model reuse is different: the coordinates/objective stay in
one model while column bounds select competitors and row bounds change the
target composition. HiGHS retains its basis between solves.

Raw basis reuse was substantially faster in a temporary prototype, but failed
tiny-energy regression cases. The implementation measured above therefore
uses HiGHS only to propose a basis. The existing equality scaling, rank and
conditioning checks, phase-II refinement, and residual checks operate on the
original active-column problem. An unsuitable basis or unsuccessful HiGHS
solve falls back to the built-in solver. The speedups above include that work;
they are not raw-prototype timings. No dimension-specific algorithm was added.

## Reproduction and limits

Environment: Intel Xeon E5-2680 v4 @ 2.40 GHz, Python 3.12.3, NumPy 2.4.3,
HiGHS/highspy 1.15.1, ASE 3.28.0, SciPy 1.17.1, Matplotlib 3.10.8.
Both routes were measured from the same working tree based on `653b43f`, with
the pending backend implementation. The analysis Python-source digest was
`efc56f435aacedf93fa4ef580cc0f6c1085bbf30eba60cf55a1ce0558020f8c8`.
The checkpoints contain hashes for the three httk Python source trees and
recorded package/environment metadata. Dependency binaries and non-Python
data are not fingerprinted. Tests/docs may change independently of that digest.

```console
python -m pip install -e '.[benchmark,highs]'
python benchmarks/run_phase_diagram_benchmarks.py \
  --httk-solver simplex --phase-counts 20 320 --species-counts 2 4 \
  --species-sweep-phases 320 --repeats 3 \
  --json benchmark-results/2026-09-06-highs/few-simplex.json
python benchmarks/run_phase_diagram_benchmarks.py \
  --httk-solver highs --phase-counts 20 320 --species-counts 2 4 \
  --species-sweep-phases 320 --repeats 3 \
  --json benchmark-results/2026-09-06-highs/few-highs.json
```

The second-seed command was run once for each solver, changing both
`--httk-solver` and the output basename:

```console
python benchmarks/run_phase_diagram_benchmarks.py \
  --httk-solver simplex --phase-counts 80 --species-counts 2 4 10 \
  --species-sweep-phases 80 --seed 20260732 --repeats 1 --phase-lines \
  --json benchmark-results/2026-09-06-highs/second-seed-simplex.json
```

Use new output paths to rerun; the supervisor intentionally refuses to
overwrite existing results. Raw JSON and generated Markdown are retained
locally in the ignored `benchmark-results/2026-09-06-highs/` directory.

Every timing runs sequentially in a fresh process, with imports outside the
timer, one BLAS thread and one HiGHS solver thread. GNU timeout is inside
`httk memguard`: 1 GiB sampled process-group RSS, 2 GiB per-process address
space, 60 seconds including setup, and a three-second kill grace. Worker peak
RSS includes imports, input generation and validation, not just the timed work.

Inputs are synthetic unique integer compositions with pure-element endpoints
and random energies, not representative material datasets. The primary seed
is 20260731, transformed by the existing generator for each case. Other host
workloads were not controlled, and backend runs were not interleaved. These
are useful engineering measurements, not publication-quality performance
guarantees. Larger phase counts and real-material data remain unmeasured here.
