#!/usr/bin/env python3
"""Compare httk and ASE phase-diagram construction across scale and dimension.

The two implementations do different amounts of eager work.  The httk timing
includes stability, energy-above-hull values, decompositions for every unstable
phase, and supported phase lines.  ASE's constructor only builds the lower
simplices, so this script reports both that constructor and a second ASE timing
that also requests a decomposition for every unstable input phase.  Stable
phases need no query because ASE's hull mask already identifies them.

Inputs are deterministic synthetic compositions.  Every dataset includes all
pure-element endpoints, followed by unique random integer compositions and
random energies per atom.  The default sweeps vary phase count at three species
and species count (up to twelve) at a fixed phase count.
"""

import argparse
import json
import os
import platform
import random
import statistics
import sys
import time
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

import numpy as np

from httk.analyse.matsci import PhaseDiagram as HttkPhaseDiagram

try:
    import ase
    import scipy
    from ase.data import chemical_symbols
    from ase.phasediagram import PhaseDiagram as ASEPhaseDiagram
except ImportError as exc:
    raise SystemExit(
        "The phase-diagram benchmark requires ASE; install this checkout with "
        "`python -m pip install -e '.[benchmark]'`."
    ) from exc

type Composition = dict[str, int]


@dataclass(frozen=True, slots=True)
class BenchmarkCase:
    """One deterministic point in a phase-count or species-count sweep."""

    sweep: str
    species: int
    phases: int
    seed: int


@dataclass(frozen=True, slots=True)
class BenchmarkResult:
    """Measurements and output-size checks for one benchmark case."""

    sweep: str
    species: int
    phases: int
    httk_seconds: float | None
    ase_construct_seconds: float | None
    ase_construct_and_decompose_unstable_seconds: float | None
    httk_over_ase_construct: float | None
    httk_over_ase_construct_and_decompose_unstable: float | None
    httk_stable_phases: int | None
    ase_stable_phases: int | None
    httk_phase_lines: int | None
    httk_lp_solves: int | None
    ase_lower_facets: int | None
    stable_membership_matches: bool | None
    httk_error: str | None
    ase_construct_error: str | None
    ase_decompose_error: str | None


def _random_composition(species: int, atom_count: int, rng: random.Random) -> tuple[int, ...]:
    """Draw one integer composition with exactly ``atom_count`` atoms."""
    weights = [rng.expovariate(1.0) for _ in range(species)]
    total_weight = sum(weights)
    scaled = [atom_count * weight / total_weight for weight in weights]
    counts = [int(value) for value in scaled]
    remaining = atom_count - sum(counts)
    largest_remainders = sorted(
        range(species),
        key=lambda index: scaled[index] - counts[index],
        reverse=True,
    )
    for index in largest_remainders[:remaining]:
        counts[index] += 1
    return tuple(counts)


def _dataset(case: BenchmarkCase, atom_count: int) -> tuple[tuple[Composition, ...], tuple[float, ...]]:
    """Build a reproducible, full-composition-span phase-diagram dataset."""
    if case.species > len(chemical_symbols) - 1:
        raise ValueError(f"at most {len(chemical_symbols) - 1} species are available")
    if case.phases <= case.species:
        raise ValueError("phase count must exceed species count so the lifted hull is nontrivial")

    rng = random.Random(case.seed)
    rows: list[tuple[int, ...]] = []
    for endpoint in range(case.species):
        row = [0] * case.species
        row[endpoint] = atom_count
        rows.append(tuple(row))

    seen = set(rows)
    while len(rows) < case.phases:
        row = _random_composition(case.species, atom_count, rng)
        if row in seen:
            continue
        seen.add(row)
        rows.append(row)

    symbols = chemical_symbols[1 : case.species + 1]
    compositions = tuple({symbol: count for symbol, count in zip(symbols, row, strict=True) if count} for row in rows)
    energies_per_atom = [0.0] * case.species
    energies_per_atom.extend(rng.uniform(-1.0, 0.2) for _ in range(case.phases - case.species))
    total_energies = tuple(atom_count * energy for energy in energies_per_atom)
    return compositions, total_energies


def _measure[ResultT](
    operation: Callable[[], ResultT],
    *,
    repeats: int,
    warmups: int,
) -> tuple[float, ResultT]:
    """Return the median wall time and the final result of an operation."""
    for _ in range(warmups):
        operation()
    timings: list[float] = []
    result: ResultT | None = None
    for _ in range(repeats):
        started = time.perf_counter()
        result = operation()
        timings.append(time.perf_counter() - started)
    if result is None:
        raise RuntimeError("benchmark operation did not run")
    return statistics.median(timings), result


def _ase_construct_and_decompose(
    references: Sequence[tuple[Composition, float]],
    compositions: Sequence[Composition],
) -> Any:
    diagram = ASEPhaseDiagram(references, verbose=False)
    for index, composition in enumerate(compositions):
        if not diagram.hull[index]:
            diagram.decompose(**composition)
    return diagram


def _error_text(exc: Exception) -> str:
    return f"{type(exc).__name__}: {exc}"


def _ratio(numerator: float | None, denominator: float | None) -> float | None:
    if numerator is None or denominator is None or denominator == 0.0:
        return None
    return numerator / denominator


def _run_case(
    case: BenchmarkCase,
    *,
    atom_count: int,
    repeats: int,
    warmups: int,
) -> BenchmarkResult:
    compositions, energies = _dataset(case, atom_count)
    references = tuple(zip(compositions, energies, strict=True))

    httk_seconds: float | None = None
    httk_diagram: HttkPhaseDiagram | None = None
    httk_error: str | None = None
    try:
        httk_seconds, httk_diagram = _measure(
            lambda: HttkPhaseDiagram.from_compositions(compositions, energies),
            repeats=repeats,
            warmups=warmups,
        )
    except Exception as exc:
        httk_error = _error_text(exc)

    ase_construct_seconds: float | None = None
    ase_diagram: Any = None
    ase_construct_error: str | None = None
    try:
        ase_construct_seconds, ase_diagram = _measure(
            lambda: ASEPhaseDiagram(references, verbose=False),
            repeats=repeats,
            warmups=warmups,
        )
    except Exception as exc:
        ase_construct_error = _error_text(exc)

    ase_full_seconds: float | None = None
    ase_decompose_error: str | None = None
    if ase_construct_error is None:
        try:
            ase_full_seconds, _ = _measure(
                lambda: _ase_construct_and_decompose(references, compositions),
                repeats=repeats,
                warmups=warmups,
            )
        except Exception as exc:
            ase_decompose_error = _error_text(exc)

    httk_stable = len(httk_diagram.hull_indices) if httk_diagram is not None else None
    ase_stable_indices = (
        tuple(int(index) for index in np.flatnonzero(ase_diagram.hull)) if ase_diagram is not None else None
    )
    ase_stable = len(ase_stable_indices) if ase_stable_indices is not None else None
    stable_match = (
        set(httk_diagram.hull_indices) == set(ase_stable_indices)
        if httk_diagram is not None and ase_stable_indices is not None
        else None
    )
    return BenchmarkResult(
        sweep=case.sweep,
        species=case.species,
        phases=case.phases,
        httk_seconds=httk_seconds,
        ase_construct_seconds=ase_construct_seconds,
        ase_construct_and_decompose_unstable_seconds=ase_full_seconds,
        httk_over_ase_construct=_ratio(httk_seconds, ase_construct_seconds),
        httk_over_ase_construct_and_decompose_unstable=_ratio(httk_seconds, ase_full_seconds),
        httk_stable_phases=httk_stable,
        ase_stable_phases=ase_stable,
        httk_phase_lines=len(httk_diagram.phase_lines) if httk_diagram is not None else None,
        httk_lp_solves=(
            case.phases + (case.phases - httk_stable) + httk_stable * (httk_stable - 1) // 2
            if httk_stable is not None
            else None
        ),
        ase_lower_facets=len(ase_diagram.simplices) if ase_diagram is not None else None,
        stable_membership_matches=stable_match,
        httk_error=httk_error,
        ase_construct_error=ase_construct_error,
        ase_decompose_error=ase_decompose_error,
    )


def _positive(value: str) -> int:
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return number


def _nonnegative(value: str) -> int:
    number = int(value)
    if number < 0:
        raise argparse.ArgumentTypeError("must be non-negative")
    return number


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--phase-counts",
        type=_positive,
        nargs="+",
        default=(20, 40, 80),
        help="phase counts for the fixed-species sweep (default: 20 40 80)",
    )
    parser.add_argument(
        "--phase-sweep-species",
        type=_positive,
        default=3,
        help="species count for the phase-count sweep (default: 3)",
    )
    parser.add_argument(
        "--species-counts",
        type=_positive,
        nargs="+",
        default=(2, 3, 4, 6, 8, 10, 12),
        help="species counts for the fixed-phase sweep (default: 2 3 4 6 8 10 12)",
    )
    parser.add_argument(
        "--species-sweep-phases",
        type=_positive,
        default=30,
        help="phase count for the species-count sweep (default: 30)",
    )
    parser.add_argument("--atom-count", type=_positive, default=10_000)
    parser.add_argument("--repeats", type=_positive, default=3, help="timed repetitions; median is reported")
    parser.add_argument("--warmups", type=_nonnegative, default=0, help="untimed repetitions before each timing")
    parser.add_argument("--seed", type=int, default=20_260_731)
    parser.add_argument("--json", type=Path, dest="json_path", help="also write measurements and metadata as JSON")
    return parser.parse_args()


def _cases(args: argparse.Namespace) -> tuple[BenchmarkCase, ...]:
    cases: list[BenchmarkCase] = []
    for phases in args.phase_counts:
        cases.append(
            BenchmarkCase(
                "phases",
                args.phase_sweep_species,
                phases,
                args.seed + 1_000_003 * args.phase_sweep_species + phases,
            )
        )
    for species in args.species_counts:
        cases.append(
            BenchmarkCase(
                "species",
                species,
                args.species_sweep_phases,
                args.seed + 1_000_003 * species + args.species_sweep_phases,
            )
        )
    return tuple(cases)


def _milliseconds(value: float | None) -> str:
    return "ERR" if value is None else f"{1000.0 * value:.3f}"


def _number(value: float | None) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.2f}x"
    return str(value)


def _print_results(results: Sequence[BenchmarkResult]) -> None:
    print("\nTimings are medians; lower is better.  Times are milliseconds.")
    print("S=species, N=phases, H=httk stable phases, LPs=httk equality-LP solves.")
    print(
        f"{'sweep':8} {'S':>3} {'N':>5} {'httk':>11} {'ASE hull':>11} {'ratio':>9} "
        f"{'ASE hull+unstable':>20} {'ratio':>9} {'H':>5} {'LPs':>7} {'facets':>8} {'lines':>7} {'match':>7}"
    )
    print("-" * 134)
    for result in results:
        match = "yes" if result.stable_membership_matches else "NO"
        if result.stable_membership_matches is None:
            match = "-"
        print(
            f"{result.sweep:8} {result.species:3d} {result.phases:5d} "
            f"{_milliseconds(result.httk_seconds):>11} "
            f"{_milliseconds(result.ase_construct_seconds):>11} "
            f"{_number(result.httk_over_ase_construct):>9} "
            f"{_milliseconds(result.ase_construct_and_decompose_unstable_seconds):>20} "
            f"{_number(result.httk_over_ase_construct_and_decompose_unstable):>9} "
            f"{_number(result.httk_stable_phases):>5} "
            f"{_number(result.httk_lp_solves):>7} "
            f"{_number(result.ase_lower_facets):>8} "
            f"{_number(result.httk_phase_lines):>7} "
            f"{match:>7}"
        )

    errors = [
        (result, label, message)
        for result in results
        for label, message in (
            ("httk", result.httk_error),
            ("ASE construction", result.ase_construct_error),
            ("ASE decomposition", result.ase_decompose_error),
        )
        if message is not None
    ]
    if errors:
        print("\nErrors:")
        for result, label, message in errors:
            print(f"  {result.sweep} S={result.species} N={result.phases}, {label}: {message}")


def _package_version(name: str) -> str:
    try:
        return version(name)
    except PackageNotFoundError:
        return "source checkout"


def _metadata(args: argparse.Namespace) -> dict[str, object]:
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "packages": {
            "httk-analyse": _package_version("httk-analyse"),
            "numpy": np.__version__,
            "ase": ase.__version__,
            "scipy": scipy.__version__,
        },
        "thread_environment": {
            name: os.environ.get(name) for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")
        },
        "parameters": {
            "phase_counts": args.phase_counts,
            "phase_sweep_species": args.phase_sweep_species,
            "species_counts": args.species_counts,
            "species_sweep_phases": args.species_sweep_phases,
            "atom_count": args.atom_count,
            "repeats": args.repeats,
            "warmups": args.warmups,
            "seed": args.seed,
        },
    }


def main() -> int:
    args = _parse_args()
    cases = _cases(args)
    for case in cases:
        if case.phases <= case.species:
            raise SystemExit(
                f"invalid {case.sweep} case S={case.species}, N={case.phases}: phase count must exceed species count"
            )

    metadata = _metadata(args)
    packages = metadata["packages"]
    assert isinstance(packages, dict)
    print(
        f"Python {platform.python_version()}, httk-analyse {packages['httk-analyse']}, "
        f"NumPy {packages['numpy']}, ASE {packages['ase']}, SciPy {packages['scipy']}"
    )
    print(
        "httk construction is eager; ASE hull+unstable includes construction and one "
        "decomposition query per unstable input phase."
    )

    results: list[BenchmarkResult] = []
    for case in cases:
        print(f"Running {case.sweep} sweep S={case.species}, N={case.phases} ...", flush=True)
        results.append(
            _run_case(
                case,
                atom_count=args.atom_count,
                repeats=args.repeats,
                warmups=args.warmups,
            )
        )
    _print_results(results)

    if args.json_path is not None:
        payload = {"metadata": metadata, "results": [asdict(result) for result in results]}
        args.json_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(f"\nWrote {args.json_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
