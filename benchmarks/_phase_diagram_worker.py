"""Run one bounded phase-diagram benchmark measurement in a child process."""

import gc
import importlib
import json
import math
import random
import sys
import time
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

_METRICS = frozenset({"httk", "httk_phase_lines", "ase_construct", "ase_full"})
_SOLVERS = frozenset({"simplex", "highs"})
_HTTK_TOLERANCE = 1e-8


def validate_case(case_dict: dict[str, object], atom_count: int) -> None:
    """Validate a deterministic synthetic benchmark case without importing scientific packages.

    :param case_dict: Case data supplied by the benchmark supervisor.
    :param atom_count: Formula-unit atom count used to generate unique compositions.
    :raises ValueError: If the case cannot produce the requested unique inputs.
    """
    if not isinstance(case_dict, dict):
        raise ValueError("case must be an object")
    if not isinstance(case_dict.get("sweep"), str):
        raise ValueError("case.sweep must be a string")
    for name in ("species", "phases", "seed"):
        value = case_dict.get(name)
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError(f"case.{name} must be an integer")
    if isinstance(atom_count, bool) or not isinstance(atom_count, int) or atom_count <= 0:
        raise ValueError("atom_count must be a positive integer")

    species = case_dict["species"]
    phases = case_dict["phases"]
    assert isinstance(species, int)
    assert isinstance(phases, int)
    if not 2 <= species <= 118:
        raise ValueError("case.species must be between 2 and 118")
    if phases <= species:
        raise ValueError("case.phases must exceed case.species")
    maximum = math.comb(atom_count + species - 1, species - 1)
    if phases > maximum:
        raise ValueError("case.phases exceeds the number of unique compositions")


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


def _dataset(case: Mapping[str, object], atom_count: int) -> tuple[tuple[dict[str, int], ...], tuple[float, ...]]:
    """Build the benchmark's reproducible full-composition-span dataset."""
    from ase.data import chemical_symbols

    species = case["species"]
    phases = case["phases"]
    seed = case["seed"]
    assert isinstance(species, int)
    assert isinstance(phases, int)
    assert isinstance(seed, int)
    rng = random.Random(seed)
    rows: list[tuple[int, ...]] = []
    for endpoint in range(species):
        row = [0] * species
        row[endpoint] = atom_count
        rows.append(tuple(row))

    seen = set(rows)
    while len(rows) < phases:
        row = _random_composition(species, atom_count, rng)
        if row not in seen:
            seen.add(row)
            rows.append(row)

    symbols = chemical_symbols[1 : species + 1]
    compositions = tuple({symbol: count for symbol, count in zip(symbols, row, strict=True) if count} for row in rows)
    energies_per_atom = [0.0] * species
    energies_per_atom.extend(rng.uniform(-1.0, 0.2) for _ in range(phases - species))
    return compositions, tuple(atom_count * energy for energy in energies_per_atom)


def _warm(operation: Callable[[], object], warmups: int) -> None:
    """Run and release untimed operations before the single measurement."""
    for _ in range(warmups):
        result = operation()
        del result
        gc.collect()


def _time(operation: Callable[[], Any]) -> tuple[float, Any]:
    """Measure one operation after collecting previous benchmark results."""
    gc.collect()
    started = time.perf_counter()
    result = operation()
    return time.perf_counter() - started, result


def _composition_row(composition: Mapping[str, int], elements: Sequence[str]) -> tuple[float, ...]:
    """Return an atomic-fraction composition row in one shared element order."""
    atoms = sum(composition.values())
    return tuple(composition.get(element, 0) / atoms for element in elements)


def _summary_from_mixtures(
    compositions: Sequence[Mapping[str, int]],
    energies: Sequence[float],
    stable_indices: Sequence[int],
    mixtures: Sequence[Sequence[tuple[int, float]]],
    hull_energies: Sequence[float],
) -> dict[str, object]:
    """Validate mixture reconstructions and return the common scientific summary."""
    elements = tuple(sorted({element for composition in compositions for element in composition}))
    rows = tuple(_composition_row(composition, elements) for composition in compositions)
    per_atom = tuple(
        energy / sum(composition.values()) for composition, energy in zip(compositions, energies, strict=True)
    )
    composition_residuals: list[float] = []
    energy_residuals: list[float] = []
    weight_errors: list[float] = []
    weights: list[float] = []
    for index, mixture in enumerate(mixtures):
        composition_residuals.extend(
            abs(sum(weight * rows[reference][axis] for reference, weight in mixture) - rows[index][axis])
            for axis in range(len(elements))
        )
        weight_errors.append(abs(sum(weight for _, weight in mixture) - 1.0))
        weights.extend(weight for _, weight in mixture)
        reconstructed_energy = sum(per_atom[reference] * weight for reference, weight in mixture)
        energy_residuals.append(abs(reconstructed_energy - hull_energies[index]))
    return {
        "stable_indices": list(stable_indices),
        "hull_energies": [float(energy) for energy in hull_energies],
        "max_composition_residual": max(composition_residuals, default=0.0),
        "max_energy_residual": max(energy_residuals, default=0.0),
        "max_weight_sum_error": max(weight_errors, default=0.0),
        "min_weight": min(weights, default=0.0),
    }


def _run_httk(
    metric: str,
    compositions: tuple[dict[str, int], ...],
    energies: tuple[float, ...],
    warmups: int,
    solver: str = "simplex",
) -> tuple[float, dict[str, object]]:
    """Measure httk construction or the cold phase-line query."""
    from httk.analyse.matsci import PhaseDiagram

    if solver == "highs":
        importlib.import_module("httk.analyse.generic._highs")
    construct = lambda: PhaseDiagram.from_compositions(compositions, energies, solver=solver)
    if metric == "httk_phase_lines":

        def construct_and_phase_lines() -> Any:
            diagram = construct()
            _ = diagram.phase_lines
            return diagram

        _warm(construct_and_phase_lines, warmups)
        diagram = construct()
        seconds, lines = _time(lambda: diagram.phase_lines)
        stable_indices = list(diagram.hull_indices)
        return seconds, {
            "stable_indices": stable_indices,
            "phase_lines": len(lines),
            "lp_solves_estimate": len(stable_indices) * (len(stable_indices) - 1) // 2,
            "algorithm_tolerance": _HTTK_TOLERANCE,
        }

    _warm(construct, warmups)
    seconds, diagram = _time(construct)
    stable = set(diagram.hull_indices)
    mixtures = tuple(
        ((index, 1.0),) if index in stable else diagram.decomposition(index) for index in range(len(diagram))
    )
    if any(mixture is None for mixture in mixtures):
        raise RuntimeError("unstable httk phase has no decomposition")
    summary = _summary_from_mixtures(
        compositions,
        energies,
        diagram.hull_indices,
        tuple(mixture for mixture in mixtures if mixture is not None),
        tuple(
            energy - above for energy, above in zip(diagram.energies_per_atom, diagram.energy_above_hull, strict=True)
        ),
    )
    summary["lp_solves_estimate"] = len(diagram) + (len(diagram) - len(diagram.hull_indices))
    summary["algorithm_tolerance"] = _HTTK_TOLERANCE
    return seconds, summary


def _run_ase(
    metric: str,
    compositions: tuple[dict[str, int], ...],
    energies: tuple[float, ...],
    warmups: int,
) -> tuple[float, dict[str, object]]:
    """Measure ASE construction, optionally including every unstable decomposition."""
    from ase.phasediagram import PhaseDiagram

    references = tuple(zip(compositions, energies, strict=True))
    construct = lambda: PhaseDiagram(references, verbose=False)
    if metric == "ase_construct":
        _warm(construct, warmups)
        seconds, diagram = _time(construct)
        return seconds, {
            "stable_indices": [index for index, stable in enumerate(diagram.hull) if stable],
            "lower_facets": len(diagram.simplices),
        }

    def construct_and_decompose() -> tuple[Any, dict[int, tuple[Any, Any, Any]]]:
        diagram = construct()
        decompositions = {
            index: diagram.decompose(**composition)
            for index, composition in enumerate(compositions)
            if not diagram.hull[index]
        }
        return diagram, decompositions

    _warm(construct_and_decompose, warmups)
    seconds, (diagram, decompositions) = _time(construct_and_decompose)
    stable_indices = tuple(index for index, stable in enumerate(diagram.hull) if stable)
    stable = set(stable_indices)
    atom_counts = tuple(sum(composition.values()) for composition in compositions)
    mixtures: list[tuple[tuple[int, float], ...]] = []
    hull_energies: list[float] = []
    for index, energy in enumerate(energies):
        if index in stable:
            mixtures.append(((index, 1.0),))
            hull_energies.append(energy / atom_counts[index])
            continue
        total_energy, indices, coefficients = decompositions[index]
        mixtures.append(
            tuple(
                (int(reference), float(coefficient) * atom_counts[int(reference)] / atom_counts[index])
                for reference, coefficient in zip(indices, coefficients, strict=True)
            )
        )
        hull_energies.append(float(total_energy) / atom_counts[index])
    summary = _summary_from_mixtures(compositions, energies, stable_indices, mixtures, hull_energies)
    summary["lower_facets"] = len(diagram.simplices)
    return seconds, summary


def _request_value(request: Mapping[str, object], name: str) -> int:
    """Return one required non-negative request integer."""
    value = request.get(name)
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return value


def _solver_value(request: Mapping[str, object]) -> str:
    """Return the requested HTTK solver, defaulting old requests to simplex."""
    solver = request.get("solver", "simplex")
    if not isinstance(solver, str) or solver not in _SOLVERS:
        raise ValueError("solver must be one of simplex, highs")
    return solver


def _run_request(request: object) -> tuple[float, dict[str, object]]:
    """Validate and execute a single worker request."""
    if not isinstance(request, dict):
        raise ValueError("request must be an object")
    case = request.get("case")
    atom_count = _request_value(request, "atom_count")
    if atom_count == 0:
        raise ValueError("atom_count must be a positive integer")
    warmups = _request_value(request, "warmups")
    solver = _solver_value(request)
    metric = request.get("metric")
    if not isinstance(metric, str) or metric not in _METRICS:
        raise ValueError(f"metric must be one of {', '.join(sorted(_METRICS))}")
    if not isinstance(case, dict):
        raise ValueError("case must be an object")
    validate_case(case, atom_count)
    compositions, energies = _dataset(case, atom_count)
    if metric in {"httk", "httk_phase_lines"}:
        return _run_httk(metric, compositions, energies, warmups, solver=solver)
    return _run_ase(metric, compositions, energies, warmups)


def _peak_rss_bytes() -> int:
    """Return the Linux peak resident set size in bytes."""
    import resource

    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024


def _write_response(path: Path, response: Mapping[str, object]) -> None:
    """Write a strict JSON response for the supervisor."""
    path.write_text(json.dumps(response, allow_nan=False) + "\n", encoding="utf-8")


def main(argv: Sequence[str] | None = None) -> int:
    """Run the request/response worker protocol.

    :param argv: Optional command-line argument sequence excluding the program name.
    :return: Process exit status.
    """
    arguments = tuple(sys.argv[1:] if argv is None else argv)
    if len(arguments) != 2:
        print("usage: _phase_diagram_worker.py REQUEST.json RESPONSE.json", file=sys.stderr)
        return 2
    request_path, response_path = (Path(argument) for argument in arguments)
    try:
        request = json.loads(request_path.read_text(encoding="utf-8"))
        seconds, summary = _run_request(request)
        response: dict[str, object] = {
            "status": "ok",
            "seconds": seconds,
            "peak_rss_bytes": _peak_rss_bytes(),
            "summary": summary,
            "error": None,
        }
    except MemoryError:
        response = {
            "status": "memory_limit",
            "seconds": None,
            "peak_rss_bytes": _peak_rss_bytes(),
            "summary": {},
            "error": "MemoryError",
        }
    except Exception as exc:
        response = {
            "status": "error",
            "seconds": None,
            "peak_rss_bytes": _peak_rss_bytes(),
            "summary": {},
            "error": f"{type(exc).__name__}: {exc}",
        }
    try:
        _write_response(response_path, response)
    except Exception as exc:
        print(f"could not write response: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2
    return 0 if response["status"] == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())
