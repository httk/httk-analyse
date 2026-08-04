"""Private deterministic solver for small equality-constrained linear programs."""

import math
from collections.abc import Sequence

import numpy as np

_PIVOT_TOLERANCE = 1e-11


def _reduced_cost_tolerances(matrix: np.ndarray, costs: np.ndarray, multipliers: np.ndarray) -> np.ndarray:
    """Return local float64 roundoff bounds for reduced-cost subtractions."""
    multiplication_terms = np.sum(np.abs(matrix) * np.abs(multipliers)[:, None], axis=0)
    return 32.0 * np.finfo(np.float64).eps * (np.abs(costs) + multiplication_terms)


class _LPInfeasibleError(ValueError):
    """The equality-constrained linear program has no feasible point."""


class _LPUnboundedError(ValueError):
    """The equality-constrained linear program is unbounded below."""


def _basis_is_well_conditioned(matrix: np.ndarray, basis: Sequence[int], tolerance: float) -> bool:
    """Return whether a candidate basis is numerically safe enough to solve."""
    if not basis:
        return True
    condition = float(np.linalg.cond(matrix[:, basis]))
    limit = 1.0 / max(tolerance, 100.0 * np.finfo(np.float64).eps)
    return math.isfinite(condition) and condition <= limit


def _matrix_rank(matrix: np.ndarray, threshold: float) -> int:
    """Return the SVD rank against one absolute singular-value threshold."""
    if matrix.size == 0:
        return 0
    singular_values = np.linalg.svd(matrix, compute_uv=False)
    return int(np.count_nonzero(singular_values > threshold))


def _scaled_independent_equalities(
    matrix: np.ndarray,
    rhs: np.ndarray,
    tolerance: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Scale equalities, reject inconsistency, and retain independent rows."""
    if matrix.shape[0] == 0:
        return matrix.copy(), rhs.copy(), matrix.copy(), rhs.copy()

    if matrix.shape[1] == 0:
        coefficient_sizes = np.zeros(matrix.shape[0], dtype=np.float64)
    else:
        coefficient_sizes = np.max(np.abs(matrix), axis=1)
    row_sizes = np.maximum(coefficient_sizes, np.abs(rhs))
    row_sizes[row_sizes == 0.0] = 1.0
    scaled_matrix = matrix / row_sizes[:, None]
    scaled_rhs = rhs / row_sizes
    augmented = np.column_stack((scaled_matrix, scaled_rhs))
    largest = float(np.linalg.svd(augmented, compute_uv=False)[0])
    rank_threshold = tolerance * max(1.0, largest)
    coefficient_rank = _matrix_rank(scaled_matrix, rank_threshold)
    augmented_rank = _matrix_rank(augmented, rank_threshold)
    if augmented_rank > coefficient_rank:
        raise _LPInfeasibleError("linear program is infeasible")

    selected: list[int] = []
    selected_rank = 0
    for row in range(scaled_matrix.shape[0]):
        trial = scaled_matrix[[*selected, row], :]
        trial_rank = _matrix_rank(trial, rank_threshold)
        if trial_rank > selected_rank:
            selected.append(row)
            selected_rank = trial_rank
        if selected_rank == coefficient_rank:
            break
    if selected_rank != coefficient_rank:
        raise RuntimeError("could not select independent equality constraints")
    return (
        scaled_matrix[selected, :],
        scaled_rhs[selected],
        scaled_matrix,
        scaled_rhs,
    )


def _simplex_iterations(
    matrix: np.ndarray,
    rhs: np.ndarray,
    costs: np.ndarray,
    basis: list[int],
    tolerance: float,
    *,
    local_objective_tolerance: bool = False,
) -> tuple[list[int], np.ndarray]:
    """Run revised-simplex pivots from a feasible basis using Bland's rule."""
    row_count, variable_count = matrix.shape
    if row_count == 0:
        if np.any(costs < 0.0):
            raise _LPUnboundedError("linear program is unbounded below")
        return basis, np.zeros(variable_count, dtype=np.float64)

    max_iterations = max(10_000, 100 * (row_count + variable_count))
    for _ in range(max_iterations):
        basis_matrix = matrix[:, basis]
        try:
            basic_values = np.linalg.solve(basis_matrix, rhs)
            multipliers = np.linalg.solve(basis_matrix.T, costs[basis])
        except np.linalg.LinAlgError as exc:
            raise RuntimeError("simplex basis became singular") from exc

        small_negative = (basic_values < 0.0) & (np.abs(basic_values) <= tolerance)
        basic_values[small_negative] = 0.0
        if np.any(basic_values < -tolerance):
            raise RuntimeError("simplex basis lost primal feasibility")

        reduced_costs = costs - matrix.T @ multipliers
        reduced_costs[basis] = 0.0
        basis_set = set(basis)
        if local_objective_tolerance:
            reduced_cost_tolerances = _reduced_cost_tolerances(matrix, costs, multipliers)
        else:
            reduced_cost_tolerances = np.full(variable_count, tolerance, dtype=np.float64)
        entering_candidates = [
            index
            for index in range(variable_count)
            if index not in basis_set and reduced_costs[index] < -reduced_cost_tolerances[index]
        ]
        if not entering_candidates:
            solution = np.zeros(variable_count, dtype=np.float64)
            solution[basis] = basic_values
            return basis, solution

        pivoted = False
        for entering in entering_candidates:
            direction = np.linalg.solve(basis_matrix, matrix[:, entering])
            eligible = [row for row in range(row_count) if direction[row] > tolerance]
            if not eligible:
                raise _LPUnboundedError("linear program is unbounded below")

            ratios = {row: basic_values[row] / direction[row] for row in eligible}
            minimum = min(ratios.values())
            machine_tolerance = 16.0 * np.finfo(np.float64).eps
            minimizers = [
                row
                for row in eligible
                if abs(ratios[row] - minimum) <= machine_tolerance * max(1.0, abs(minimum), abs(ratios[row]))
            ]
            for leaving_row in sorted(minimizers, key=basis.__getitem__):
                candidate_basis = basis.copy()
                candidate_basis[leaving_row] = entering
                if not _basis_is_well_conditioned(matrix, candidate_basis, tolerance):
                    continue
                basis = candidate_basis
                pivoted = True
                break
            if pivoted:
                break
        if not pivoted:
            raise RuntimeError("simplex found no numerically safe pivot")

    raise RuntimeError("simplex iteration limit exceeded")


def _solve_equality_lp(
    costs: Sequence[float],
    matrix: Sequence[Sequence[float]],
    rhs: Sequence[float],
    *,
    pivot_tolerance: float = _PIVOT_TOLERANCE,
) -> tuple[float, tuple[float, ...]]:
    """Minimize ``costs @ x`` subject to ``matrix @ x == rhs`` and ``x >= 0``.

    This deterministic dense two-phase revised simplex max-scales equality rows,
    uses SVD rank cleanup, and starts from artificial variables. Both entering and
    genuinely tied leaving variables follow Bland's anti-cycling rule. Candidate
    pivots whose bases are too ill-conditioned are skipped.

    Raises:
        _LPInfeasibleError: If the equality constraints cannot be satisfied.
        _LPUnboundedError: If the objective is unbounded below.
    """
    tolerance = float(pivot_tolerance)
    if not math.isfinite(tolerance) or tolerance <= 0.0:
        raise ValueError("pivot_tolerance must be a finite positive number")

    objective = np.asarray(costs, dtype=np.float64)
    coefficients = np.asarray(matrix, dtype=np.float64)
    target = np.asarray(rhs, dtype=np.float64)
    if objective.ndim != 1 or coefficients.ndim != 2 or target.ndim != 1:
        raise ValueError("costs and rhs must be vectors and matrix must be two-dimensional")
    row_count, variable_count = coefficients.shape
    if objective.shape != (variable_count,) or target.shape != (row_count,):
        raise ValueError("linear-program dimensions do not agree")
    if not (np.all(np.isfinite(objective)) and np.all(np.isfinite(coefficients)) and np.all(np.isfinite(target))):
        raise ValueError("linear-program inputs must be finite")
    coefficients, target, scaled_coefficients, scaled_target = _scaled_independent_equalities(
        coefficients,
        target,
        tolerance,
    )
    row_count = coefficients.shape[0]
    if variable_count == 0:
        return 0.0, ()

    negative_rhs = target < 0.0
    coefficients = coefficients.copy()
    target = target.copy()
    coefficients[negative_rhs] *= -1.0
    target[negative_rhs] *= -1.0

    artificial = np.eye(row_count, dtype=np.float64)
    phase_one_matrix = np.concatenate((coefficients, artificial), axis=1)
    phase_one_costs = np.concatenate(
        (
            np.zeros(variable_count, dtype=np.float64),
            np.ones(row_count, dtype=np.float64),
        )
    )
    basis = list(range(variable_count, variable_count + row_count))
    basis, phase_one_solution = _simplex_iterations(
        phase_one_matrix,
        target,
        phase_one_costs,
        basis,
        tolerance,
    )
    artificial_sum = float(np.sum(phase_one_solution[variable_count:]))
    if artificial_sum > tolerance:
        raise _LPInfeasibleError("linear program is infeasible")

    # A zero artificial basic variable either pivots onto an original column or marks a
    # redundant equality.
    while any(index >= variable_count for index in basis):
        artificial_row = min(
            (row for row, index in enumerate(basis) if index >= variable_count),
            key=basis.__getitem__,
        )
        basis_matrix = phase_one_matrix[:, basis]
        tableau_original = np.linalg.solve(basis_matrix, phase_one_matrix[:, :variable_count])
        basis_set = set(basis)
        entering_candidates = [
            index
            for index in range(variable_count)
            if index not in basis_set and abs(tableau_original[artificial_row, index]) > tolerance
        ]
        pivoted = False
        for entering in entering_candidates:
            candidate_basis = basis.copy()
            candidate_basis[artificial_row] = entering
            if not _basis_is_well_conditioned(phase_one_matrix, candidate_basis, tolerance):
                continue
            basis = candidate_basis
            pivoted = True
            break
        if pivoted:
            continue
        if entering_candidates:
            raise RuntimeError("simplex found no numerically safe artificial-variable pivot")
        phase_one_matrix = np.delete(phase_one_matrix, artificial_row, axis=0)
        target = np.delete(target, artificial_row)
        del basis[artificial_row]

    phase_two_matrix = phase_one_matrix[:, :variable_count]
    _, solution = _simplex_iterations(
        phase_two_matrix,
        target,
        objective,
        basis,
        tolerance,
        local_objective_tolerance=True,
    )
    residual = scaled_coefficients @ solution - scaled_target
    if np.any(np.abs(residual) > 100.0 * tolerance):
        raise _LPInfeasibleError("linear program did not reach a feasible point")
    return float(objective @ solution), tuple(float(value) for value in solution)
