"""Optional HiGHS candidate-basis provider for lower-hull mixtures."""

import importlib
from collections.abc import Sequence
from typing import Any

import numpy as np

from ._simplex import _PIVOT_TOLERANCE, _basis_is_well_conditioned, _scaled_independent_equalities, _simplex_iterations

highspy: Any = importlib.import_module("highspy")


class _HighsMixtureSolver:
    """Reuse one HiGHS model while retaining the local simplex as final arbiter."""

    def __init__(self, points: Sequence[Sequence[float]], values: Sequence[float]) -> None:
        self._points = points
        self._values = values
        self._enabled = set(range(len(points)))
        self._model: Any = None
        try:
            coordinates = np.asarray(points, dtype=np.float64)
            origin = coordinates[0]
            relative = coordinates - origin
            scale = np.max(np.abs(relative), axis=0)
            scale[scale == 0.0] = 1.0
            matrix = np.vstack(((relative / scale).T, np.ones(len(points))))
            costs = np.asarray(values, dtype=np.float64)
            shifted_costs = costs - np.min(costs)
            if not (np.all(np.isfinite(matrix)) and np.all(np.isfinite(shifted_costs))):
                raise FloatingPointError
        except (FloatingPointError, OverflowError, ValueError):
            return

        model = highspy.Highs()
        for name, value in {
            "output_flag": False,
            "solver": "simplex",
            "threads": 1,
            "primal_feasibility_tolerance": 1e-10,
            "dual_feasibility_tolerance": 1e-10,
            "small_matrix_value": 1e-12,
        }.items():
            if model.setOptionValue(name, value) == highspy.HighsStatus.kError:
                return
        row_count, column_count = matrix.shape
        lp = highspy.HighsLp()
        lp.num_col_, lp.num_row_ = column_count, row_count
        lp.col_cost_ = shifted_costs
        lp.col_lower_ = np.zeros(column_count)
        lp.col_upper_ = np.full(column_count, highspy.kHighsInf)
        rhs = np.r_[np.zeros(row_count - 1), 1.0]
        lp.row_lower_ = rhs
        lp.row_upper_ = rhs
        lp.a_matrix_.format_ = highspy.MatrixFormat.kRowwise
        lp.a_matrix_.start_ = np.arange(row_count + 1) * column_count
        lp.a_matrix_.index_ = np.tile(np.arange(column_count), row_count)
        lp.a_matrix_.value_ = matrix.ravel()
        if model.passModel(lp) == highspy.HighsStatus.kError:
            return
        self._model = model
        self._origin = origin
        self._scale = scale

    def solve(
        self,
        indices: Sequence[int],
        origin: tuple[float, ...],
        offsets: Sequence[float],
    ) -> tuple[float, tuple[float, ...]] | None:
        """Return a locally-polished mixture, or ``None`` for the reference solver."""
        if not indices or self._model is None:
            return None
        try:
            model = self._model
            current = set(indices)
            for index in current ^ self._enabled:
                upper = highspy.kHighsInf if index in current else 0.0
                if model.changeColBounds(index, 0.0, upper) == highspy.HighsStatus.kError:
                    self._model = None
                    return None
            self._enabled = current
            rhs = np.r_[
                (np.asarray(offsets) + (np.asarray(origin) - self._origin)) / self._scale,
                1.0,
            ]
            if not np.all(np.isfinite(rhs)):
                return None
            rows = np.arange(len(rhs))
            if model.changeRowsBounds(len(rhs), rows, rhs, rhs) == highspy.HighsStatus.kError:
                self._model = None
                return None
            if model.run() == highspy.HighsStatus.kError:
                self._model = None
                return None
            if model.getModelStatus() != highspy.HighsModelStatus.kOptimal:
                return None

            matrix = np.asarray(
                [[self._points[index][axis] - origin[axis] for index in indices] for axis in range(len(origin))]
                + [[1.0] * len(indices)],
                dtype=np.float64,
            )
            target = np.asarray([*offsets, 1.0], dtype=np.float64)
            costs = np.asarray([self._values[index] for index in indices], dtype=np.float64)
            baseline = float(np.min(costs))
            objective = costs - baseline
            if not (np.all(np.isfinite(matrix)) and np.all(np.isfinite(objective))):
                return None
            coefficients, target, full_coefficients, full_target = _scaled_independent_equalities(
                matrix, target, _PIVOT_TOLERANCE
            )
            statuses = model.getBasis().col_status
            basis = [
                position for position, index in enumerate(indices) if statuses[index] == highspy.HighsBasisStatus.kBasic
            ]
            if len(basis) != coefficients.shape[0] or not _basis_is_well_conditioned(
                coefficients, basis, _PIVOT_TOLERANCE
            ):
                return None
            _, weights = _simplex_iterations(
                coefficients,
                target,
                objective,
                basis,
                _PIVOT_TOLERANCE,
                local_objective_tolerance=True,
            )
            if np.any(np.abs(full_coefficients @ weights - full_target) > 100.0 * _PIVOT_TOLERANCE):
                return None
            return baseline + float((costs - baseline) @ weights), tuple(float(weight) for weight in weights)
        except (FloatingPointError, OverflowError, ValueError, RuntimeError, np.linalg.LinAlgError):
            return None
