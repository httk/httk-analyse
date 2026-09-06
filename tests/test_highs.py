"""Optional HiGHS lower-hull backend checks."""

import importlib

import pytest

from httk.analyse.generic import LowerConvexHull, lower_hull


def _require_highs() -> None:
    pytest.importorskip("highspy")


def _assert_matching_hulls(points: list[tuple[float, ...]], values: list[float], tolerance: float = 1e-8) -> None:
    _require_highs()
    simplex = LowerConvexHull(points, values, tolerance=tolerance)
    highs = LowerConvexHull(points, values, tolerance=tolerance, solver="highs")
    before = repr(highs), hash(highs)

    assert highs.solver == "highs"
    assert highs.hull_indices == simplex.hull_indices
    assert highs.value_above_hull == pytest.approx(simplex.value_above_hull, abs=1e-10)
    assert highs.supported_segments == simplex.supported_segments
    assert (repr(highs), hash(highs)) == before
    for index in range(len(highs)):
        decomposition = highs.decomposition(index)
        if decomposition is None:
            continue
        assert sum(weight for _, weight in decomposition) == pytest.approx(1.0, abs=1e-10)
        for axis in range(len(points[index])):
            assert sum(highs.points[candidate][axis] * weight for candidate, weight in decomposition) == pytest.approx(
                highs.points[index][axis], abs=1e-9
            )
        assert sum(highs.values[candidate] * weight for candidate, weight in decomposition) == pytest.approx(
            highs.values[index] - highs.value_above_hull[index], abs=1e-9
        )


@pytest.mark.parametrize(
    ("points", "values", "tolerance"),
    [
        ([(0.0,), (1.0,), (0.5,)], [0.0, 0.0, 1.0], 1e-8),
        (
            [(1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0), (1 / 3, 1 / 3, 1 / 3)],
            [0.0, 0.0, 0.0, -1.0],
            1e-8,
        ),
        ([(0.0,), (0.0,), (0.0,)], [5e-12, 0.0, 3e-12], 0.0),
        ([(0.0,), (1.0,), (5e-11,)], [0.0, 0.0, 1.0], 1e-8),
        ([(1e12,), (1e12 + 1.0,), (1e12 + 0.5,)], [0.0, 0.0, 1.0], 1e-8),
        ([(0.0,), (5e-10,)], [0.0, 0.0], 1e-8),
        ([(0.0,), (1.0,), (2.0,), (1e12,)], [0.0, 0.0, 0.0, 0.0], 1e-8),
        ([(0.0,)], [0.0], 1e-8),
        ([(), (), ()], [0.0, 0.0, 0.0], 1e-8),
        ([(0.0,), (1.0,), (2.0,)], [0.0, 0.0, 0.0], 1e-8),
    ],
)
def test_highs_matches_simplex(points: list[tuple[float, ...]], values: list[float], tolerance: float) -> None:
    _assert_matching_hulls(points, values, tolerance)


def test_highs_polishes_a_real_candidate_basis(monkeypatch: pytest.MonkeyPatch) -> None:
    _require_highs()
    highs = importlib.import_module("httk.analyse.generic._highs")
    calls = 0
    original = highs._simplex_iterations

    def polish(*args: object, **kwargs: object) -> object:
        nonlocal calls
        calls += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(highs, "_simplex_iterations", polish)
    LowerConvexHull([(0.0,), (1.0,), (0.5,)], [0.0, 0.0, 1.0], solver="highs")

    assert calls > 0


def test_solver_choice_is_excluded_from_exact_identity() -> None:
    _require_highs()
    simplex = LowerConvexHull([(0.0,), (1.0,)], [0.0, 0.0])
    highs = LowerConvexHull([(0.0,), (1.0,)], [0.0, 0.0], solver="highs")

    assert highs == simplex
    assert hash(highs) == hash(simplex)


def test_highs_preserves_tiny_weights_and_energies() -> None:
    _require_highs()
    weighted = LowerConvexHull([(0.0,), (1.0,), (5e-11,)], [0.0, 0.0, 1.0], solver="highs")
    duplicate = LowerConvexHull([(0.0,), (0.0,), (0.0,)], [5e-12, 0.0, 3e-12], solver="highs", tolerance=0.0)

    decomposition = weighted.decomposition(2)
    assert decomposition is not None
    assert tuple(index for index, _ in decomposition) == (0, 1)
    assert tuple(weight for _, weight in decomposition) == pytest.approx((1.0 - 5e-11, 5e-11), rel=0, abs=1e-15)
    assert duplicate.value_above_hull == pytest.approx((5e-12, 0.0, 3e-12), abs=1e-15)


def test_highs_reuses_one_model_for_analysis_and_one_for_segments(monkeypatch: pytest.MonkeyPatch) -> None:
    _require_highs()
    highs = importlib.import_module("httk.analyse.generic._highs")
    calls = 0
    original = highs._HighsMixtureSolver.__init__

    def build(self: object, *args: object, **kwargs: object) -> None:
        nonlocal calls
        calls += 1
        original(self, *args, **kwargs)

    monkeypatch.setattr(highs._HighsMixtureSolver, "__init__", build)
    hull = LowerConvexHull([(0.0,), (0.5,), (1.0,)], [0.0, -1.0, 0.0], solver="highs")

    assert calls == 1
    assert hull.supported_segments == ((0, 1), (1, 2))
    assert calls == 2


def test_highs_solver_falls_back_when_candidate_is_unavailable(monkeypatch: pytest.MonkeyPatch) -> None:
    _require_highs()
    highs = importlib.import_module("httk.analyse.generic._highs")
    calls = 0

    def unavailable(self: object, *args: object, **kwargs: object) -> None:
        nonlocal calls
        calls += 1

    monkeypatch.setattr(highs._HighsMixtureSolver, "solve", unavailable)
    simplex = LowerConvexHull([(0.0,), (1.0,), (0.5,)], [0.0, 0.0, 1.0])
    fallback = LowerConvexHull([(0.0,), (1.0,), (0.5,)], [0.0, 0.0, 1.0], solver="highs")

    assert calls > 0
    assert fallback.hull_indices == simplex.hull_indices
    assert fallback.value_above_hull == simplex.value_above_hull


@pytest.mark.parametrize("status", ["kInfeasible", "kUnbounded"])
def test_highs_rejects_nonoptimal_results(status: str) -> None:
    _require_highs()
    highs = importlib.import_module("httk.analyse.generic._highs")
    solver = highs._HighsMixtureSolver(((0.0,), (1.0,)), (0.0, 0.0))

    class NonoptimalModel:
        def changeColBounds(self, *args: object) -> object:
            return highs.highspy.HighsStatus.kOk

        def changeRowsBounds(self, *args: object) -> object:
            return highs.highspy.HighsStatus.kOk

        def run(self) -> object:
            return highs.highspy.HighsStatus.kOk

        def getModelStatus(self) -> object:
            return getattr(highs.highspy.HighsModelStatus, status)

    solver._model = NonoptimalModel()
    assert solver.solve((0, 1), (0.0,), (0.5,)) is None


def test_highs_rejects_an_unsafe_basis(monkeypatch: pytest.MonkeyPatch) -> None:
    _require_highs()
    highs = importlib.import_module("httk.analyse.generic._highs")
    solver = highs._HighsMixtureSolver(((0.0,), (1.0,)), (0.0, 0.0))
    monkeypatch.setattr(highs, "_basis_is_well_conditioned", lambda *args: False)

    assert solver.solve((0, 1), (0.0,), (0.5,)) is None


def test_invalid_highs_selection_is_validated_before_empty_input() -> None:
    with pytest.raises(ValueError, match="solver"):
        LowerConvexHull([], [], solver="other")  # type: ignore[arg-type]


def test_missing_highs_dependency_explains_the_extra(monkeypatch: pytest.MonkeyPatch) -> None:
    original = lower_hull.importlib.import_module

    def missing(name: str, package: str | None = None) -> object:
        if name == "._highs":
            raise ModuleNotFoundError("No module named 'highspy'", name="highspy")
        return original(name, package)

    monkeypatch.setattr(lower_hull.importlib, "import_module", missing)
    with pytest.raises(ImportError, match="httk-analyse\\[highs\\]"):
        LowerConvexHull([(0.0,)], [0.0], solver="highs")
