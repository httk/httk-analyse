"""Focused checks for the isolated phase-diagram benchmark worker."""

import json

import pytest
from benchmarks import _phase_diagram_worker as worker


def _request(metric: str = "httk") -> dict[str, object]:
    return {
        "case": {"sweep": "tiny", "species": 2, "phases": 3, "seed": 17},
        "metric": metric,
        "atom_count": 2,
        "warmups": 0,
    }


def test_validate_case_rejects_impossible_unique_compositions() -> None:
    with pytest.raises(ValueError, match="unique compositions"):
        worker.validate_case({"sweep": "tiny", "species": 2, "phases": 4, "seed": 1}, 2)


def test_httk_summary_validates_duplicate_coplanar_and_near_degenerate_inputs(monkeypatch: pytest.MonkeyPatch) -> None:
    compositions = (
        {"H": 4},
        {"He": 4},
        {"H": 2, "He": 2},
        {"H": 2, "He": 2},
        {"H": 2, "He": 2},
    )
    energies = (0.0, 0.0, -2.0, -2.0 + 1e-9, -1.5)
    monkeypatch.setattr(worker, "_dataset", lambda case, atom_count: (compositions, energies))
    request = _request()
    request["case"] = {"sweep": "tiny", "species": 2, "phases": 5, "seed": 17}
    request["atom_count"] = 4

    seconds, summary = worker._run_request(request)

    assert seconds >= 0.0
    assert summary["stable_indices"] == [0, 1, 2, 3]
    assert summary["max_composition_residual"] == pytest.approx(0.0)
    assert summary["max_weight_sum_error"] == pytest.approx(0.0)
    assert summary["max_energy_residual"] <= 1e-8
    assert summary["min_weight"] > 0.0


def test_phase_lines_summary_is_cold_query_only() -> None:
    seconds, summary = worker._run_httk(
        "httk_phase_lines",
        ({"H": 2}, {"He": 2}, {"H": 1, "He": 1}),
        (0.0, 0.0, -2.0),
        0,
    )

    assert seconds >= 0.0
    assert isinstance(summary["phase_lines"], int)
    stable = summary["stable_indices"]
    assert summary["lp_solves_estimate"] == len(stable) * (len(stable) - 1) // 2


def test_invalid_request_writes_strict_json_response(tmp_path) -> None:
    request_path = tmp_path / "request.json"
    response_path = tmp_path / "response.json"
    request = _request()
    request["atom_count"] = 0
    request_path.write_text(json.dumps(request), encoding="utf-8")

    assert worker.main((str(request_path), str(response_path))) == 1

    response = json.loads(response_path.read_text(encoding="utf-8"))
    assert response["status"] == "error"
    assert response["seconds"] is None
    assert "positive integer" in response["error"]


def test_ase_full_summary_uses_atomic_fraction_weights() -> None:
    pytest.importorskip("ase")

    compositions = ({"H": 2}, {"He": 3}, {"H": 1, "He": 1}, {"H": 1, "He": 3})
    energies = (0.0, 0.0, -2.0, -1.0)
    httk_seconds, httk_summary = worker._run_httk("httk", compositions, energies, 0)
    seconds, summary = worker._run_ase("ase_full", compositions, energies, 0)

    assert seconds >= 0.0
    assert httk_seconds >= 0.0
    assert summary["stable_indices"] == httk_summary["stable_indices"]
    assert summary["hull_energies"] == pytest.approx(httk_summary["hull_energies"])
    assert summary["max_composition_residual"] <= 1e-7
    assert summary["max_energy_residual"] <= 1e-7
    assert summary["max_weight_sum_error"] <= 1e-7
    assert summary["min_weight"] > 0.0
    assert isinstance(summary["lower_facets"], int)
