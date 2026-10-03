"""Synthetic command-line checks for analysis and SQLite provenance recipes."""

import csv
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

EXAMPLES = Path(__file__).resolve().parents[1] / "examples"


def _command(name: str, *args: object) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(EXAMPLES / name), *(str(arg) for arg in args)],
        check=True,
        capture_output=True,
        text=True,
    )


def _csv(path: Path, fieldnames: tuple[str, ...], rows: list[tuple[object, ...]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(fieldnames)
        writer.writerows(rows)


def test_volume_scan(tmp_path: Path) -> None:
    source = tmp_path / "scan.csv"
    _csv(source, ("volume_angstrom3", "energy_eV"), [(v, 0.01 * (v - 10) ** 2) for v in (8, 9, 10, 11, 12, 13)])
    summary = tmp_path / "fit.json"
    _command("volume_scan.py", source, summary)
    value = json.loads(summary.read_text(encoding="utf-8"))
    assert value["result"]["equilibrium_volume"] == pytest.approx(10, abs=0.1)
    assert value["sources"][0]["name"] == "scan.csv"


def test_real_sqlite_reopen(tmp_path: Path) -> None:
    if importlib.util.find_spec("httk.store") is None or importlib.util.find_spec("sqlalchemy") is None:
        pytest.skip("optional httk-store[db] is not installed")
    source = tmp_path / "scan.csv"
    _csv(source, ("volume_angstrom3", "energy_eV"), [(v, 0.01 * (v - 10) ** 2) for v in (8, 9, 10, 11, 12, 13)])
    summary = tmp_path / "fit.json"
    _command("volume_scan.py", source, summary)
    stored = json.loads(_command("store_analysis.py", summary, tmp_path / "analysis.sqlite").stdout)
    assert stored["record_id"] and stored["run_id"] and len(stored["file_ids"]) == 1
    linked = json.loads(
        _command(
            "store_analysis.py",
            summary,
            tmp_path / "analysis.sqlite",
            "--source-id",
            "workspace:job",
            "--upstream-run-id",
            stored["run_id"],
        ).stdout
    )
    assert linked["record_id"] == stored["record_id"]
    assert linked["run_id"] != stored["run_id"]


def test_md_recipe_streams_rdf_and_selects_unwrapped_dynamics(tmp_path: Path) -> None:
    dump = tmp_path / "atoms.dump"
    dump.write_text(
        "".join(
            f"ITEM: TIMESTEP\n{step}\nITEM: NUMBER OF ATOMS\n2\n"
            "ITEM: BOX BOUNDS pp pp pp\n0 8\n0 8\n0 8\n"
            "ITEM: ATOMS id type xu yu zu vx vy vz\n"
            f"1 1 {step} 0 0 1 0 0\n2 1 {step + 1} 0 0 1 0 0\n"
            for step in range(4)
        ),
        encoding="utf-8",
    )
    current = tmp_path / "current.csv"
    _csv(
        current,
        ("time_ps", "Jx_eV_angstrom_per_ps", "Jy_eV_angstrom_per_ps", "Jz_eV_angstrom_per_ps"),
        [(i, i % 2, 0, 0) for i in range(4)],
    )
    summary = tmp_path / "md.json"
    _command(
        "md_properties.py",
        dump,
        summary,
        "--species",
        "1:Ar",
        "--units",
        "metal",
        "--dimension",
        3,
        "--segment",
        0,
        "--timestep",
        1,
        "--cutoff",
        3,
        "--bin-spacing",
        1,
        "--max-lag",
        2,
        "--msd",
        "--vacf",
        "--heat-current",
        current,
        "--temperature",
        300,
        "--volume",
        512,
    )
    value = json.loads(summary.read_text(encoding="utf-8"))
    assert value["result"]["rdf"]["frame_count"] == 4
    assert value["units"]["rdf.edges"] == "angstrom"
    assert value["units"]["msd.tensors"] == "angstrom^2"
    assert value["units"]["thermal_conductivity.times"] == "ps"
    assert [tensor[0][0] for tensor in value["result"]["msd"]["tensors"]] == [0, 1, 4]
    assert [tensor[0][0] for tensor in value["result"]["vacf"]["tensors"]] == [1, 1, 1]
    assert len(value["sources"]) == 2


def test_mlip_recipe_compares_separate_curves_and_reports_checks(tmp_path: Path) -> None:
    reference, prediction, nve, committee = (
        tmp_path / name for name in ("reference.csv", "prediction.csv", "nve.csv", "committee.csv")
    )
    volumes = (8, 9, 10, 11, 12, 13)
    fields = ("volume_angstrom3", "energy_eV", "composition", "protocol", "energy_basis")
    _csv(reference, fields, [(v, 0.01 * (v - 10) ** 2, "Ar2", "fixed-cell", "total") for v in volumes])
    _csv(prediction, fields, [(v, 0.011 * (v - 10.1) ** 2, "Ar2", "fixed-cell", "total") for v in volumes])
    _csv(nve, ("time_ps", "total_energy_eV"), [(0, -2), (1, -1.99), (2, -1.98)])
    _csv(committee, ("prediction_1", "prediction_2"), [(1, 1.1), (2, 2.1)])
    summary = tmp_path / "mlip.json"
    _command(
        "mlip_properties.py",
        reference,
        prediction,
        nve,
        committee,
        summary,
        "--atom-count",
        2,
        "--spring-constant",
        3,
        "--displacement",
        0.01,
        "--holdout-group",
        "material-family-A",
    )
    value = json.loads(summary.read_text(encoding="utf-8"))
    assert value["selection"]["holdout_group"] == "material-family-A"
    assert value["selection"]["matched_volume_grid_angstrom3"] == list(volumes)
    assert value["parameters"]["matched_eos_metadata"]["composition"] == "Ar2"
    assert value["units"]["nve_drift.intercept"] == "eV/atom"
    assert value["units"]["eos_parity.bulk_modulus.residuals"] == "eV/angstrom^3"
    assert len(value["result"]["harmonic_force_checks"]) == 2
    assert value["result"]["nve_drift"]["slope"] == pytest.approx(0.005)
    assert len(value["sources"]) == 4
