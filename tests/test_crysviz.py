"""Tests for the optional CrysViz structure-viewer integration."""

from fractions import Fraction as F
from pathlib import Path
from typing import Any

import crysviz as crysviz_package
import httk.core
import pytest
from httk.analyse import crysviz
from httk.atomistic import Cell, Sites, Species, UnitcellStructure


@pytest.fixture
def si_structure() -> UnitcellStructure:
    return UnitcellStructure(
        Cell([[F("5.43"), 0, 0], [0, F("5.43"), 0], [0, 0, F("5.43")]]),
        Sites([[0, 0, 0], [F(1, 4), F(1, 4), F(1, 4)]]),
        (Species("Si", ("Si",), (1,)),),
        ("Si", "Si"),
    )


def test_to_payload_writes_a_loadable_poscar(si_structure: UnitcellStructure, tmp_path: Path) -> None:
    payload = crysviz.to_payload(si_structure)

    assert isinstance(payload, crysviz_package.Payload)
    assert payload.name.endswith(".vasp")
    poscar = tmp_path / "POSCAR"
    poscar.write_text(payload.data, encoding="utf-8")
    loaded = httk.core.load(poscar, raw=True)
    assert loaded["format"] == "vasp-poscar"
    assert loaded["symbols"] == ["Si"]
    assert loaded["counts"] == [2]


def test_to_payload_can_write_cif(si_structure: UnitcellStructure) -> None:
    payload = crysviz.to_payload(si_structure, format="cif", name="si")

    assert payload.name == "si.cif"
    assert "_cell_length_a" in payload.data


def test_to_payload_rejects_unsupported_formats(si_structure: UnitcellStructure) -> None:
    with pytest.raises(ValueError):
        crysviz.to_payload(si_structure, format="xyz")


def test_show_converts_structures_and_forwards_sources(
    si_structure: UnitcellStructure, monkeypatch: pytest.MonkeyPatch
) -> None:
    recorded: dict[str, Any] = {}
    sentinel = object()

    def fake_show(sources: list[Any], **kwargs: Any) -> object:
        recorded["sources"] = sources
        recorded["kwargs"] = kwargs
        return sentinel

    monkeypatch.setattr(crysviz_package, "show", fake_show)

    result = crysviz.show(si_structure, "some/path.cif", command_timeout=3)

    assert result is sentinel
    assert isinstance(recorded["sources"][0], crysviz_package.Payload)
    assert recorded["sources"][1] == "some/path.cif"
    assert recorded["kwargs"] == {"command_timeout": 3}
    assert "Structure visualisation uses CrysViz" in httk.core.credits.entries()
