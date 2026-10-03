"""Portable analysis summaries with explicit source and numerical provenance."""

import json
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, is_dataclass
from importlib.metadata import version
from pathlib import Path
from typing import Any

from httk.core.digests import sha256_file

__all__ = ["AnalysisSummary", "analysis_summary"]


@dataclass(frozen=True, slots=True)
class AnalysisSummary:
    """Immutable canonical JSON summary, with no embedded source trajectories.

    :param canonical_json: JSON object containing analysis results and provenance.
    """

    canonical_json: str

    def __post_init__(self) -> None:
        """Validate JSON and store its canonical finite representation."""
        value = json.loads(self.canonical_json)
        if not isinstance(value, dict):
            raise ValueError("summary must be a JSON object")
        object.__setattr__(
            self, "canonical_json", json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
        )

    @property
    def value(self) -> dict[str, Any]:
        """Return a fresh decoded copy of the summary."""
        return json.loads(self.canonical_json)

    def write(self, path: str | Path) -> None:
        """Write canonical JSON and a newline to an explicitly selected file.

        :param path: Output file, replaced if it already exists.
        """
        Path(path).write_text(self.canonical_json + "\n", encoding="utf-8")


def analysis_summary(
    result: Any,
    *,
    algorithm: str,
    units: Mapping[str, str],
    parameters: Mapping[str, Any],
    selection: Mapping[str, Any],
    sources: Sequence[str | Path],
    assumptions: Sequence[str],
) -> AnalysisSummary:
    """Serialize a result with explicit units, parameters and hashed input files.

    Local source files are read in chunks only to calculate SHA-256; their bytes
    and frames are not embedded. Sources should be stable snapshots during
    analysis. Complex numbers use explicit real/imaginary JSON objects. Record
    the same selected files and parameters used to compute the supplied result.

    :param result: Frozen result dataclass or JSON-compatible numerical summary.
    :param algorithm: Fully qualified routine name or declared algorithm identifier.
    :param units: Unit for each physically dimensioned result field.
    :param parameters: Actual numerical/model options used for the calculation.
    :param selection: Frame ranges, fit windows or other explicit sample selection.
    :param sources: Source paths whose URLs, sizes and SHA-256 checksums are retained.
    :param assumptions: Scientific assumptions needed to interpret the result.
    :return: Immutable canonical JSON result and provenance envelope.
    :raises ValueError: If metadata is incomplete, nonfinite or unsupported.
    :raises OSError: If a source cannot be read.
    """
    if not isinstance(algorithm, str) or not algorithm.strip():
        raise ValueError("algorithm must be a nonempty identifier")
    if not isinstance(units, Mapping) or any(not isinstance(v, str) or not v.strip() for v in units.values()):
        raise ValueError("units must map field names to nonempty unit labels")
    if isinstance(assumptions, (str, bytes)) or any(not isinstance(v, str) for v in assumptions):
        raise ValueError("assumptions must be a sequence of strings")
    files = []
    for source in sources:
        path = Path(source).resolve()
        files.append(
            {"url": path.as_uri(), "name": path.name, "size": path.stat().st_size, "sha256": sha256_file(path)}
        )
    payload = {
        "algorithm": algorithm,
        "software": {"httk-analyse": version("httk-analyse"), "numpy": version("numpy")},
        "result_type": f"{type(result).__module__}.{type(result).__qualname__}",
        "result": _json_value(asdict(result) if is_dataclass(result) and not isinstance(result, type) else result),
        "units": dict(units),
        "parameters": _json_value(dict(parameters)),
        "selection": _json_value(dict(selection)),
        "sources": files,
        "assumptions": list(assumptions),
    }
    return AnalysisSummary(json.dumps(payload, sort_keys=True, allow_nan=False))


def _json_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        if any(not isinstance(key, str) for key in value):
            raise ValueError("JSON mappings require string keys")
        return {key: _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    if isinstance(value, complex):
        return {"real": value.real, "imaginary": value.imag}
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    raise ValueError(f"unsupported summary value {type(value).__name__}")
