"""Open httk structures in the CrysViz viewer."""

import os
import tempfile
from pathlib import Path
from typing import Any

from httk.core import register_citation, save


def _import_crysviz() -> Any:
    try:
        import crysviz  # type: ignore[import-not-found]
    except ImportError as exc:
        raise ImportError("httk.analyse.crysviz requires crysviz; install httk-analyse[crysviz]") from exc
    return crysviz


def to_payload(
    structure: Any,
    *,
    name: str | None = None,
    format: str = "vasp-poscar",
) -> Any:
    """Serialize an httk structure as an in-memory CrysViz payload.

    :param structure: Structure to serialize.
    :param name: Optional filename for the payload, without a path.
    :param format: Serialization format, either ``"vasp-poscar"`` or ``"cif"``.
    :return: A CrysViz payload containing the serialized structure.
    :raises ImportError: If CrysViz is not installed.
    :raises ValueError: If ``format`` is not supported.
    """
    if format not in {"vasp-poscar", "cif"}:
        raise ValueError("format must be 'vasp-poscar' or 'cif'")

    crysviz = _import_crysviz()
    suffix = ".vasp" if format == "vasp-poscar" else ".cif"
    if not name:
        formula = getattr(structure, "formula", None)
        formula_text = str(formula) if formula is not None else ""
        filename = formula_text or "structure"
    else:
        filename = name
    if not filename.casefold().endswith(suffix):
        filename += suffix

    with tempfile.TemporaryDirectory() as directory:
        destination = Path(directory) / filename
        save(structure, destination, format=format)
        text = destination.read_text(encoding="utf-8")
    return crysviz.Payload(filename, text)


def show(*structures: Any, **viewer_kwargs: Any) -> Any:
    r"""Open structures in CrysViz and return when its window is ready.

    :param \*structures: CrysViz payloads, source paths, or httk structures to display.
    :param \**viewer_kwargs: Keyword arguments forwarded to ``crysviz.show``.
    :return: The ready CrysViz viewer, which also supports the context-manager protocol.
    :raises ImportError: If CrysViz is not installed.

    The call is non-blocking after the window is ready. Call ``viewer.wait()`` to
    block until the window closes.
    """
    crysviz = _import_crysviz()
    sources: list[Any] = []
    for structure in structures:
        if isinstance(structure, (crysviz.Payload, str, os.PathLike)):
            sources.append(structure)
        else:
            sources.append(to_payload(structure))

    register_citation(
        applies_to="Structure visualisation uses CrysViz",
        references={
            "authors": (
                {"name": "Florian Trybel"},
                {"name": "Abhijith S Parackal"},
                {"name": "Oscar Bulancea-Lindvall"},
                {"name": "Henricus R.A. ten Eikelder"},
                {"name": "Rickard Armiento"},
            ),
            "title": "CrysViz - Crystal Structure Visualisation & Analysis",
            "url": "https://github.com/CrysViz/crysviz",
            "year": "2026",
            "bib_type": "misc",
        },
    )
    return crysviz.show(sources, **viewer_kwargs)
