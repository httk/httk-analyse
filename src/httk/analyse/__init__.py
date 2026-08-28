"""Analysis capabilities for the httk namespace package."""

from httk.core import register_citation

register_citation(
    applies_to="Numerical analysis uses NumPy",
    references={
        "authors": (
            {"name": "Charles R. Harris"},
            {"name": "K. Jarrod Millman"},
            {"name": "Stéfan J. van der Walt"},
            {"name": "and others"},
        ),
        "title": "Array programming with NumPy",
        "journal": "Nature",
        "volume": "585",
        "pages": "357-362",
        "year": "2020",
        "doi": "10.1038/s41586-020-2649-2",
        "bib_type": "article",
    },
)

from . import crysviz, generic, matsci

__all__ = ["crysviz", "generic", "matsci"]
