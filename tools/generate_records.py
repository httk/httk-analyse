"""Regenerate (or with ``--check`` verify) ``src/httk/analyse/property_records.py``."""

import sys
from pathlib import Path

from httk.core import definition_ids as core
from httk.core._typed_record_tool import main

from httk.analyse import definitions

#: Every vendored analyse property definition IRI (derivation terms are not properties), sorted.
DEFINITION_IDS = sorted(
    iri for name in definitions.__all__ if name.isupper() and "/properties/" in (iri := getattr(definitions, name))
)
_LABELS = {
    definitions.MEAN: "mean",
    definitions.STANDARD_ERROR: "standard error",
    definitions.STANDARD_DEVIATION: "standard deviation",
    definitions.RMSE: "root-mean-square error",
    definitions.MAE: "mean absolute error",
    definitions.BIAS: "bias",
    definitions.MAXIMUM_ABSOLUTE_ERROR: "maximum absolute error",
}
_ERRORS = (definitions.RMSE, definitions.MAE, definitions.BIAS, definitions.MAXIMUM_ABSOLUTE_ERROR)
#: Every (base, derivation) statistic a binding emits with a fixed base definition, with its label, sorted.
#: Caller-chosen bases (``PropertyParity.definition``) are not listed; they stay ``DerivedDataRecord``.
DERIVED = sorted(
    (base, derivation, _LABELS[derivation])
    for base, derivation in (
        (core.TOTAL_ENERGY, definitions.RMSE),  # matsci/eos.py: EOS fit RMSE
        # matsci/mlip.py: energy, force and stress error statistics
        *(
            (base, d)
            for base in (definitions.TOTAL_ENERGY_PER_ATOM, core.ATOMIC_FORCE, core.STRESS_TENSOR)
            for d in _ERRORS
        ),
        # matsci/thermodynamics.py: block standard errors of the names equilibrium_response produces
        *(
            (base, definitions.STANDARD_ERROR)
            for base in (
                definitions.HEAT_CAPACITY_CONSTANT_VOLUME,
                definitions.HEAT_CAPACITY_CONSTANT_PRESSURE,
                definitions.ISOTHERMAL_COMPRESSIBILITY,
                definitions.VOLUMETRIC_THERMAL_EXPANSION,
            )
        ),
        # matsci/transport.py: replica mean and standard error at the selected lag
        (definitions.THERMAL_CONDUCTIVITY_TENSOR, definitions.MEAN),
        (definitions.THERMAL_CONDUCTIVITY, definitions.MEAN),
        (definitions.SHEAR_VISCOSITY, definitions.MEAN),
        (definitions.THERMAL_CONDUCTIVITY_TENSOR, definitions.STANDARD_ERROR),
    )
)
MODULE_DOC = """Generated typed result records for the httk-analyse property definitions.

Each class stores one value of its property definition in typed columns; see
:mod:`httk.core.typed_records`. ``RECORD_KINDS`` maps each definition IRI to its class.
"""
STORAGE_PREFIX = "analyse"
TARGET = Path(__file__).resolve().parents[1] / "src" / "httk" / "analyse" / "property_records.py"
COMMAND = "python tools/generate_records.py"

if __name__ == "__main__":
    sys.exit(
        main(
            sys.argv[1:],
            definition_ids=DEFINITION_IDS,
            derived=DERIVED,
            module_doc=MODULE_DOC,
            storage_prefix=STORAGE_PREFIX,
            target=TARGET,
            command=COMMAND,
        )
    )
