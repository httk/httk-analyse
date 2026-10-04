"""Compare held-out EOS curves and explicit model validation inputs."""

import argparse
import csv
import math
from dataclasses import asdict
from pathlib import Path

import numpy as np
from httk.core.definition_ids import TOTAL_ENERGY

from httk.analyse.definitions import BULK_MODULUS, EQUILIBRIUM_VOLUME
from httk.analyse.matsci.eos import fit_birch_murnaghan
from httk.analyse.matsci.validation import committee_spread, energy_drift, force_energy_consistency, property_parity
from httk.analyse.summary import analysis_summary


def _rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def main() -> None:
    """Write parity, derivative, NVE and committee summaries with source hashes."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reference_csv", type=Path, help="volume_angstrom3,energy_eV,composition,protocol,energy_basis")
    parser.add_argument(
        "prediction_csv", type=Path, help="volume_angstrom3,energy_eV,composition,protocol,energy_basis"
    )
    parser.add_argument("nve_csv", type=Path, help="time_ps,total_energy_eV")
    parser.add_argument("committee_csv", type=Path, help="prediction_1,... prediction_N columns")
    parser.add_argument("output", type=Path)
    parser.add_argument("--atom-count", type=int, required=True)
    parser.add_argument(
        "--spring-constant", type=float, required=True, help="Synthetic harmonic callback in eV/angstrom^2"
    )
    parser.add_argument(
        "--displacement",
        type=float,
        required=True,
        help="Larger finite-difference displacement in angstrom; second is half",
    )
    parser.add_argument(
        "--holdout-group", required=True, help="Material family or independent trajectory withheld from training"
    )
    args = parser.parse_args()
    if (
        not math.isfinite(args.spring_constant)
        or args.spring_constant <= 0
        or not math.isfinite(args.displacement)
        or args.displacement <= 0
    ):
        parser.error("spring constant and displacement must be finite positive values")
    reference, predicted = _rows(args.reference_csv), _rows(args.prediction_csv)
    if len(reference) != len(predicted) or not reference:
        raise ValueError("EOS curves require matching nonempty rows")
    rv = [float(row["volume_angstrom3"]) for row in reference]
    pv = [float(row["volume_angstrom3"]) for row in predicted]
    if rv != pv:
        raise ValueError("reference and prediction volumes must match in row order")
    metadata = {name: reference[0][name] for name in ("composition", "protocol", "energy_basis")}
    if any(
        not value or any(row[name] != value for row in (*reference, *predicted)) for name, value in metadata.items()
    ):
        raise ValueError("both EOS curves must declare identical nonempty composition, protocol and energy_basis")
    ref_fit = fit_birch_murnaghan(rv, [float(row["energy_eV"]) for row in reference])
    pred_fit = fit_birch_murnaghan(pv, [float(row["energy_eV"]) for row in predicted])
    parity = {
        "equilibrium_volume": property_parity(
            [ref_fit.equilibrium_volume],
            [pred_fit.equilibrium_volume],
            labels=[args.holdout_group],
            definition=EQUILIBRIUM_VOLUME,
        ),
        "bulk_modulus": property_parity(
            [ref_fit.bulk_modulus], [pred_fit.bulk_modulus], labels=[args.holdout_group], definition=BULK_MODULUS
        ),
    }
    position = np.array([[0.2, 0.0, 0.0]])
    force = -args.spring_constant * position

    def harmonic(positions: np.ndarray) -> float:
        return float(args.spring_constant * np.sum(positions**2) / 2)

    checks = [
        force_energy_consistency(harmonic, position, force, displacement=step)
        for step in (args.displacement, args.displacement / 2)
    ]
    nve = _rows(args.nve_csv)
    drift = energy_drift(
        [float(row["time_ps"]) for row in nve],
        [float(row["total_energy_eV"]) for row in nve],
        atom_count=args.atom_count,
        ensemble="NVE",
    )
    committee = _rows(args.committee_csv)
    columns = sorted(name for name in committee[0] if name.startswith("prediction_")) if committee else []
    if len(columns) < 2:
        raise ValueError("committee CSV needs at least prediction_1 and prediction_2")
    spread = committee_spread([[float(row[name]) for row in committee] for name in columns], definition=TOTAL_ENERGY)
    result = {
        "reference_eos": asdict(ref_fit),
        "prediction_eos": asdict(pred_fit),
        "eos_parity": {name: asdict(value) for name, value in parity.items()},
        "harmonic_force_checks": [asdict(value) for value in checks],
        "nve_drift": asdict(drift),
        "committee_spread": asdict(spread),
    }
    # Composite summary: nested fields take OPTIMADE unit expressions. Parity and committee
    # results also carry their property-definition IRI in their ``definition`` field.
    units = {
        "nve_drift.slope": "eV*ps^-1",
        "nve_drift.intercept": "eV",
        "nve_drift.residual_rms": "eV",
        "nve_drift.endpoint_change": "eV",
        "nve_drift.start": "ps",
        "nve_drift.stop": "ps",
        "committee_spread.mean": "eV",
        "committee_spread.standard_deviation": "eV",
        "selection.matched_volume_grid_angstrom3": "angstrom^3",
    }
    for prefix in ("reference_eos", "prediction_eos"):
        units.update(
            {
                f"{prefix}.volumes": "angstrom^3",
                f"{prefix}.energies": "eV",
                f"{prefix}.equilibrium_volume": "angstrom^3",
                f"{prefix}.equilibrium_energy": "eV",
                f"{prefix}.bulk_modulus": "GPa",
                f"{prefix}.bulk_modulus_derivative": "dimensionless",
                f"{prefix}.residuals": "eV",
                f"{prefix}.rmse": "eV",
            }
        )
    for name, unit in (("equilibrium_volume", "angstrom^3"), ("bulk_modulus", "GPa")):
        prefix = f"eos_parity.{name}"
        units.update({f"{prefix}.{field}": unit for field in ("reference", "predicted", "residuals")})
        units.update(
            {
                f"{prefix}.statistics.{field}": unit
                for field in ("bias", "mae", "rmse", "maximum_absolute_error", "percentile95_absolute_error")
            }
        )
    for index in range(2):
        prefix = f"harmonic_force_checks[{index}]"
        units.update({f"{prefix}.{field}": "angstrom^-1*eV" for field in ("reference", "predicted", "residuals")})
        units.update(
            {
                f"{prefix}.statistics.{field}": "angstrom^-1*eV"
                for field in ("bias", "mae", "rmse", "maximum_absolute_error", "percentile95_absolute_error")
            }
        )
    analysis_summary(
        result,
        algorithm="httk.analyse.matsci.validation.property_parity+force_energy_consistency+energy_drift+committee_spread",
        units=units,
        parameters={
            "eos_model": "algebraic-bm3",
            "matched_eos_metadata": metadata,
            "spring_constant_eV_per_angstrom2": args.spring_constant,
            "displacements_angstrom": [args.displacement, args.displacement / 2],
            "atom_count": args.atom_count,
            "nve_drift_energy_basis": "per atom",
            "ensemble": "NVE",
            "committee_columns": columns,
        },
        selection={
            "holdout_group": args.holdout_group,
            "matched_volume_grid_angstrom3": rv,
            "eos_rows": len(reference),
            "nve_samples": len(nve),
            "committee_samples": len(committee),
        },
        sources=[args.reference_csv, args.prediction_csv, args.nve_csv, args.committee_csv],
        assumptions=[
            "Reference and prediction curves represent the same composition, electronic state and volume grid.",
            "Holdout group is an independent material family or trajectory absent from model training.",
            "Harmonic force callback demonstrates finite-difference sensitivity; it is not a test of an external MLIP.",
            "NVE total energies are on a fixed atom-count basis.",
            "Committee spread is uncalibrated model disagreement.",
        ],
    ).write(args.output)


if __name__ == "__main__":
    main()
