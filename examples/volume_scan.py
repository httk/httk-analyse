"""Fit an explicitly unit-labelled static volume scan from CSV."""

import argparse
import csv
from pathlib import Path

from httk.analyse.matsci.eos import fit_birch_murnaghan
from httk.analyse.matsci.eos_models import fit_eos
from httk.analyse.summary import analysis_summary


def main() -> None:
    """Fit a selected EOS and write its result and source digest."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv", type=Path, help="CSV with volume_angstrom3,energy_eV columns")
    parser.add_argument("output", type=Path)
    parser.add_argument(
        "--model", choices=("algebraic-bm3", "birch-murnaghan", "murnaghan", "vinet"), default="algebraic-bm3"
    )
    args = parser.parse_args()
    with args.csv.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    if not rows or any(row.get("volume_angstrom3") in (None, "") or row.get("energy_eV") in (None, "") for row in rows):
        raise ValueError("CSV needs nonempty volume_angstrom3 and energy_eV columns")
    volumes = [float(row["volume_angstrom3"]) for row in rows]
    energies = [float(row["energy_eV"]) for row in rows]
    fit = (
        fit_birch_murnaghan(volumes, energies)
        if args.model == "algebraic-bm3"
        else fit_eos(volumes, energies, model=args.model)
    )
    analysis_summary(
        fit,
        algorithm=f"httk.analyse.matsci.{'eos.fit_birch_murnaghan' if args.model == 'algebraic-bm3' else 'eos_models.fit_eos'}",
        # Fitted parameters carry property definitions (see ``fields``); only the unbound
        # input/diagnostic series need OPTIMADE unit expressions.
        units={
            "volumes": "angstrom^3",
            "energies": "eV",
            "residuals": "eV",
            "condition_number": "dimensionless",
            **({"weighted_rmse": "eV", "weights": "dimensionless"} if args.model != "algebraic-bm3" else {}),
        },
        parameters={"model": args.model, "energy_basis": "total energy per structure"},
        selection={"rows": len(rows), "order": "CSV row order"},
        sources=[args.csv],
        assumptions=[
            "All rows have the same composition, electronic state, relaxation protocol and extensive energy basis.",
            "Fitted minimum must lie inside the sampled volume range.",
        ],
    ).write(args.output)


if __name__ == "__main__":
    main()
