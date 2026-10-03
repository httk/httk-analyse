"""Analyze an explicitly selected native LAMMPS dump and optional heat current."""

import argparse
import csv
import math
from dataclasses import asdict
from itertools import pairwise
from pathlib import Path
from typing import Any, cast

from httk.atomistic.integrations.lammps.trajectory import LammpsTrajectory

from httk.analyse.integrations.trajectory import msd_from_trajectory, rdf_from_trajectory, vacf_from_trajectory
from httk.analyse.matsci.transport import thermal_conductivity
from httk.analyse.summary import analysis_summary


def main() -> None:
    """Stream RDF and compute explicitly selected dynamics and transport."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dump", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--species", required=True, help="Atom type mapping, e.g. 1:Ar,2:Ne")
    parser.add_argument("--units", required=True, choices=("metal", "real"))
    parser.add_argument("--dimension", required=True, type=int, choices=(3,))
    parser.add_argument("--segment", required=True, type=int)
    parser.add_argument("--timestep", required=True, help="Positive source time per LAMMPS step")
    parser.add_argument("--cutoff", required=True, type=float, help="RDF cutoff in angstrom")
    parser.add_argument("--bin-spacing", required=True, type=float, help="RDF bin spacing in angstrom")
    parser.add_argument("--max-lag", required=True, type=int, help="Inclusive dynamic lag in frames")
    parser.add_argument("--msd", action="store_true", help="Require known unwrapped coordinates")
    parser.add_argument("--vacf", action="store_true", help="Require velocities")
    parser.add_argument(
        "--heat-current",
        type=Path,
        help="CSV time_ps,Jx_eV_angstrom_per_ps,Jy_eV_angstrom_per_ps,Jz_eV_angstrom_per_ps",
    )
    parser.add_argument("--temperature", type=float, help="Equilibrium K, required with heat current")
    parser.add_argument("--volume", type=float, help="Fixed angstrom^3, required with heat current")
    args = parser.parse_args()
    if (
        args.segment < 0
        or args.max_lag < 1
        or not math.isfinite(args.cutoff)
        or not math.isfinite(args.bin_spacing)
        or args.cutoff <= 0
        or args.bin_spacing <= 0
    ):
        parser.error("segment must be nonnegative; max-lag, cutoff and bin-spacing must be positive")
    try:
        species = {int(key): symbol for key, symbol in (part.split(":", 1) for part in args.species.split(","))}
    except ValueError as exc:
        parser.error(f"invalid species mapping: {exc}")
    if args.heat_current is None and (args.temperature is not None or args.volume is not None):
        parser.error("temperature and volume require --heat-current")
    if args.heat_current is not None and (args.temperature is None or args.volume is None):
        parser.error("heat current requires --temperature and --volume")
    count = math.ceil(args.cutoff / args.bin_spacing)
    bins = [min(i * args.bin_spacing, args.cutoff) for i in range(count + 1)]
    trajectory = LammpsTrajectory(
        args.dump, species=species, units=args.units, segment=args.segment, timestep=args.timestep
    )
    result: dict[str, Any] = {"rdf": rdf_from_trajectory(trajectory, bins)}
    if args.msd:
        result["msd"] = msd_from_trajectory(trajectory, max_lag=args.max_lag)
    if args.vacf:
        result["vacf"] = vacf_from_trajectory(trajectory, max_lag=args.max_lag)
    sources = [args.dump]
    if args.heat_current is not None:
        with args.heat_current.open(newline="", encoding="utf-8") as stream:
            rows = list(csv.DictReader(stream))
        names = ("Jx_eV_angstrom_per_ps", "Jy_eV_angstrom_per_ps", "Jz_eV_angstrom_per_ps")
        times = [float(row["time_ps"]) for row in rows]
        if len(times) <= args.max_lag or any(not math.isfinite(t) for t in times):
            raise ValueError("heat current requires more than max-lag finite samples")
        intervals = [b - a for a, b in pairwise(times)]
        if any(dt <= 0 or not math.isclose(dt, intervals[0], rel_tol=1e-10) for dt in intervals):
            raise ValueError("heat-current times must be uniformly increasing")
        current = [[float(row[name]) for name in names] for row in rows]
        result["thermal_conductivity"] = thermal_conductivity(
            current,
            intervals[0],
            temperature=cast(float, args.temperature),
            volume=cast(float, args.volume),
            max_lag=args.max_lag,
        )
        sources.append(args.heat_current)
    result_units = {
        "rdf.edges": "angstrom",
        "rdf.centers": "angstrom",
        "rdf.g": "1",
        "rdf.mean_coordination": "1",
    }
    if args.msd:
        result_units.update({"msd.times": "ps", "msd.tensors": "angstrom^2"})
    if args.vacf:
        result_units.update({"vacf.times": "ps", "vacf.tensors": "angstrom^2/ps^2"})
    if args.heat_current is not None:
        result_units.update(
            {
                "thermal_conductivity.times": "ps",
                "thermal_conductivity.correlations": "(eV*angstrom/ps)^2",
                "thermal_conductivity.integrals": "W/(m*K)",
                "thermal_conductivity.temperature": "K",
                "thermal_conductivity.volume": "angstrom^3",
            }
        )
    analysis_summary(
        {name: asdict(value) for name, value in result.items()},
        algorithm="httk.analyse.integrations.trajectory.rdf_from_trajectory+selected_dynamics",
        units=result_units,
        parameters={
            "species": {str(key): value for key, value in species.items()},
            "units": args.units,
            "dimension": args.dimension,
            "timestep_source": args.timestep,
            "cutoff_angstrom": args.cutoff,
            "bin_spacing_angstrom": args.bin_spacing,
            "max_lag_frames": args.max_lag,
            "heat_current_definition": "caller-declared extensive total microscopic current in eV*angstrom/ps"
            if args.heat_current
            else None,
            "temperature_K": args.temperature,
            "volume_angstrom3": args.volume,
        },
        selection={
            "dump_segment": args.segment,
            "msd": args.msd,
            "vacf": args.vacf,
            "rdf_frames": result["rdf"].frame_count,
        },
        sources=sources,
        assumptions=[
            "Periodic homogeneous three-dimensional bulk for RDF.",
            "Selected dynamic frames have a fixed cell, fixed atom order, regular physical time, and declared source units.",
            "Unwrapped coordinates and velocities are consumed only when their observables are present.",
            "Heat current, if supplied, has the declared extensive physical definition and stationary equilibrium sampling; no plateau is inferred.",
        ],
    ).write(args.output)


if __name__ == "__main__":
    main()
