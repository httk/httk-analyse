"""Print a selected LAMMPS dump/log join using canonical units."""

import argparse

from httk.analyse.integrations.lammps import lammps_samples


def main() -> None:
    """Run the file analysis command-line example."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dump")
    parser.add_argument("log")
    parser.add_argument("--species", required=True, help="Comma-separated type:element pairs, e.g. 1:Si,2:O")
    parser.add_argument("--units", required=True, choices=("metal", "real"))
    parser.add_argument("--dimension", required=True, type=int, choices=(3,))
    parser.add_argument("--segment", required=True, type=int)
    parser.add_argument("--table", required=True, type=int)
    parser.add_argument("--timestep", help="Constant timestep in source units, only if printed time is absent")
    args = parser.parse_args()
    species = {int(pair.split(":")[0]): pair.split(":")[1] for pair in args.species.split(",")}
    for sample in lammps_samples(
        args.dump,
        args.log,
        species=species,
        units=args.units,
        dimension=args.dimension,
        segment=args.segment,
        table_index=args.table,
        columns=("Temp", "PotEng", "TotEng", "Volume"),
        observables=("atom_ids",),
        trajectory_options={"timestep": args.timestep},
        join="strict",
    ):
        print(sample.step, {name: value for name, value, _ in sample.thermo})


if __name__ == "__main__":
    main()
