"""Measure generated LAMMPS parse/join/RDF and selected dynamics costs."""

import argparse
import json
import platform
import tempfile
import time
import tracemalloc
from importlib.metadata import version
from pathlib import Path

from httk.atomistic.integrations.lammps.trajectory import LammpsTrajectory

from httk.analyse.integrations.lammps import lammps_samples
from httk.analyse.integrations.trajectory import msd_from_trajectory, rdf_from_trajectory, vacf_from_trajectory


def _sources(directory: Path, frames: int, atoms: int) -> tuple[Path, Path]:
    dump, log = directory / "atoms.dump", directory / "log.lammps"
    side = 1
    while side**3 < atoms:
        side += 1
    box = max(40, 4 * (side + 2))
    with dump.open("w", encoding="utf-8") as stream:
        for step in range(frames):
            stream.write(
                f"ITEM: TIMESTEP\n{step}\nITEM: NUMBER OF ATOMS\n{atoms}\nITEM: BOX BOUNDS pp pp pp\n0 {box}\n0 {box}\n0 {box}\nITEM: ATOMS id type xu yu zu vx vy vz\n"
            )
            for atom in range(atoms):
                x = 1 + (atom % side) * 4 + step * 0.01
                y = 1 + ((atom // side) % side) * 4
                z = 1 + (atom // side**2) * 4
                stream.write(f"{atom + 1} 1 {x:.6f} {y:.6f} {z:.6f} 0.01 0 0\n")
    with log.open("w", encoding="utf-8") as stream:
        stream.write("units metal\nthermo_style custom step atoms temp pe etotal vol\nthermo_modify norm no\n")
        stream.write("Step Atoms Temp PotEng TotEng Volume\n")
        for step in range(frames):
            stream.write(f"{step} {atoms} 300 -1 -0.5 {box**3}\n")
        stream.write(f"Loop time of 0.1 on 1 procs for {frames} steps with {atoms} atoms\n")
    return dump, log


def _measure(callable_):
    tracemalloc.start()
    start = time.perf_counter()
    result = callable_()
    seconds = time.perf_counter() - start
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return {"seconds": seconds, "peak_python_bytes": peak, "result": result}


def main() -> None:
    """Generate source files and print timings with process and input metadata."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frames", type=int, default=12)
    parser.add_argument("--atoms", type=int, default=32)
    parser.add_argument("--max-lag", type=int, default=4)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.frames < 3 or args.atoms < 2 or not 1 <= args.max_lag < args.frames:
        parser.error("frames >=3, atoms >=2 and 1 <= max-lag < frames are required")
    with tempfile.TemporaryDirectory() as tmp:
        dump, log = _sources(Path(tmp), args.frames, args.atoms)

        def join_count():
            return sum(
                1
                for _ in lammps_samples(
                    dump,
                    log,
                    species={1: "Ar"},
                    units="metal",
                    dimension=3,
                    segment=0,
                    table_index=0,
                    columns=("Temp", "PotEng", "TotEng", "Volume"),
                    join="strict",
                    trajectory_options={"timestep": "0.001"},
                )
            )

        joined = _measure(join_count)
        trajectory = LammpsTrajectory(dump, species={1: "Ar"}, units="metal", segment=0, timestep="0.001")
        rdf = _measure(lambda: rdf_from_trajectory(trajectory, [0, 2, 4, 6, 8]).frame_count)
        msd = _measure(lambda: len(msd_from_trajectory(trajectory, max_lag=args.max_lag).times))
        vacf = _measure(lambda: len(vacf_from_trajectory(trajectory, max_lag=args.max_lag).times))
        report = {
            "command": "python benchmarks/run_materials_benchmarks.py",
            "python": platform.python_version(),
            "numpy_version": version("numpy"),
            "httk_analyse_version": version("httk-analyse"),
            "frames": args.frames,
            "atoms": args.atoms,
            "max_lag": args.max_lag,
            "dump_bytes": dump.stat().st_size,
            "log_bytes": log.stat().st_size,
            "join": joined,
            "rdf": rdf,
            "msd": msd,
            "vacf": vacf,
            "peak_memory_scope": "tracemalloc Python allocations only; native NumPy and OS buffers may be omitted",
        }
    output = json.dumps(report, sort_keys=True, indent=2) + "\n"
    if args.output:
        args.output.write_text(output, encoding="utf-8")
    else:
        print(output, end="")


if __name__ == "__main__":
    main()
