"""Focused checks for the VASP analysis adapters."""

from fractions import Fraction

import pytest
from httk.atomistic.integrations.vasp.io.doscar import DOSCAR, DOSChannel
from httk.atomistic.wavefunction import PlaneWaveFunctions

from httk.analyse.integrations.vasp import VaspDOS, band_edges_from_wavefunctions, dos_from_vasp


def test_dos_adapter_keeps_nonmagnetic_total_and_sums_collinear_channels() -> None:
    nonmagnetic = DOSCAR(
        (Fraction(0), Fraction(1)),
        Fraction(1, 2),
        "nonmagnetic",
        (DOSChannel("total", (Fraction(2), Fraction(3)), (Fraction(4), Fraction(5))),),
    )
    presented = dos_from_vasp(nonmagnetic)
    assert presented.density == (2.0, 3.0)
    assert presented.spin_basis == "spin-summed"

    collinear = DOSCAR(
        (Fraction(0), Fraction(1)),
        Fraction(1, 2),
        "collinear",
        (
            DOSChannel("up", (Fraction(2), Fraction(3)), (Fraction(4), Fraction(5))),
            DOSChannel("down", (Fraction(7), Fraction(11)), (Fraction(13), Fraction(17))),
        ),
    )
    assert dos_from_vasp(collinear).density == (9.0, 14.0)
    assert dos_from_vasp(collinear, "up").integrated_density == (4.0, 5.0)
    assert dos_from_vasp(collinear, "down").spin_basis == "down"
    with pytest.raises(ValueError, match="no resolved"):
        dos_from_vasp(nonmagnetic, "up")


def test_vasp_dos_snapshots_mutable_input_sequences() -> None:
    energies = [0, 1]
    density = [-1, 2]
    integrated = [3, 4]
    dos = VaspDOS(energies, density, integrated, 0.5, "spin-summed", "total")
    energies.clear()
    density[0] = 8
    integrated.append(5)
    assert dos.energies == (0.0, 1.0)
    assert dos.density == (-1.0, 2.0)
    assert dos.integrated_density == (3.0, 4.0)


def _wavefunctions() -> PlaneWaveFunctions:
    return PlaneWaveFunctions(
        cell=[[4, 0, 0], [0, 4, 0], [0, 0, 4]],
        encut=100,
        kpoints=[[0, 0, 0], [0.25, 0, 0]],
        eigenvalues=[[[-1.0, 0.5], [-0.8, 0.3]], [[-2.0, 1.5], [-1.8, 1.3]]],
        occupations=[[[2, 0], [2, 0]], [[1, 0], [1, 0]]],
        coefficients={(spin, kpt, band): [1j] for spin in range(2) for kpt in range(2) for band in range(2)},
    )


def test_band_edge_adapter_selects_spin_and_preserves_spin_kpoint_band_axes() -> None:
    wavefunctions = _wavefunctions()
    first = band_edges_from_wavefunctions(
        wavefunctions, 0, max_occupation=2, occupancy_tolerance=1e-8, energy_reference=-0.2
    )
    second = band_edges_from_wavefunctions(
        wavefunctions, 1, max_occupation=1, occupancy_tolerance=1e-8, energy_reference=0.0
    )
    assert (first.vbm_band, first.vbm_kpoint) == (0, 1)
    assert (first.vbm_energy, first.indirect_gap) == pytest.approx((-0.6, 1.1))
    assert (second.vbm_energy, second.indirect_gap) == pytest.approx((-1.8, 3.1))
    with pytest.raises(ValueError, match="zero-based"):
        band_edges_from_wavefunctions(wavefunctions, 2, max_occupation=2, occupancy_tolerance=1e-8, energy_reference=0)
