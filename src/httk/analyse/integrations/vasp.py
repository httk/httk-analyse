"""Analysis adapters for explicit VASP electronic-structure inputs."""

import math
from dataclasses import dataclass
from typing import Literal

from httk.atomistic.integrations.vasp.io.doscar import DOSCAR
from httk.atomistic.wavefunction import PlaneWaveFunctions

from httk.analyse.matsci.electronic import BandEdges, band_edges

__all__ = ["VaspDOS", "band_edges_from_wavefunctions", "dos_from_vasp"]


@dataclass(frozen=True, slots=True)
class VaspDOS:
    """Hold float-presented DOS data and its explicit spin basis.

    :param energies: Energy grid in eV.
    :param density: DOS in states/eV.
    :param integrated_density: Integrated DOS in states.
    :param fermi_energy: Fermi energy in eV.
    :param spin_basis: ``spin-summed`` or one resolved spin channel.
    :param spin: Requested output selection.
    """

    energies: tuple[float, ...]
    density: tuple[float, ...]
    integrated_density: tuple[float, ...]
    fermi_energy: float
    spin_basis: str
    spin: str

    def __post_init__(self) -> None:
        """Copy all sample arrays into immutable float tuples."""
        for name in ("energies", "density", "integrated_density"):
            object.__setattr__(self, name, tuple(float(value) for value in getattr(self, name)))
        object.__setattr__(self, "fermi_energy", float(self.fermi_energy))


def dos_from_vasp(payload: DOSCAR, spin: Literal["total", "up", "down"] = "total") -> VaspDOS:
    """Present exact DOSCAR values as floats on an explicit spin basis.

    Collinear ``total`` sums the up and down channels. Nonmagnetic total DOS
    is returned unchanged because it already represents the spin-summed DOS.

    :param payload: Parsed total-DOS data from ``read_doscar``.
    :param spin: Total, up, or down channel selection.
    :return: Float analysis data with spin-basis metadata.
    :raises ValueError: If the selection is unavailable or values overflow float.
    """
    if not isinstance(payload, DOSCAR):
        raise TypeError("payload must be a parsed DOSCAR")
    if spin not in {"total", "up", "down"}:
        raise ValueError("spin must be 'total', 'up', or 'down'")
    if payload.spin_mode == "nonmagnetic":
        if spin != "total":
            raise ValueError("nonmagnetic DOSCAR has no resolved spin channels")
        density = payload.channels[0].density
        integrated = payload.channels[0].integrated_density
        basis = "spin-summed"
    elif payload.spin_mode == "collinear" and len(payload.channels) == 2 and spin == "total":
        up, down = payload.channels
        density = tuple(a + b for a, b in zip(up.density, down.density, strict=True))
        integrated = tuple(a + b for a, b in zip(up.integrated_density, down.integrated_density, strict=True))
        basis = "spin-summed"
    elif payload.spin_mode == "collinear" and len(payload.channels) == 2:
        channel = payload.channels[0 if spin == "up" else 1]
        density, integrated, basis = channel.density, channel.integrated_density, spin
    else:
        raise ValueError("payload must contain a valid nonmagnetic or collinear DOSCAR")
    try:
        values = tuple(float(value) for value in (*payload.energies, *density, *integrated, payload.fermi_energy))
    except OverflowError as exc:
        raise ValueError("DOSCAR values exceed finite float range") from exc
    if not all(math.isfinite(value) for value in values):
        raise ValueError("DOSCAR values exceed finite float range")
    n = len(payload.energies)
    return VaspDOS(values[:n], values[n : 2 * n], values[2 * n : 3 * n], values[-1], basis, spin)


def band_edges_from_wavefunctions(
    wavefunctions: PlaneWaveFunctions,
    spin: int,
    *,
    max_occupation: float,
    occupancy_tolerance: float,
    energy_reference: float,
) -> BandEdges:
    """Analyze one WAVECAR spin channel using its stored eigenvalues and occupations.

    :param wavefunctions: Existing plane-wave data with axes spin, k point, band.
    :param spin: Zero-based spin index.
    :param max_occupation: Explicit maximum occupancy of a state.
    :param occupancy_tolerance: Explicit absolute empty/full endpoint tolerance.
    :param energy_reference: Explicit energy reference to subtract, in eV.
    :return: Sampled band edges from :func:`~httk.analyse.matsci.electronic.band_edges`.
    :raises ValueError: If the input or spin index is invalid.
    """
    if not isinstance(wavefunctions, PlaneWaveFunctions):
        raise TypeError("wavefunctions must be a PlaneWaveFunctions instance")
    if isinstance(spin, bool) or not isinstance(spin, int) or not 0 <= spin < wavefunctions.nspins:
        raise ValueError(f"spin must be a zero-based index in [0, {wavefunctions.nspins})")
    return band_edges(
        wavefunctions.eigenvalues[spin].T,
        wavefunctions.occupations[spin].T,
        maximum_occupation=max_occupation,
        occupation_tolerance=occupancy_tolerance,
        energy_reference=energy_reference,
    )
