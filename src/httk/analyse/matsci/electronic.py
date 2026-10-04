"""Electronic, magnetic and static dielectric analysis for sampled data."""

import math
from collections.abc import Sequence
from dataclasses import dataclass
from itertools import pairwise
from typing import Any

import numpy as np

from .. import definitions as defs
from ..definitions import BoundValue, FieldBinding, _reject_selection

__all__ = [
    "BandEdges",
    "DielectricSummary",
    "EffectiveMassFit",
    "MagneticMoments",
    "align_energies",
    "band_edges",
    "electron_count",
    "fit_effective_mass",
    "integrate_dos",
    "magnetic_moments",
    "solve_chemical_potential",
    "summarize_dielectric",
]

_KB_EV_K = 8.617333262145e-5
# CODATA 2022 constants give hbar**2 / m_e = 7.619964 eV angstrom**2.
_HBAR2_OVER_ME_EV_A2 = 7.619964231073853
_GAUSS_X, _GAUSS_W = np.polynomial.legendre.leggauss(16)


@dataclass(frozen=True, slots=True)
class BandEdges:
    """Band extrema on the supplied sampled k points, with energies in eV.

    :param metallic: Whether any state is partially occupied by the supplied criterion.
    :param indirect_gap: Sampled conduction minimum minus valence maximum in eV.
    :param direct_gap: Smallest sampled same-k conduction-valence separation in eV.
    :param vbm_energy: Valence-band maximum in the aligned reference, in eV.
    :param cbm_energy: Conduction-band minimum in the aligned reference, in eV.
    :param vbm_band: Input band index of the valence maximum.
    :param vbm_kpoint: Input k-point index of the valence maximum.
    :param cbm_band: Input band index of the conduction minimum.
    :param cbm_kpoint: Input k-point index of the conduction minimum.
    :param energy_reference: Subtracted reference energy in eV.
    """

    metallic: bool
    indirect_gap: float
    direct_gap: float
    vbm_energy: float
    cbm_energy: float
    vbm_band: int
    vbm_kpoint: int
    cbm_band: int
    cbm_kpoint: int
    energy_reference: float

    def _bound_values(self, **selection: Any) -> tuple[BoundValue, ...]:
        """Bind the gaps, reported as 0.0 (the definitions' metallic value) when ``metallic``."""
        _reject_selection(self, selection)
        return (
            BoundValue("indirect_gap", FieldBinding(defs.BAND_GAP), 0.0 if self.metallic else float(self.indirect_gap)),
            BoundValue(
                "direct_gap", FieldBinding(defs.DIRECT_BAND_GAP), 0.0 if self.metallic else float(self.direct_gap)
            ),
        )


@dataclass(frozen=True, slots=True)
class EffectiveMassFit:
    """Full signed effective-mass tensor from a local quadratic band fit.

    :param center: Cartesian fit center in inverse angstroms, with 2π included.
    :param energy_at_center: Fitted band energy at the center in eV.
    :param gradient: Local energy gradient in eV angstroms.
    :param hessian: Energy Hessian in eV angstroms squared.
    :param mass_tensor: Effective mass tensor in electron-mass units.
    :param eigenvalues: Signed principal masses in electron-mass units.
    :param principal_axes: Corresponding Cartesian unit eigenvectors by column.
    :param rank: Numerical rank of the quadratic design.
    :param condition_number: Singular-value condition number of the scaled design.
    :param residual_rms: Root mean square energy residual in eV.
    :param residuals: Input-order observed-minus-fitted energies in eV.
    """

    center: tuple[float, float, float]
    energy_at_center: float
    gradient: tuple[float, float, float]
    hessian: tuple[tuple[float, ...], ...]
    mass_tensor: tuple[tuple[float, ...], ...]
    eigenvalues: tuple[float, float, float]
    principal_axes: tuple[tuple[float, ...], ...]
    rank: int
    condition_number: float
    residual_rms: float
    residuals: tuple[float, ...]

    def __post_init__(self) -> None:
        """Copy all array-like fields into immutable tuples."""
        for name in ("center", "gradient", "eigenvalues", "residuals"):
            object.__setattr__(self, name, tuple(float(x) for x in getattr(self, name)))
        for name in ("hessian", "mass_tensor", "principal_axes"):
            object.__setattr__(self, name, tuple(tuple(float(x) for x in row) for row in getattr(self, name)))


@dataclass(frozen=True, slots=True)
class MagneticMoments:
    """Total vector moment and optional pair spin correlations.

    :param total: Sum of input vector moments in Bohr magnetons.
    :param pairs: Input-order atom-index pairs.
    :param correlations: Pair dot products in squared Bohr magnetons, or dimensionless if normalized.
    :param normalized: Whether each nonzero moment was normalized before its pair dot product.
    """

    total: tuple[float, float, float]
    pairs: tuple[tuple[int, int], ...]
    correlations: tuple[float, ...]
    normalized: bool

    def __post_init__(self) -> None:
        """Copy sequence fields into immutable tuples."""
        object.__setattr__(self, "total", tuple(float(x) for x in self.total))
        object.__setattr__(self, "pairs", tuple(tuple(int(x) for x in pair) for pair in self.pairs))
        object.__setattr__(self, "correlations", tuple(float(x) for x in self.correlations))


@dataclass(frozen=True, slots=True)
class DielectricSummary:
    """Principal values and axes of a real static dielectric tensor.

    :param eigenvalues: Principal relative dielectric constants, ascending.
    :param principal_axes: Corresponding Cartesian unit eigenvectors by column.
    :param mean: Isotropic mean of the principal relative dielectric constants.
    """

    eigenvalues: tuple[float, float, float]
    principal_axes: tuple[tuple[float, ...], ...]
    mean: float

    def __post_init__(self) -> None:
        """Copy array-like fields into immutable tuples."""
        object.__setattr__(self, "eigenvalues", tuple(float(x) for x in self.eigenvalues))
        object.__setattr__(self, "principal_axes", tuple(tuple(float(x) for x in row) for row in self.principal_axes))


def integrate_dos(
    energies: Sequence[float],
    density: Sequence[float],
    lower: float,
    upper: float,
    *,
    spin_degeneracy: float,
) -> float:
    """Integrate a piecewise-linear DOS between energy bounds.

    DOS values are states/eV for one spin channel unless the caller supplies a
    different convention through ``spin_degeneracy``. Bounds outside the grid
    are clipped to its endpoints and in-grid bounds are linearly interpolated.

    :param energies: Strictly increasing energy grid in eV.
    :param density: Nonnegative DOS values in states/eV on the declared spin basis.
    :param lower: Lower integration bound in eV.
    :param upper: Upper integration bound in eV.
    :param spin_degeneracy: Explicit multiplicative channel count, commonly 1 or 2.
    :return: Integrated state count after applying ``spin_degeneracy``.
    :raises ValueError: If arrays, bounds, or spin degeneracy are invalid.
    """
    grid, dos = _dos_arrays(energies, density)
    lo, hi = _finite(lower, "lower"), _finite(upper, "upper")
    degeneracy = _positive(spin_degeneracy, "spin_degeneracy")
    if hi < lo:
        raise ValueError("upper must be greater than or equal to lower")
    lo, hi = max(lo, float(grid[0])), min(hi, float(grid[-1]))
    if hi <= lo:
        return 0.0
    inside = grid[(grid > lo) & (grid < hi)]
    x = np.concatenate(([lo], inside, [hi]))
    y = np.interp(x, grid, dos)
    return float(np.trapezoid(y, x) * degeneracy)


def electron_count(
    energies: Sequence[float],
    density: Sequence[float],
    chemical_potential: float,
    *,
    temperature: float,
    spin_degeneracy: float,
) -> float:
    """Integrate occupied DOS using Fermi-Dirac occupations.

    At zero temperature the occupation is one below the chemical potential,
    zero above it, and one half exactly at the chemical potential. At finite
    temperature, the piecewise-linear DOS times occupation is integrated
    with thermally resolved Gaussian panels. The DOS spin basis and multiplier must be
    chosen explicitly by the caller.

    :param energies: Strictly increasing grid in eV.
    :param density: Nonnegative DOS in states/eV on the declared spin basis.
    :param chemical_potential: Chemical potential in eV.
    :param temperature: Temperature in K, including zero.
    :param spin_degeneracy: Explicit multiplier for spin channels represented.
    :return: Integrated occupied electron count.
    :raises ValueError: If inputs are invalid or the result is non-finite.
    """
    grid, dos = _dos_arrays(energies, density)
    mu = _finite(chemical_potential, "chemical_potential")
    temp = _nonnegative(temperature, "temperature")
    degeneracy = _positive(spin_degeneracy, "spin_degeneracy")
    if temp == 0:
        if mu <= grid[0]:
            return 0.0
        return integrate_dos(grid.tolist(), dos.tolist(), float(grid[0]), mu, spin_degeneracy=degeneracy)
    # Integrate the linear DOS times Fermi occupation on short Gaussian panels.
    thermal_width = _KB_EV_K * temp
    total = _integrate_linear_dos(grid, dos, float(grid[0]), float(grid[-1]))
    # The omitted high-energy Fermi tail is bounded by total * exp(-cutoff).
    cutoff = max(40.0, math.log(total) + math.log(degeneracy) - math.log(1e-14)) if total > 0 else 40.0
    lower = min(float(grid[-1]), max(float(grid[0]), mu - cutoff * thermal_width))
    upper = max(float(grid[0]), min(float(grid[-1]), mu + cutoff * thermal_width))
    count = 0.0
    if lower > grid[0]:
        count = _integrate_linear_dos(grid, dos, float(grid[0]), lower)
    if upper <= lower:
        return count * degeneracy
    panel_count = max(1, math.ceil((upper - lower) / (2 * thermal_width)))
    boundaries = np.unique(
        np.concatenate(
            (
                grid[(grid > lower) & (grid < upper)],
                np.linspace(lower, upper, panel_count + 1),
            )
        )
    )
    for start, stop in pairwise(boundaries):
        midpoint, half_width = (start + stop) / 2, (stop - start) / 2
        x = midpoint + half_width * _GAUSS_X
        count += half_width * float(np.dot(_GAUSS_W, np.interp(x, grid, dos) * _fermi((x - mu) / thermal_width)))
    return float(count * degeneracy)


def solve_chemical_potential(
    energies: Sequence[float],
    density: Sequence[float],
    target_electrons: float,
    *,
    temperature: float,
    spin_degeneracy: float,
    tolerance: float = 1e-10,
    max_iterations: int = 200,
) -> float:
    """Find a chemical potential whose DOS integral matches a target count.

    A bracket is expanded outside the grid for finite-temperature tails. At
    zero temperature, a count on a gap plateau returns a chemical potential within that plateau. The DOS integral is never renormalized to force the requested
    count.

    :param energies: Strictly increasing DOS grid in eV.
    :param density: Nonnegative DOS in states/eV on the declared spin basis.
    :param target_electrons: Desired electron count.
    :param temperature: Temperature in K, including zero.
    :param spin_degeneracy: Explicit multiplier for spin channels represented.
    :param tolerance: Absolute electron-count convergence tolerance.
    :param max_iterations: Maximum bisection iterations.
    :return: Chemical potential in eV.
    :raises ValueError: If target is unreachable, inputs are invalid, or convergence fails.
    """
    grid, dos = _dos_arrays(energies, density)
    target = _nonnegative(target_electrons, "target_electrons")
    temp = _nonnegative(temperature, "temperature")
    degeneracy = _positive(spin_degeneracy, "spin_degeneracy")
    tol = _positive(tolerance, "tolerance")
    if not isinstance(max_iterations, int) or max_iterations < 1:
        raise ValueError("max_iterations must be a positive integer")
    total = integrate_dos(grid.tolist(), dos.tolist(), float(grid[0]), float(grid[-1]), spin_degeneracy=degeneracy)
    if target > total:
        raise ValueError("target electron count exceeds the integrated DOS")
    if target <= tol:
        return float(grid[0] - max(1.0, 40 * _KB_EV_K * temp))
    if total - target <= tol:
        return float(grid[-1] + max(1.0, 40 * _KB_EV_K * temp))
    width = max(float(grid[-1] - grid[0]), 1.0, 40 * _KB_EV_K * temp)
    lo, hi = float(grid[0] - width), float(grid[-1] + width)
    for _ in range(max_iterations):
        mid = lo + (hi - lo) / 2
        count = electron_count(grid.tolist(), dos.tolist(), mid, temperature=temp, spin_degeneracy=degeneracy)
        if abs(count - target) <= tol:
            return mid
        if count < target:
            lo = mid
        else:
            hi = mid
    if temp == 0 and hi - lo <= tolerance:
        return lo + (hi - lo) / 2
    raise ValueError("chemical-potential bisection did not converge to the requested count")


def band_edges(
    energies: Sequence[Sequence[float]],
    occupations: Sequence[Sequence[float]],
    *,
    maximum_occupation: float,
    occupation_tolerance: float = 1e-6,
    energy_reference: float,
) -> BandEdges:
    """Classify sampled band edges using explicit occupations.

    Arrays have shape (bands, k points). Energies are in eV and occupations
    use the same per-state convention as ``maximum_occupation``. Band-path
    sampling is retained as supplied and is not a Brillouin-zone integration.

    Occupations outside [0, ``maximum_occupation``] by more than ``occupation_tolerance`` are
    rejected. Methfessel-Paxton smearing (VASP ISMEAR >= 1) can yield slightly negative or
    super-maximal occupations, so callers must clean them or choose ``occupation_tolerance``
    accordingly. The analysis is per spin channel: the spin-polarised material gap is
    min(CBM over both spins) - max(VBM over both spins).

    :param energies: Finite band energies in eV.
    :param occupations: Finite occupations matching ``energies``.
    :param maximum_occupation: Explicit maximum occupancy of a state.
    :param occupation_tolerance: Absolute tolerance at empty/full endpoints.
    :param energy_reference: Explicit energy to subtract from every band energy.
    :return: Sampled valence/conduction extrema and sampled gaps.
    :raises ValueError: If the arrays or occupation conventions are invalid.
    """
    e, occ = _matrix(energies, "energies"), _matrix(occupations, "occupations")
    maximum = _positive(maximum_occupation, "maximum_occupation")
    tol = _nonnegative(occupation_tolerance, "occupation_tolerance")
    reference = _finite(energy_reference, "energy_reference")
    if e.shape != occ.shape or min(e.shape) < 1:
        raise ValueError("energies and occupations must have the same nonempty shape")
    if tol >= maximum / 2 or np.any(occ < -tol) or np.any(occ > maximum + tol):
        raise ValueError("occupations must lie within the declared occupancy range")
    partial = (occ > tol) & (occ < maximum - tol)
    occupied, empty = occ > tol, occ <= tol
    if not np.any(occupied) or not np.any(empty):
        raise ValueError("both occupied and empty states are required to define band edges")
    aligned = e - reference
    vbm_flat = int(np.argmax(np.where(occupied, aligned, -np.inf)))
    cbm_flat = int(np.argmin(np.where(empty, aligned, np.inf)))
    vband, vk = np.unravel_index(vbm_flat, e.shape)
    cband, ck = np.unravel_index(cbm_flat, e.shape)
    direct_values = [
        float(np.min(aligned[empty[:, k], k]) - np.max(aligned[occupied[:, k], k]))
        for k in range(e.shape[1])
        if np.any(empty[:, k]) and np.any(occupied[:, k])
    ]
    if not direct_values:
        raise ValueError("at least one k point must contain both occupied and empty states for a same-k direct gap")
    direct = min(direct_values)
    return BandEdges(
        bool(np.any(partial)),
        float(aligned[cband, ck] - aligned[vband, vk]),
        direct,
        float(aligned[vband, vk]),
        float(aligned[cband, ck]),
        int(vband),
        int(vk),
        int(cband),
        int(ck),
        reference,
    )


def fit_effective_mass(
    kpoints: Sequence[Sequence[float]],
    energies: Sequence[float],
    center: Sequence[float],
) -> EffectiveMassFit:
    """Fit a full local quadratic band and return its signed mass tensor.

    Cartesian k coordinates use inverse angstroms with the 2π included (physical k,
    ``|k| = 2π/λ``), energies use eV, and ``center`` uses the same Cartesian reciprocal-space
    basis. Reciprocal-lattice fractions or 1/λ coordinates give masses off by (2π)² ≈ 39.5.
    The local model
    is E=E0+g·dk+0.5*dk.T@H@dk. The conversion is m*/m_e =
    (hbar²/m_e)*H⁻¹, with hbar²/m_e in eV angstrom².

    :param kpoints: Cartesian sample vectors with shape (n, 3), in inverse angstroms including 2π.
    :param energies: Matching energies in eV.
    :param center: Expansion point in inverse angstroms, including 2π.
    :return: Fitted Hessian, signed mass tensor, axes, and residual diagnostics.
    :raises ValueError: If data are malformed, rank deficient, or the Hessian is singular.
    """
    k = _matrix(kpoints, "kpoints")
    energy = _vector(energies, "energies")
    origin = _vector(center, "center")
    if k.shape != (len(energy), 3) or origin.shape != (3,) or len(energy) < 10:
        raise ValueError("at least ten matching Cartesian k points and energies are required")
    delta = k - origin
    scale = float(np.max(np.linalg.norm(delta, axis=1)))
    if not math.isfinite(scale) or scale <= 0:
        raise ValueError("k points must span a nonzero neighborhood of center")
    q = delta / scale
    design = np.column_stack(
        (
            np.ones(len(q)),
            q,
            q[:, 0] ** 2 / 2,
            q[:, 1] ** 2 / 2,
            q[:, 2] ** 2 / 2,
            q[:, 0] * q[:, 1],
            q[:, 0] * q[:, 2],
            q[:, 1] * q[:, 2],
        )
    )
    coeff, _, rank, singular = np.linalg.lstsq(design, energy, rcond=None)
    if rank != 10 or not np.isfinite(coeff).all() or singular[-1] <= 0:
        raise ValueError(f"quadratic k-space design is rank deficient (rank {rank}/10)")
    condition = float(singular[0] / singular[-1])
    hessian = (
        np.array([[coeff[4], coeff[7], coeff[8]], [coeff[7], coeff[5], coeff[9]], [coeff[8], coeff[9], coeff[6]]])
        / scale**2
    )
    try:
        mass = _HBAR2_OVER_ME_EV_A2 * np.linalg.inv(hessian)
    except np.linalg.LinAlgError as exc:
        raise ValueError("fitted energy Hessian is singular") from exc
    if not np.isfinite(mass).all():
        raise ValueError("effective mass tensor is non-finite")
    predicted = design @ coeff
    residuals = energy - predicted
    values, axes = np.linalg.eigh(mass)
    return EffectiveMassFit(
        _tuple3(origin),
        float(coeff[0]),
        _tuple3(coeff[1:4] / scale),
        tuple(map(tuple, hessian)),
        tuple(map(tuple, mass)),
        _tuple3(values),
        tuple(map(tuple, axes)),
        int(rank),
        condition,
        float(np.sqrt(np.mean(residuals**2))),
        tuple(map(float, residuals)),
    )


def align_energies(energies: Sequence[float], reference_energy: float) -> tuple[float, ...]:
    """Subtract an explicitly supplied energy reference.

    :param energies: Finite energies in eV.
    :param reference_energy: Reference energy in eV.
    :return: Aligned energies in eV, in input order.
    :raises ValueError: If any input is non-finite.
    """
    values = _vector(energies, "energies")
    reference = _finite(reference_energy, "reference_energy")
    return tuple(float(x) for x in values - reference)


def magnetic_moments(
    moments: Sequence[Sequence[float]],
    pairs: Sequence[tuple[int, int]] = (),
    *,
    normalize_pairs: bool = False,
) -> MagneticMoments:
    """Sum vector moments and calculate requested pair spin correlations.

    Input vectors are in Bohr magnetons. Pair order and indices are preserved;
    normalization is an explicit option and zero vectors cannot be normalized.

    :param moments: Per-site vector moments in Bohr magnetons, shape (n, 3).
    :param pairs: Atom-index pairs for which to calculate dot products.
    :param normalize_pairs: Normalize the two moments before each correlation.
    :return: Total vector moment and pair correlations.
    :raises ValueError: If moments or pair indices are malformed.
    """
    vectors = _matrix(moments, "moments")
    if vectors.shape[1:] != (3,) or not len(vectors):
        raise ValueError("moments must have shape (n, 3) with n positive")
    checked: list[tuple[int, int]] = []
    correlations: list[float] = []
    for pair in pairs:
        if len(pair) != 2 or any(not isinstance(i, int) or i < 0 or i >= len(vectors) for i in pair):
            raise ValueError("each pair must contain two valid moment indices")
        i, j = pair
        left, right = vectors[i], vectors[j]
        if normalize_pairs:
            norms = np.linalg.norm((left, right), axis=1)
            if np.any(norms == 0):
                raise ValueError("zero moments cannot be normalized")
            left, right = left / norms[0], right / norms[1]
        checked.append((i, j))
        correlations.append(float(np.dot(left, right)))
    return MagneticMoments(_tuple3(vectors.sum(axis=0)), tuple(checked), tuple(correlations), normalize_pairs)


def summarize_dielectric(tensor: Sequence[Sequence[float]], *, tolerance: float = 1e-10) -> DielectricSummary:
    """Summarize a real symmetric static relative dielectric tensor.

    :param tensor: Finite 3 by 3 dimensionless relative dielectric tensor.
    :param tolerance: Absolute symmetry tolerance.
    :return: Ascending principal values, eigenvectors by column, and isotropic mean.
    :raises ValueError: If the tensor is malformed, non-real, or not symmetric.
    """
    matrix = _matrix(tensor, "tensor")
    tol = _nonnegative(tolerance, "tolerance")
    if matrix.shape != (3, 3) or not np.allclose(matrix, matrix.T, atol=tol, rtol=0):
        raise ValueError("dielectric tensor must be a real symmetric 3 by 3 matrix")
    eigenvalues, axes = np.linalg.eigh((matrix + matrix.T) / 2)
    return DielectricSummary(_tuple3(eigenvalues), tuple(map(tuple, axes)), float(np.mean(eigenvalues)))


def _dos_arrays(energies: Sequence[float], density: Sequence[float]) -> tuple[np.ndarray, np.ndarray]:
    grid, dos = _vector(energies, "energies"), _vector(density, "density")
    if len(grid) < 2 or dos.shape != grid.shape or np.any(np.diff(grid) <= 0) or np.any(dos < 0):
        raise ValueError("DOS needs matching increasing energy grid and nonnegative density")
    return grid, dos


def _integrate_linear_dos(grid: np.ndarray, dos: np.ndarray, lower: float, upper: float) -> float:
    """Integrate a piecewise-linear DOS over an already validated interval."""
    inside = grid[(grid > lower) & (grid < upper)]
    x = np.concatenate(([lower], inside, [upper]))
    return float(np.trapezoid(np.interp(x, grid, dos), x))


def _fermi(x: np.ndarray) -> np.ndarray:
    result = np.empty_like(x)
    positive = x >= 0
    decay = np.exp(-x[positive])
    result[positive] = decay / (1 + decay)
    growth = np.exp(x[~positive])
    result[~positive] = 1 / (1 + growth)
    return result


def _matrix(values: Sequence[Sequence[float]], name: str) -> np.ndarray:
    try:
        result = np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite numeric array") from exc
    if result.ndim != 2 or not np.isfinite(result).all():
        raise ValueError(f"{name} must be a finite 2D array")
    return result


def _vector(values: Sequence[float], name: str) -> np.ndarray:
    try:
        result = np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite numeric vector") from exc
    if result.ndim != 1 or not len(result) or not np.isfinite(result).all():
        raise ValueError(f"{name} must be a nonempty finite 1D array")
    return result


def _tuple3(values: np.ndarray) -> tuple[float, float, float]:
    return float(values[0]), float(values[1]), float(values[2])


def _finite(value: float, name: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be finite") from exc
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _positive(value: float, name: str) -> float:
    result = _finite(value, name)
    if result <= 0:
        raise ValueError(f"{name} must be positive")
    return result


def _nonnegative(value: float, name: str) -> float:
    result = _finite(value, name)
    if result < 0:
        raise ValueError(f"{name} must be nonnegative")
    return result
