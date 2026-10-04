"""Elastic stiffness, compliance, directional response, and linear fits."""

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, cast

import numpy as np

from httk.analyse import definitions as defs
from httk.analyse._constants import GPA_PER_EV_PER_A3
from httk.analyse.definitions import BoundValue, FieldBinding, _bind_fields

__all__ = ["ElasticFit", "ElasticTensor", "fit_energy_strain", "fit_stress_strain"]

_PAIRS = ((0, 0), (1, 1), (2, 2), (1, 2), (0, 2), (0, 1))
_SHEAR = np.array((1.0, 1.0, 1.0, 2.0, 2.0, 2.0))
_KELVIN = np.sqrt(_SHEAR)
_ZERO_STRESS: tuple[float, float, float, float, float, float] = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
_ELASTIC_BINDINGS = {
    "stiffness": FieldBinding(defs.ELASTIC_TENSOR),
    "compliance": FieldBinding(defs.COMPLIANCE_TENSOR),
    "bulk_modulus_voigt": FieldBinding(defs.BULK_MODULUS_VOIGT),
    "bulk_modulus_reuss": FieldBinding(defs.BULK_MODULUS_REUSS),
    "bulk_modulus_hill": FieldBinding(defs.BULK_MODULUS_HILL),
    "shear_modulus_voigt": FieldBinding(defs.SHEAR_MODULUS_VOIGT),
    "shear_modulus_reuss": FieldBinding(defs.SHEAR_MODULUS_REUSS),
    "shear_modulus_hill": FieldBinding(defs.SHEAR_MODULUS_HILL),
    "universal_anisotropy": FieldBinding(defs.UNIVERSAL_ANISOTROPY_INDEX),
}


@dataclass(frozen=True, slots=True, init=False)
class ElasticTensor:
    """Hold a symmetric stiffness matrix in engineering-strain Voigt form.

    Entries use GPa, with components ``xx, yy, zz, yz, xz, xy``.
    Strain shear components are engineering strains ``2*epsilon_ij`` and
    stress shear components are tensor stresses ``sigma_ij``. Finite unstable
    stiffnesses are retained so callers can diagnose them.

    :param stiffness: Finite symmetric 6 by 6 stiffness matrix in GPa.
    :raises ValueError: If the matrix is not finite, real, symmetric, or 6 by 6.
    """

    stiffness: tuple[tuple[float, ...], ...]

    def __init__(self, stiffness: Sequence[Sequence[float]] | np.ndarray) -> None:
        values = _real_array(stiffness, "stiffness", (6, 6))
        if not np.allclose(values, values.T, rtol=1e-12, atol=1e-12):
            raise ValueError("stiffness must be symmetric")
        values = values / 2.0 + values.T / 2.0
        object.__setattr__(self, "stiffness", _tuple_matrix(values))

    @classmethod
    def from_full(cls, tensor: Sequence[Sequence[Sequence[Sequence[float]]]]) -> "ElasticTensor":
        """Construct from a full fourth-rank stiffness tensor.

        The tensor must satisfy both minor symmetries and major symmetry to
        relative and absolute tolerance 1e-12.

        :param tensor: Full finite stiffness tensor in GPa.
        :return: The immutable Voigt representation.
        :raises ValueError: If shape, values, or elastic symmetries are invalid.
        """
        values = _real_array(tensor, "tensor", (3, 3, 3, 3))
        if not (
            np.allclose(values, values.swapaxes(0, 1), rtol=1e-12, atol=1e-12)
            and np.allclose(values, values.swapaxes(2, 3), rtol=1e-12, atol=1e-12)
            and np.allclose(values, values.transpose(2, 3, 0, 1), rtol=1e-12, atol=1e-12)
        ):
            raise ValueError("full stiffness tensor must have both minor and major symmetries")
        matrix = np.array([[values[i, j, k, l] for k, l in _PAIRS] for i, j in _PAIRS])
        return cls(_tuple_matrix(matrix))

    def _bound_values(self, **selection: Any) -> tuple[BoundValue, ...]:
        """Bind stiffness, compliance, the six averaged moduli and the anisotropy index.

        Raises :class:`ValueError` when Reuss quantities are undefined (singular or unstable stiffness).
        """
        return _bind_fields(self, selection, _ELASTIC_BINDINGS)

    @property
    def compliance(self) -> tuple[tuple[float, ...], ...]:
        """Return the inverse stiffness in engineering Voigt form.

        :return: Compliance matrix in GPa⁻¹.
        :raises ValueError: If stiffness is singular or its inverse is non-finite.
        """
        try:
            result = np.linalg.inv(np.asarray(self.stiffness))
        except np.linalg.LinAlgError as exc:
            raise ValueError("stiffness is singular") from exc
        if not np.isfinite(result).all():
            raise ValueError("compliance is non-finite")
        return _tuple_matrix(result)

    @property
    def bulk_modulus_voigt(self) -> float:
        """Return the Voigt bulk modulus in GPa."""
        c = np.asarray(self.stiffness)
        return float((c[0, 0] + c[1, 1] + c[2, 2] + 2.0 * (c[0, 1] + c[0, 2] + c[1, 2])) / 9.0)

    @property
    def shear_modulus_voigt(self) -> float:
        """Return the Voigt shear modulus in GPa."""
        c = np.asarray(self.stiffness)
        normal = c[:3, :3]
        return float((np.trace(normal) - normal[0, 1] - normal[0, 2] - normal[1, 2] + 3.0 * np.trace(c[3:, 3:])) / 15.0)

    @property
    def bulk_modulus_reuss(self) -> float:
        """Return the Reuss bulk modulus in GPa.

        :raises ValueError: If compliance is singular or its bulk denominator is not positive.
        """
        s = np.asarray(self.compliance)
        denominator = s[0, 0] + s[1, 1] + s[2, 2] + 2.0 * (s[0, 1] + s[0, 2] + s[1, 2])
        if not math.isfinite(float(denominator)) or denominator <= 0.0:
            raise ValueError("Reuss bulk modulus is undefined for this compliance")
        return float(1.0 / denominator)

    @property
    def shear_modulus_reuss(self) -> float:
        """Return the Reuss shear modulus in GPa.

        :raises ValueError: If compliance is singular or its shear denominator is not positive.
        """
        s = np.asarray(self.compliance)
        denominator = 4.0 * (s[0, 0] + s[1, 1] + s[2, 2] - s[0, 1] - s[0, 2] - s[1, 2]) + 3.0 * (
            s[3, 3] + s[4, 4] + s[5, 5]
        )
        if not math.isfinite(float(denominator)) or denominator <= 0.0:
            raise ValueError("Reuss shear modulus is undefined for this compliance")
        return float(15.0 / denominator)

    @property
    def bulk_modulus_hill(self) -> float:
        """Return the Voigt-Reuss-Hill bulk modulus in GPa."""
        return (self.bulk_modulus_voigt + self.bulk_modulus_reuss) / 2.0

    @property
    def shear_modulus_hill(self) -> float:
        """Return the Voigt-Reuss-Hill shear modulus in GPa."""
        return (self.shear_modulus_voigt + self.shear_modulus_reuss) / 2.0

    @property
    def universal_anisotropy(self) -> float:
        """Return the universal anisotropy index ``5*Gv/Gr + Kv/Kr - 6``.

        :raises ValueError: If Reuss moduli are undefined or non-positive.
        """
        kv, kr = self.bulk_modulus_voigt, self.bulk_modulus_reuss
        gv, gr = self.shear_modulus_voigt, self.shear_modulus_reuss
        if kv <= 0.0 or gv <= 0.0:
            raise ValueError("universal anisotropy is undefined for non-positive Voigt moduli")
        return 5.0 * gv / gr + kv / kr - 6.0

    def to_full(self) -> tuple[tuple[tuple[tuple[float, ...], ...], ...], ...]:
        """Return stiffness as a full fourth-rank tensor.

        :return: Tensor ``C_ijkl`` in GPa with all minor and major symmetries.
        """
        c = np.asarray(self.stiffness)
        result = np.empty((3, 3, 3, 3), dtype=np.float64)
        for i, (a, b) in enumerate(_PAIRS):
            for j, (c0, d) in enumerate(_PAIRS):
                for row in ((a, b), (b, a)):
                    for col in ((c0, d), (d, c0)):
                        result[row[0], row[1], col[0], col[1]] = c[i, j]
        return _tuple_tensor(result)

    def compliance_full(self) -> tuple[tuple[tuple[tuple[float, ...], ...], ...], ...]:
        """Return compliance as a full fourth-rank tensor.

        :return: Tensor ``S_ijkl`` in GPa⁻¹, including engineering-shear factors.
        :raises ValueError: If stiffness is singular.
        """
        s = np.asarray(self.compliance)
        result = np.empty((3, 3, 3, 3), dtype=np.float64)
        for i, (a, b) in enumerate(_PAIRS):
            for j, (c0, d) in enumerate(_PAIRS):
                value = s[i, j] / (_SHEAR[i] * _SHEAR[j])
                for row in ((a, b), (b, a)):
                    for col in ((c0, d), (d, c0)):
                        result[row[0], row[1], col[0], col[1]] = value
        return _tuple_tensor(result)

    def rotate(self, rotation: Sequence[Sequence[float]]) -> "ElasticTensor":
        """Rotate stiffness with an active Cartesian rotation matrix.

        :param rotation: Proper orthogonal 3 by 3 Cartesian rotation.
        :return: Rotated stiffness, with ``C'ijkl = Ria Rjb Rkc Rld Cabcd``.
        :raises ValueError: If the matrix is not a proper rotation.
        """
        r = _real_array(rotation, "rotation", (3, 3))
        if not np.allclose(r @ r.T, np.eye(3), rtol=1e-10, atol=1e-10) or not np.isclose(
            np.linalg.det(r), 1.0, rtol=1e-10, atol=1e-10
        ):
            raise ValueError("rotation must be orthogonal with determinant +1")
        full = np.asarray(self.to_full())
        rotated = np.einsum("ia,jb,kc,ld,abcd->ijkl", r, r, r, r, full, optimize=True)
        return ElasticTensor.from_full(rotated)

    @property
    def stability_eigenvalues(self) -> tuple[float, ...]:
        """Return ascending Kelvin-basis eigenvalues of zero-pressure stiffness."""
        c = np.asarray(self.stiffness) * _KELVIN[:, None] * _KELVIN[None, :]
        return tuple(float(value) for value in np.linalg.eigvalsh(c))

    def is_stable(self, tolerance: float = 0.0) -> bool:
        """Test positive definiteness using the supplied absolute tolerance.

        :param tolerance: Minimum accepted Kelvin-basis eigenvalue in GPa.
        :return: Whether every stiffness eigenvalue is greater than tolerance.
        :raises ValueError: If tolerance is negative or non-finite.
        """
        tol = _nonnegative_scalar(tolerance, "tolerance")
        return min(self.stability_eigenvalues) > tol

    def pressure_stability_eigenvalues(self, pressure: float) -> tuple[float, ...]:
        """Return eigenvalues after the hydrostatic pressure correction.

        Pressure is positive in compression and measured in GPa.
        This is an incremental hydrostatic criterion, not a general
        pre-stressed finite-strain stability test.

        This tensor must be the thermodynamic stiffness: the second derivative
        of energy with respect to Lagrangian strain about the pressurised
        reference state. The correction ``C_ijkl + P*(delta_ij*delta_kl -
        delta_ik*delta_jl - delta_il*delta_jk)`` yields Wallace's stress-strain
        coefficients B for ``sigma = -P*I`` (Voigt ``C11-P, C12+P, C44-P``).
        Stress-strain coefficients, such as the output of
        ``fit_stress_strain`` or a code's elastic constants computed under
        pressure, are already B: test them with ``is_stable``, because
        applying this correction to them double-counts the pressure.

        :param pressure: Applied hydrostatic pressure in GPa.
        :return: Ascending eigenvalues of the corrected stiffness.
        """
        p = _real_scalar(pressure, "pressure")
        c = np.asarray(self.to_full())
        identity = np.eye(3)
        correction = p * (
            np.einsum("ij,kl->ijkl", identity, identity)
            - np.einsum("ik,jl->ijkl", identity, identity)
            - np.einsum("il,jk->ijkl", identity, identity)
        )
        corrected = ElasticTensor.from_full(c + correction)
        return corrected.stability_eigenvalues

    def is_stable_under_pressure(self, pressure: float, tolerance: float = 0.0) -> bool:
        """Test positive definiteness under the hydrostatic pressure correction.

        The same input contract as ``pressure_stability_eigenvalues``
        applies: the tensor must be the thermodynamic stiffness (Lagrangian
        strain energy derivative). Stress-strain coefficients are already
        corrected and must be tested with ``is_stable`` instead.

        :param pressure: Hydrostatic pressure in GPa, positive in compression.
        :param tolerance: Minimum accepted eigenvalue in GPa.
        :return: Whether all corrected stiffness eigenvalues exceed tolerance.
        :raises ValueError: If pressure or tolerance is not finite, or tolerance is negative.
        """
        tol = _nonnegative_scalar(tolerance, "tolerance")
        return min(self.pressure_stability_eigenvalues(pressure)) > tol

    def young_modulus(self, direction: Sequence[float]) -> float:
        """Return directional Young's modulus in GPa.

        :param direction: Nonzero Cartesian loading direction.
        :return: Reciprocal longitudinal compliance.
        :raises ValueError: If direction is invalid, stiffness is singular, or compliance is non-positive.
        """
        n = _unit_vector(direction, "direction")
        s = np.asarray(self.compliance_full())
        value = float(np.einsum("i,j,k,l,ijkl->", n, n, n, n, s))
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError("directional Young's modulus is undefined for this compliance")
        return 1.0 / value

    def shear_modulus(self, direction: Sequence[float], transverse: Sequence[float]) -> float:
        """Return shear modulus for an orthogonal direction pair in GPa.

        :param direction: Nonzero Cartesian shear-plane normal.
        :param transverse: Nonzero Cartesian shear direction perpendicular to ``direction``.
        :return: Directional shear modulus.
        :raises ValueError: If directions are invalid, compliance is singular, or its denominator is non-positive.
        """
        n, m = _orthogonal_directions(direction, transverse)
        s = np.asarray(self.compliance_full())
        value = float(np.einsum("i,j,k,l,ijkl->", n, m, n, m, s)) * 4.0
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError("directional shear modulus is undefined for this compliance")
        return 1.0 / value

    def poisson_ratio(self, direction: Sequence[float], transverse: Sequence[float]) -> float:
        """Return directional Poisson ratio for an orthogonal direction pair.

        :param direction: Nonzero Cartesian uniaxial loading direction.
        :param transverse: Nonzero Cartesian transverse direction perpendicular to ``direction``.
        :return: Negative transverse-to-longitudinal strain ratio.
        :raises ValueError: If directions are invalid, stiffness is singular, or longitudinal compliance is non-positive.
        """
        n, m = _orthogonal_directions(direction, transverse)
        s = np.asarray(self.compliance_full())
        longitudinal = float(np.einsum("i,j,k,l,ijkl->", n, n, n, n, s))
        transverse_strain = float(np.einsum("i,j,k,l,ijkl->", m, m, n, n, s))
        if not math.isfinite(longitudinal) or longitudinal <= 0.0 or not math.isfinite(transverse_strain):
            raise ValueError("directional Poisson ratio is undefined for this compliance")
        return -transverse_strain / longitudinal


@dataclass(frozen=True, slots=True)
class ElasticFit:
    """Retain an elastic fit, offsets, residuals, and design diagnostics.

    Residuals are observed minus fitted values. Stress residuals are in
    GPa; energy residuals are in eV. ``energy_offset`` is ``None``
    for a stress fit.

    :param tensor: Fitted stiffness tensor.
    :param stress_offset: Fitted tensile-positive stress offset in Voigt order.
    :param energy_offset: Fitted constant energy in eV, when applicable.
    :param residuals: Input-order residual tuple or tuples.
    :param rmse: Root mean square of residual components in the fit's units.
    :param condition_number: Condition number of the column-scaled design matrix.
    :param strain_range: Minimum and maximum engineering strain for each component.
    """

    tensor: ElasticTensor
    stress_offset: tuple[float, float, float, float, float, float]
    energy_offset: float | None
    residuals: tuple[float, ...] | tuple[tuple[float, ...], ...]
    rmse: float
    condition_number: float
    strain_range: tuple[tuple[float, float], ...]

    def __post_init__(self) -> None:
        """Copy nested result arrays into immutable tuples."""
        object.__setattr__(self, "stress_offset", tuple(float(x) for x in self.stress_offset))
        object.__setattr__(self, "residuals", _nested_tuple(self.residuals))
        object.__setattr__(self, "strain_range", tuple(tuple(float(x) for x in pair) for pair in self.strain_range))


def fit_stress_strain(
    strains: Sequence[Sequence[float]],
    stresses: Sequence[Sequence[float]],
    *,
    fit_offset: bool = True,
) -> ElasticFit:
    """Fit stiffness and optional stress offsets by joint least squares.

    Strains and tensile-positive stresses use ``xx, yy, zz, yz, xz, xy``
    components, with engineering shear strain ``2*epsilon_ij`` and tensor
    shear stress. The fitted tensor is the stress-strain coefficient tensor;
    at finite pressure that is Wallace's B, not the thermodynamic stiffness C,
    so test its stability with ``ElasticTensor.is_stable``, not
    ``ElasticTensor.is_stable_under_pressure``. At least enough independent
    observations for full design rank are required. The reported condition number describes only the
    column-scaled linear design, not physical parameter confidence.

    :param strains: Finite ``(samples, 6)`` engineering strain components.
    :param stresses: Matching finite tensile-positive stresses in GPa.
    :param fit_offset: Fit six constant stress offsets when true.
    :return: Immutable fitted tensor and input-order stress residuals.
    :raises ValueError: If shapes are invalid or the scaled design is rank deficient.
    """
    x = _sample_array(strains, "strains")
    y = _sample_array(stresses, "stresses")
    if x.shape != y.shape:
        raise ValueError("strains and stresses must have the same (samples, 6) shape")
    _check_fit_offset(fit_offset)
    rows: list[np.ndarray] = []
    targets: list[float] = []
    unknown_count = 21 + (6 if fit_offset else 0)
    for sample, stress in zip(x, y, strict=True):
        for component in range(6):
            row = np.zeros(unknown_count)
            for column, (a, b) in enumerate(_symmetric_pairs()):
                if component == a:
                    row[column] = sample[b]
                elif component == b:
                    row[column] = sample[a]
            if fit_offset:
                row[21 + component] = 1.0
            rows.append(row)
            targets.append(float(stress[component]))
    design = np.asarray(rows)
    solution, condition = _least_squares(design, np.asarray(targets))
    matrix = _matrix_from_symmetric(solution[:21])
    offset = _six(solution[21:27]) if fit_offset else _ZERO_STRESS
    predicted = x @ matrix.T + np.asarray(offset)
    residuals = y - predicted
    return ElasticFit(
        ElasticTensor(_tuple_matrix(matrix)),
        offset,
        None,
        tuple(tuple(float(value) for value in row) for row in residuals),
        _rmse(residuals),
        condition,
        _strain_range(x),
    )


def fit_energy_strain(
    strains: Sequence[Sequence[float]],
    energies: Sequence[float],
    volume: float,
    *,
    fit_offset: bool = True,
) -> ElasticFit:
    """Fit stiffness from a quadratic total-energy-versus-strain model.

    The model is ``E = E0 + V*(stress_offset @ strain +
    0.5*strain @ C @ strain)``. With ``fit_offset=False``, both ``E0`` and
    stress offset are fixed to zero; use that mode only when the inputs have
    this explicit zero-reference convention. At finite pressure the quadratic
    fit yields the thermodynamic stiffness C only for Lagrangian strains;
    small linear strains give neither C nor B. The design condition number
    describes the column-scaled linear solve, not physical parameter
    confidence.

    :param strains: Finite ``(samples, 6)`` engineering strain components.
    :param energies: Matching total energies in eV.
    :param volume: Positive reference volume in angstrom³.
    :param fit_offset: Fit one constant energy and six linear stress terms when true.
    :return: Fitted tensor and stress offset in GPa, with input-order energy residuals in eV.
    :raises ValueError: If inputs are invalid or the scaled design is rank deficient.
    """
    x = _sample_array(strains, "strains")
    y = _real_array(energies, "energies", (x.shape[0],))
    v = _positive_scalar(volume, "volume")
    _check_fit_offset(fit_offset)
    unknown_count = 21 + (7 if fit_offset else 0)
    design = np.empty((len(x), unknown_count), dtype=np.float64)
    for row_index, sample in enumerate(x):
        for column, (a, b) in enumerate(_symmetric_pairs()):
            design[row_index, column] = v * (0.5 * sample[a] * sample[b] if a == b else sample[a] * sample[b])
        if fit_offset:
            design[row_index, 21] = 1.0
            design[row_index, 22:28] = v * sample
    solution, condition = _least_squares(design, y, center_target=fit_offset)
    matrix = _matrix_from_symmetric(solution[:21]) * GPA_PER_EV_PER_A3
    energy_offset = float(solution[21]) if fit_offset else 0.0
    stress_offset = _six(solution[22:28] * GPA_PER_EV_PER_A3) if fit_offset else _ZERO_STRESS
    predicted = design @ solution
    residuals = y - predicted
    return ElasticFit(
        ElasticTensor(_tuple_matrix(matrix)),
        stress_offset,
        energy_offset,
        tuple(float(value) for value in residuals),
        _rmse(residuals),
        condition,
        _strain_range(x),
    )


def _real_array(values: object, name: str, shape: tuple[int, ...]) -> np.ndarray:
    """Copy a finite real array and enforce its shape."""
    try:
        raw = np.asarray(values)
        if raw.dtype.kind in "bcUSV":
            raise ValueError
        result = np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be a finite real array with shape {shape}") from exc
    if result.shape != shape or not np.isfinite(result).all():
        raise ValueError(f"{name} must be a finite real array with shape {shape}")
    return np.array(result, copy=True)


def _sample_array(values: object, name: str) -> np.ndarray:
    """Copy and validate a nonempty finite ``(samples, 6)`` array."""
    try:
        raw = np.asarray(values)
        if raw.dtype.kind in "bcUSV" or raw.ndim != 2 or raw.shape[1] != 6 or raw.shape[0] == 0:
            raise ValueError
        result = np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be a nonempty finite (samples, 6) real array") from exc
    if not np.isfinite(result).all():
        raise ValueError(f"{name} must be a nonempty finite (samples, 6) real array")
    return np.array(result, copy=True)


def _symmetric_pairs() -> tuple[tuple[int, int], ...]:
    """Return upper-triangle matrix coordinates in stable order."""
    return tuple((i, j) for i in range(6) for j in range(i, 6))


def _matrix_from_symmetric(values: np.ndarray) -> np.ndarray:
    """Expand the 21 upper-triangle stiffness values into a symmetric matrix."""
    result = np.zeros((6, 6), dtype=np.float64)
    for value, (i, j) in zip(values, _symmetric_pairs(), strict=True):
        result[i, j] = result[j, i] = value
    return result


def _six(values: np.ndarray) -> tuple[float, float, float, float, float, float]:
    """Return six scalar values as a fixed-length tuple."""
    return (float(values[0]), float(values[1]), float(values[2]), float(values[3]), float(values[4]), float(values[5]))


def _least_squares(design: np.ndarray, target: np.ndarray, *, center_target: bool = False) -> tuple[np.ndarray, float]:
    """Solve full-rank least squares after scaling columns and target."""
    shifted = target - np.mean(target) if center_target else target.copy()
    target_scale = float(np.max(np.abs(shifted)))
    if not math.isfinite(target_scale):
        raise ValueError("fit targets are not finite")
    if target_scale == 0.0:
        target_scale = 1.0
    if not np.isfinite(design).all():
        raise ValueError("fit design is not finite")
    scales = np.linalg.norm(design, axis=0)
    if not np.isfinite(scales).all() or np.any(scales == 0.0):
        raise ValueError("fit design is rank deficient")
    normalized = design / scales
    coefficients, _, rank, singular = np.linalg.lstsq(normalized, shifted / target_scale, rcond=None)
    if rank != design.shape[1] or not np.isfinite(coefficients).all() or singular[-1] <= 0.0:
        raise ValueError("scaled fit design is rank deficient")
    condition = float(singular[0] / singular[-1])
    solution = coefficients * target_scale / scales
    if center_target:
        solution[design.shape[1] - 7] += float(np.mean(target))
    if not math.isfinite(condition) or not np.isfinite(solution).all():
        raise ValueError("fit produced non-finite parameters or condition number")
    return solution, condition


def _tuple_matrix(values: np.ndarray) -> tuple[tuple[float, ...], ...]:
    """Return a nested immutable float tuple."""
    return tuple(tuple(float(value) for value in row) for row in values)


def _tuple_tensor(values: np.ndarray) -> tuple[tuple[tuple[tuple[float, ...], ...], ...], ...]:
    """Return a nested immutable rank-four float tuple."""
    return tuple(
        tuple(tuple(tuple(float(value) for value in row) for row in plane) for plane in cube) for cube in values
    )


def _nested_tuple(
    values: tuple[float, ...] | tuple[tuple[float, ...], ...],
) -> tuple[float, ...] | tuple[tuple[float, ...], ...]:
    """Copy one- or two-dimensional residuals into immutable tuples."""
    if not values:
        return ()
    if values and isinstance(values[0], (tuple, list, np.ndarray)):
        rows = cast(tuple[tuple[float, ...], ...], values)
        return tuple(tuple(float(value) for value in row) for row in rows)
    scalars = cast(tuple[float, ...], values)
    return tuple(float(value) for value in scalars)


def _unit_vector(values: Sequence[float], name: str) -> np.ndarray:
    """Normalize a finite nonzero Cartesian vector."""
    vector = _real_array(values, name, (3,))
    norm = float(np.linalg.norm(vector))
    if not math.isfinite(norm) or norm == 0.0:
        raise ValueError(f"{name} must be nonzero")
    return vector / norm


def _orthogonal_directions(first: Sequence[float], second: Sequence[float]) -> tuple[np.ndarray, np.ndarray]:
    """Normalize and verify a perpendicular direction pair."""
    a, b = _unit_vector(first, "direction"), _unit_vector(second, "transverse")
    if abs(float(a @ b)) > 1e-10:
        raise ValueError("transverse direction must be perpendicular to direction")
    return a, b


def _positive_scalar(value: float, name: str) -> float:
    """Validate a positive finite scalar."""
    result = _real_scalar(value, name)
    if result <= 0.0:
        raise ValueError(f"{name} must be positive")
    return result


def _nonnegative_scalar(value: float, name: str) -> float:
    """Validate a nonnegative finite scalar."""
    result = _real_scalar(value, name)
    if result < 0.0:
        raise ValueError(f"{name} must be nonnegative")
    return result


def _real_scalar(value: float, name: str) -> float:
    """Validate a finite real scalar."""
    if isinstance(value, (str, bytes, complex, np.complexfloating, bool)):
        raise ValueError(f"{name} must be finite and real")
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be finite and real") from exc
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite and real")
    return result


def _check_fit_offset(value: bool) -> None:
    """Require an actual bool for the offset choice."""
    if not isinstance(value, bool):
        raise ValueError("fit_offset must be bool")


def _strain_range(values: np.ndarray) -> tuple[tuple[float, float], ...]:
    """Return per-component observed engineering-strain ranges."""
    return tuple((float(np.min(values[:, i])), float(np.max(values[:, i]))) for i in range(6))


def _rmse(residuals: np.ndarray) -> float:
    """Return the root mean square residual with overflow checking."""
    value = float(np.sqrt(np.mean(np.square(residuals))))
    if not math.isfinite(value):
        raise ValueError("fit residual RMSE is non-finite")
    return value
