"""Approximate geometry and structural statistics for atomistic configurations."""

import itertools
import math
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass
from typing import Any, cast

import numpy as np

from .. import definitions as defs
from ..definitions import BoundValue, _reject_selection, _series

__all__ = [
    "RadialDistribution",
    "bond_angles",
    "coordination_numbers",
    "minimum_image",
    "radial_distribution",
    "static_structure_factor",
]

_MAX_LATTICE_CANDIDATES = 100_000


@dataclass(frozen=True, slots=True)
class RadialDistribution:
    """Hold a streamed radial distribution and its directed histogram counts.

    :param edges: Radial bin edges in angstrom.
    :param centers: Bin centers in angstrom.
    :param g: Dimensionless radial distribution values.
    :param hist_counts: Directed pair counts accumulated in each bin.
    :param frame_count: Number of configurations accumulated.
    :param mean_coordination: Mean number of selected neighbors within the binned range per selected central atom and frame.
    :param pair: Ordered ``(central, neighbor)`` labels of a partial RDF, or ``None`` for the total RDF.
    """

    edges: tuple[float, ...]
    centers: tuple[float, ...]
    g: tuple[float, ...]
    hist_counts: tuple[int, ...]
    frame_count: int
    mean_coordination: float
    pair: tuple[str, str] | None = None

    def __post_init__(self) -> None:
        """Copy sequence fields into immutable tuples."""
        object.__setattr__(self, "edges", tuple(float(value) for value in self.edges))
        object.__setattr__(self, "centers", tuple(float(value) for value in self.centers))
        object.__setattr__(self, "g", tuple(float(value) for value in self.g))
        object.__setattr__(self, "hist_counts", tuple(int(value) for value in self.hist_counts))
        if self.pair is not None:
            object.__setattr__(self, "pair", tuple(self.pair))

    def _bound_values(self, **selection: Any) -> tuple[BoundValue, ...]:
        """Bind the radial distribution function series; raise ValueError unless edges outnumber ``g`` by one."""
        _reject_selection(self, selection)
        if len(self.edges) != len(self.g) + 1:
            raise ValueError("radial distribution needs exactly one more bin edge than g values")
        pair = {} if self.pair is None else {"pair": list(self.pair)}
        return (_series(defs.RADIAL_DISTRIBUTION_FUNCTION, bin_edges=list(self.edges), g=list(self.g), **pair),)


def minimum_image(
    displacements: object,
    cell: object,
    periodicity: Sequence[bool] = (True, True, True),
) -> tuple[object, ...]:
    """Return the exact nearest periodic image(s) as nested immutable tuples.

    Cell rows are Cartesian lattice vectors in angstrom. The bounded integer
    search is exact for the supplied float64 cell and displacement values; it
    raises instead of using an incomplete neighbor stencil when the search is
    too large.

    :param displacements: Finite Cartesian vectors with final dimension three.
    :param cell: Finite nonsingular 3 by 3 row-vector cell matrix.
    :param periodicity: Boolean periodicity for the three cell directions.
    :return: A nested tuple with the same shape as ``displacements``.
    :raises ValueError: If inputs are invalid or exact enumeration exceeds its safety bound.
    """
    vectors = _real_array(displacements, "displacements", min_ndim=1)
    if vectors.shape[-1] != 3:
        raise ValueError("displacements must have final dimension 3")
    basis, inverse = _cell(cell)
    pbc = _periodicity(periodicity)
    result = _minimum_image_rows(vectors.reshape(-1, 3), basis, inverse, pbc).reshape(vectors.shape)
    return _nested_tuple(result)


def coordination_numbers(
    positions: object,
    cell: object,
    cutoff: float,
    periodicity: Sequence[bool] = (True, True, True),
) -> tuple[int, ...]:
    """Count neighbors within a cutoff for each atom in input order.

    Distances use the true nearest lattice image. A cutoff must be strictly below half
    the shortest periodic translation, which guarantees a unique image.

    :param positions: Finite Cartesian atom positions in angstrom, shape ``(N, 3)``.
    :param cell: Finite nonsingular 3 by 3 row-vector cell matrix in angstrom.
    :param cutoff: Positive neighbor distance cutoff in angstrom; the boundary is inclusive.
    :param periodicity: Boolean periodicity for the three cell directions.
    :return: Integer coordination counts in atom order, excluding each atom itself.
    :raises ValueError: If inputs are invalid or the cutoff violates the unique-image limit.
    """
    xyz = _positions(positions)
    radius = _positive_scalar(cutoff, "cutoff")
    basis, inverse = _cell(cell)
    pbc = _periodicity(periodicity)
    _check_unique_cutoff(radius, basis, inverse, pbc)
    return tuple(_neighbor_vectors(xyz, index, radius, basis, inverse, pbc).shape[0] for index in range(len(xyz)))


def bond_angles(
    positions: object,
    cell: object,
    cutoff: float,
    periodicity: Sequence[bool] = (True, True, True),
) -> tuple[tuple[float, ...], ...]:
    """Return per-center angles between unordered neighbor pairs in degrees.

    Center atoms and neighbor pairs retain input order. Periodic displacements
    use the true nearest image and the cutoff follows the unique-image rule.

    :param positions: Finite Cartesian atom positions in angstrom, shape ``(N, 3)``.
    :param cell: Finite nonsingular 3 by 3 row-vector cell matrix in angstrom.
    :param cutoff: Positive inclusive neighbor distance cutoff in angstrom.
    :param periodicity: Boolean periodicity for the three cell directions.
    :return: One tuple per center, containing angles in lexicographic neighbor-pair order.
    :raises ValueError: If inputs are invalid or the cutoff violates the unique-image limit.
    """
    xyz = _positions(positions)
    radius = _positive_scalar(cutoff, "cutoff")
    basis, inverse = _cell(cell)
    pbc = _periodicity(periodicity)
    _check_unique_cutoff(radius, basis, inverse, pbc)
    output: list[tuple[float, ...]] = []
    for center in range(len(xyz)):
        vectors = _neighbor_vectors(xyz, center, radius, basis, inverse, pbc)
        angles = []
        for first, second in itertools.combinations(vectors, 2):
            denominator = float(np.linalg.norm(first) * np.linalg.norm(second))
            if denominator == 0.0 or not math.isfinite(denominator):
                raise ValueError("bond angles are undefined for zero-length or unrepresentable neighbor vectors")
            cosine = float(np.dot(first, second) / denominator)
            angles.append(math.degrees(math.acos(min(1.0, max(-1.0, cosine)))))
        output.append(tuple(angles))
    return tuple(output)


def radial_distribution(
    frames: Iterable[object],
    cells: object,
    bins: Sequence[float],
    *,
    species: Sequence[str] | None = None,
    pair: tuple[str, str] | None = None,
    periodicity: Sequence[bool] = (True, True, True),
) -> RadialDistribution:
    """Stream a total or explicitly selected partial RDF for periodic bulk cells.

    Counts are directed and exclude self pairs. Each frame contributes its own
    ``N_A * (N_B - delta_AB) / V`` ideal-gas count to the shell-volume
    normalization. Partial RDFs require fixed per-atom labels and an explicit
    ordered ``(A, B)`` pair. The maximum edge must fit within the unique-image
    radius for every frame.

    :param frames: Single-pass iterable of finite Cartesian configurations in angstrom.
    :param cells: One fixed cell matrix or a single-pass iterable with one cell per frame.
    :param bins: Strictly increasing nonnegative bin edges in angstrom.
    :param species: Optional fixed atom labels, one per atom in every frame.
    :param pair: Explicit ordered labels ``(central, neighbor)`` for a partial RDF.
    :param periodicity: Must be fully periodic for homogeneous bulk normalization.
    :return: Immutable bin geometry, normalized values, directed counts and frame summary.
    :raises ValueError: If geometry, bins, labels or frame/cell counts are invalid.
    """
    pbc = _periodicity(periodicity)
    if pbc != (True, True, True):
        raise ValueError("radial_distribution requires full three-dimensional periodicity")
    edges = _real_array(bins, "bins", min_ndim=1)
    if edges.ndim != 1 or len(edges) < 2 or edges[0] < 0.0 or np.any(np.diff(edges) <= 0.0):
        raise ValueError("bins must be strictly increasing nonnegative edges with at least two values")
    shell_volumes = (4.0 * math.pi / 3.0) * (edges[1:] ** 3 - edges[:-1] ** 3)
    if not np.isfinite(shell_volumes).all() or np.any(shell_volumes <= 0.0):
        raise ValueError("bin shell volumes must be finite and positive")
    if (species is None) != (pair is None):
        raise ValueError("species and pair must be supplied together for a partial RDF")
    labels: tuple[str, ...] | None = None
    if species is not None:
        labels = _labels(species)
        assert pair is not None
        if (
            isinstance(pair, (str, bytes))
            or len(pair) != 2
            or any(not isinstance(label, str) or not label or label not in labels for label in pair)
        ):
            raise ValueError("pair must contain two labels present in species")
    cell_source = _cell_source(cells)
    counts = np.zeros(len(edges) - 1, dtype=np.int64)
    expected = np.zeros(len(edges) - 1, dtype=np.float64)
    selected_centers = 0
    frame_count = 0
    same_species = labels is None or (pair is not None and pair[0] == pair[1])
    for frame in frames:
        xyz = _positions(frame)
        current_cell = cell_source.fixed if cell_source.fixed is not None else _next_cell(cell_source.iterator)
        basis, inverse = _cell(current_cell)
        _check_unique_cutoff(float(edges[-1]), basis, inverse, pbc)
        volume = abs(float(np.linalg.det(basis)))
        if not math.isfinite(volume) or volume <= 0.0:
            raise ValueError("cell volume must be finite and positive")
        if labels is None:
            n_a = n_b = len(xyz)
            if n_a < 2:
                raise ValueError("total RDF requires at least two atoms per frame")
            center_indices: Sequence[int] | range = range(len(xyz))
            neighbor_indices: Sequence[int] | range = range(len(xyz))
        else:
            if len(labels) != len(xyz):
                raise ValueError("species length must match atom count in every frame")
            assert pair is not None
            center_indices = tuple(i for i, label in enumerate(labels) if label == pair[0])
            neighbor_indices = tuple(i for i, label in enumerate(labels) if label == pair[1])
            n_a, n_b = len(center_indices), len(neighbor_indices)
            if n_a == 0 or n_b == 0 or (pair[0] == pair[1] and n_a < 2):
                raise ValueError("selected species pair has zero distinct pairs in a frame")
        ideal_pairs = n_a * (n_b - int(same_species))
        if ideal_pairs <= 0:
            raise ValueError("selected species pair has zero distinct pairs in a frame")
        frame_counts = np.zeros_like(counts)
        neighbors = set(neighbor_indices)
        for center in center_indices:
            displacement = xyz - xyz[center]
            minimum = _minimum_image_rows(displacement, basis, inverse, pbc)
            distances = np.linalg.norm(minimum, axis=1)
            mask = np.fromiter((index in neighbors and index != center for index in range(len(xyz))), dtype=bool)
            selected = distances[mask]
            selected = selected[selected <= edges[-1]]
            frame_counts += np.histogram(selected, bins=edges)[0]
        counts += frame_counts
        expected += ideal_pairs / volume * shell_volumes
        selected_centers += n_a
        frame_count += 1
    if frame_count == 0:
        raise ValueError("frames must contain at least one configuration")
    if cell_source.fixed is None:
        try:
            next(cell_source.iterator)
        except StopIteration:
            pass
        else:
            raise ValueError("cells must contain exactly one matrix per frame")
    values = np.divide(counts, expected, out=np.zeros_like(expected), where=expected > 0.0)
    mean_coordination = float(np.sum(counts) / selected_centers)
    return RadialDistribution(
        tuple(float(value) for value in edges),
        tuple(float((low + high) / 2.0) for low, high in itertools.pairwise(edges)),
        tuple(float(value) for value in values),
        tuple(int(value) for value in counts),
        frame_count,
        mean_coordination,
        None if pair is None else (pair[0], pair[1]),
    )


def static_structure_factor(positions: object, wavevectors: object, weights: object | None = None) -> tuple[float, ...]:
    """Evaluate the direct static structure factor at caller-supplied wavevectors.

    The result is ``|sum_j w_j exp(i q.r_j)|² / sum_j w_j²``. Unit weights
    therefore give the requested ``/N`` normalization. The zero-wavevector
    density peak is retained; callers select species by passing only those atoms.

    :param positions: Finite Cartesian positions in angstrom, shape ``(N, 3)``.
    :param wavevectors: Finite q vectors in inverse angstrom, shape ``(M, 3)``.
    :param weights: Optional finite real atom weights, shape ``(N,)``.
    :return: Structure-factor values in wavevector order.
    :raises ValueError: If arrays have invalid shapes or weights have zero norm.
    """
    xyz = _positions(positions)
    q = _real_array(wavevectors, "wavevectors", min_ndim=2)
    if q.ndim != 2 or q.shape[1] != 3:
        raise ValueError("wavevectors must have shape (M, 3)")
    if weights is None:
        atom_weights = np.ones(len(xyz), dtype=np.float64)
    else:
        atom_weights = _real_array(weights, "weights", min_ndim=1)
        if atom_weights.ndim != 1 or atom_weights.shape != (len(xyz),):
            raise ValueError("weights must have shape (N,)")
    denominator = float(np.dot(atom_weights, atom_weights))
    if not math.isfinite(denominator) or denominator <= 0.0:
        raise ValueError("weights must have positive finite squared norm")
    phase = q @ xyz.T
    amplitudes = np.exp(1j * phase) @ atom_weights
    values = np.abs(amplitudes) ** 2 / denominator
    if not np.isfinite(values).all():
        raise ValueError("structure factor is non-finite")
    return tuple(float(value) for value in values)


def _minimum_image_rows(
    vectors: np.ndarray, cell: np.ndarray, inverse: np.ndarray, pbc: tuple[bool, ...]
) -> np.ndarray:
    gram = cell @ cell.T
    if np.array_equal(gram, np.diag(np.diag(gram))):
        shifts = np.rint(vectors @ inverse) * np.asarray(pbc)
        result = vectors - shifts @ cell
        if not np.isfinite(result).all():
            raise ValueError("minimum-image distance is not representable as float64")
        return result
    return np.asarray([_minimum_image_vector(row, cell, inverse, pbc) for row in vectors]).reshape(vectors.shape)


def _minimum_image_vector(
    vector: np.ndarray, cell: np.ndarray, inverse: np.ndarray, pbc: tuple[bool, ...]
) -> np.ndarray:
    fractional = vector @ inverse
    initial = np.array(
        [round(float(value)) if periodic else 0 for value, periodic in zip(fractional, pbc, strict=True)]
    )
    best = vector - initial @ cell
    best_norm = float(np.linalg.norm(best))
    if not math.isfinite(best_norm):
        raise ValueError("minimum-image distance is not representable as float64")
    bounds = best_norm * np.linalg.norm(inverse, axis=0)
    if not np.isfinite(bounds).all():
        raise ValueError("minimum-image lattice bounds are not representable as float64")
    ranges: list[range] = []
    for index, periodic in enumerate(pbc):
        if periodic:
            low = math.ceil(float(fractional[index] - bounds[index] - 1e-12))
            high = math.floor(float(fractional[index] + bounds[index] + 1e-12))
            ranges.append(range(low, high + 1))
        else:
            ranges.append(range(1))
    _check_search_size(ranges, "minimum-image")
    for shift in itertools.product(*ranges):
        candidate = vector - np.asarray(shift) @ cell
        norm = float(np.linalg.norm(candidate))
        if norm < best_norm:
            best, best_norm = candidate, norm
    return best


def _shortest_periodic_translation(cell: np.ndarray, inverse: np.ndarray, pbc: tuple[bool, ...]) -> float | None:
    axes = tuple(index for index, periodic in enumerate(pbc) if periodic)
    if not axes:
        return None
    best = min(float(np.linalg.norm(cell[index])) for index in axes)
    if not math.isfinite(best):
        raise ValueError("shortest periodic translation is not representable as float64")
    bounds = best * np.linalg.norm(inverse, axis=0)
    if not np.isfinite(bounds).all():
        raise ValueError("shortest-translation lattice bounds are not representable as float64")
    ranges = [
        range(math.ceil(float(-bounds[index] - 1e-12)), math.floor(float(bounds[index] + 1e-12)) + 1)
        if index in axes
        else range(1)
        for index in range(3)
    ]
    _check_search_size(ranges, "shortest-translation")
    for shift in itertools.product(*ranges):
        if not any(shift):
            continue
        vector = np.asarray(shift) @ cell
        norm = float(np.linalg.norm(vector))
        best = min(best, norm)
    return best


def _check_unique_cutoff(cutoff: float, cell: np.ndarray, inverse: np.ndarray, pbc: tuple[bool, ...]) -> None:
    shortest = _shortest_periodic_translation(cell, inverse, pbc)
    if shortest is not None and cutoff >= shortest / 2.0:
        raise ValueError("cutoff must be below half the shortest periodic translation")


def _neighbor_vectors(
    positions: np.ndarray, center: int, cutoff: float, cell: np.ndarray, inverse: np.ndarray, pbc: tuple[bool, ...]
) -> np.ndarray:
    displacement = _minimum_image_rows(positions - positions[center], cell, inverse, pbc)
    distances = np.linalg.norm(displacement, axis=1)
    return displacement[(distances <= cutoff) & (np.arange(len(positions)) != center)]


def _positions(values: object) -> np.ndarray:
    result = _real_array(values, "positions", min_ndim=2)
    if result.ndim != 2 or result.shape[1] != 3 or result.shape[0] == 0:
        raise ValueError("positions must have nonzero shape (N, 3)")
    return result


def _cell(value: object) -> tuple[np.ndarray, np.ndarray]:
    cell = _real_array(value, "cell", min_ndim=2)
    if cell.shape != (3, 3):
        raise ValueError("cell must have shape (3, 3)")
    try:
        inverse = np.linalg.inv(cell)
    except np.linalg.LinAlgError as exc:
        raise ValueError("cell must be nonsingular") from exc
    if not np.isfinite(inverse).all():
        raise ValueError("cell inverse must be finite")
    return cell, inverse


def _periodicity(values: Sequence[bool]) -> tuple[bool, bool, bool]:
    try:
        result = tuple(values)
    except TypeError as exc:
        raise ValueError("periodicity must contain three booleans") from exc
    if len(result) != 3 or any(not isinstance(value, (bool, np.bool_)) for value in result):
        raise ValueError("periodicity must contain three booleans")
    return bool(result[0]), bool(result[1]), bool(result[2])


def _labels(values: Sequence[str]) -> tuple[str, ...]:
    try:
        result = tuple(values)
    except TypeError as exc:
        raise ValueError("species must be a sequence of nonempty labels") from exc
    if any(not isinstance(label, str) or not label for label in result):
        raise ValueError("species must contain nonempty string labels")
    return result


@dataclass(slots=True)
class _CellSource:
    fixed: np.ndarray | None
    iterator: Iterator[object]


def _cell_source(cells: object) -> _CellSource:
    if isinstance(cells, np.ndarray):
        if cells.shape == (3, 3):
            return _CellSource(cells, iter(()))
        if cells.ndim == 3 and cells.shape[1:] == (3, 3):
            return _CellSource(None, iter(cells))
    if isinstance(cells, Sequence):
        try:
            array = np.asarray(cells)
        except (TypeError, ValueError):
            array = np.asarray(())
        if array.shape == (3, 3):
            return _CellSource(array, iter(()))
    try:
        return _CellSource(None, iter(cast(Iterable[object], cells)))
    except TypeError as exc:
        raise ValueError("cells must be one cell matrix or an iterable of cell matrices") from exc


def _next_cell(iterator: Iterator[object]) -> object:
    try:
        return next(iterator)
    except StopIteration as exc:
        raise ValueError("cells ended before frames") from exc


def _check_search_size(ranges: Sequence[range], name: str) -> None:
    count = math.prod(len(values) for values in ranges)
    if count > _MAX_LATTICE_CANDIDATES:
        raise ValueError(f"{name} exact lattice search needs {count} candidates; cell is too ill-conditioned")


def _real_array(values: object, name: str, *, min_ndim: int) -> np.ndarray:
    try:
        raw = np.asarray(values)
        if (
            np.iscomplexobj(raw)
            or raw.dtype.kind in "SU"
            or (
                raw.dtype.kind == "O"
                and any(isinstance(value, (str, bytes, complex, np.complexfloating)) for value in raw.flat)
            )
        ):
            raise ValueError(f"{name} must contain real numeric values")
        array = np.asarray(values, dtype=np.float64)
    except (TypeError, OverflowError) as exc:
        raise ValueError(f"{name} must contain finite real float64 values") from exc
    if array.ndim < min_ndim or not np.isfinite(array).all():
        raise ValueError(f"{name} must be a finite real array with at least {min_ndim} dimensions")
    return array.copy()


def _positive_scalar(value: float, name: str) -> float:
    if isinstance(value, (str, bytes, complex, np.complexfloating)):
        raise ValueError(f"{name} must be a positive finite scalar")
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be a positive finite scalar") from exc
    if not math.isfinite(result) or result <= 0.0:
        raise ValueError(f"{name} must be a positive finite scalar")
    return result


def _nested_tuple(values: np.ndarray) -> tuple[object, ...]:
    if values.ndim == 1:
        return tuple(float(value) for value in values)
    return tuple(_nested_tuple(row) for row in values)
