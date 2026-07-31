"""Construct a deterministic one-dimensional lower convex hull."""

from httk.analyse.generic import LowerConvexHull

hull = LowerConvexHull(
    [(0.0,), (0.5,), (1.0,)],
    [0.0, -1.0, 0.0],
)

assert hull.hull_indices == (0, 1, 2)
assert hull.value_above_hull == (0.0, 0.0, 0.0)
