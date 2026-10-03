"""Generic numerical analysis independent of a scientific domain."""

from .lower_hull import LowerConvexHull
from .timeseries import BlockAverage, autocorrelation, block_average

__all__ = ["BlockAverage", "LowerConvexHull", "autocorrelation", "block_average"]
