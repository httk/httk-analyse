"""Scalar sampling estimators for finite regularly sampled series."""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

import numpy as np

__all__ = ["BlockAverage", "autocorrelation", "block_average"]


@dataclass(frozen=True, slots=True)
class BlockAverage:
    """Immutable summary of non-overlapping block means.

    :param block_means: Means of the complete retained blocks.
    :param mean: Mean of all retained samples.
    :param standard_deviation: Sample standard deviation of the block means.
    :param standard_error: Block-mean standard deviation divided by the square root
        of the number of blocks.
    :param block_size: Number of samples in each complete block.
    :param used_samples: Number of samples included in the estimates.
    :param dropped_samples: Number of trailing samples omitted by explicit policy.
    """

    block_means: tuple[float, ...]
    mean: float
    standard_deviation: float
    standard_error: float
    block_size: int
    used_samples: int
    dropped_samples: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "block_means", tuple(float(value) for value in self.block_means))


def block_average(
    values: Sequence[float],
    *,
    block_size: int,
    remainder: Literal["raise", "drop"] = "raise",
) -> BlockAverage:
    """Summarize retained samples using non-overlapping block means.

    Blocks are independent only if the caller's sampling assumptions make them so.
    Sweep block sizes and look for a stable uncertainty estimate; two blocks alone
    do not establish reliable uncertainty. No equilibration samples are removed.

    :param values: Finite real one-dimensional samples.
    :param block_size: Positive number of samples per block.
    :param remainder: Reject a partial final block, or explicitly drop it.
    :return: Immutable block means and summary statistics.
    :raises ValueError: If samples, block size, tail policy, or finite statistics are invalid.
    """
    if isinstance(block_size, bool) or not isinstance(block_size, (int, np.integer)) or block_size <= 0:
        raise ValueError("block_size must be a positive integer")
    if remainder not in ("raise", "drop"):
        raise ValueError("remainder must be 'raise' or 'drop'")
    samples = _samples(values)
    complete_blocks, dropped = divmod(samples.size, int(block_size))
    if complete_blocks < 2:
        raise ValueError("at least two complete blocks are required")
    if dropped and remainder == "raise":
        raise ValueError("sample count is not divisible by block_size; use remainder='drop' to omit the tail")

    used = samples[: complete_blocks * int(block_size)].reshape(complete_blocks, int(block_size))
    means = tuple(_stable_mean(block) for block in used)
    retained_mean = _stable_mean(used.reshape(-1))
    block_array = np.asarray(means, dtype=np.float64)
    with np.errstate(over="ignore", invalid="ignore"):
        differences = block_array - block_array[0]
    if np.all(np.isfinite(differences)):
        scale = float(np.max(np.abs(differences)))
        normalized = differences / scale if scale else differences
    else:
        scale = float(np.max(np.abs(block_array)))
        normalized = block_array / scale if scale else block_array
    normalized_sd = float(np.std(normalized, ddof=1))
    standard_deviation = normalized_sd * scale
    standard_error = standard_deviation / np.sqrt(complete_blocks)
    if not np.isfinite(standard_deviation) or not np.isfinite(standard_error):
        raise ValueError("block statistics are not finite representable float64 values")

    return BlockAverage(
        means,
        retained_mean,
        standard_deviation,
        float(standard_error),
        int(block_size),
        int(used.size),
        int(dropped),
    )


def autocorrelation(
    values: Sequence[float],
    *,
    max_lag: int | None = None,
) -> tuple[float, ...]:
    """Compute the whole-series-centered, pair-count-normalized autocorrelation.

    This estimator assumes finite, stationary data sampled at regular intervals.
    At lag ``k`` it divides the covariance sum by ``N-k`` before normalization by
    lag zero. Long-lag values can exceed ``[-1, 1]``; this sequence is not
    guaranteed positive-semidefinite and is not a transport or spectral kernel.

    :param values: Finite, non-constant real one-dimensional samples.
    :param max_lag: Largest returned lag, or all lags through ``N - 1``.
    :return: Autocorrelation values beginning with exactly ``1.0``.
    :raises ValueError: If samples, lag, or the lag-zero normalization are invalid.
    """
    samples = _samples(values)
    if samples.size < 2:
        raise ValueError("at least two samples are required")
    if max_lag is not None and (
        isinstance(max_lag, bool)
        or not isinstance(max_lag, (int, np.integer))
        or max_lag < 0
        or max_lag >= samples.size
    ):
        raise ValueError("max_lag must be an integer in [0, N - 1]")

    # Subtract a representable origin first to preserve small fluctuations on a
    # large offset. Scale only when opposite extremes make that subtraction overflow.
    with np.errstate(over="ignore", invalid="ignore"):
        differences = samples - samples[0]
    if np.all(np.isfinite(differences)):
        difference_scale = float(np.max(np.abs(differences)))
        scaled = differences / difference_scale if difference_scale else differences
    else:
        sample_scale = float(np.max(np.abs(samples)))
        scaled_samples = samples / sample_scale if sample_scale else samples
        scaled = scaled_samples - scaled_samples[0]
    centered = scaled - np.mean(scaled)
    centered_scale = float(np.max(np.abs(centered)))
    if centered_scale == 0.0:
        raise ValueError("autocorrelation is undefined for a constant series")
    centered /= centered_scale

    size = samples.size
    fft_size = 1 << (2 * size - 1).bit_length()
    transformed = np.fft.rfft(centered, n=fft_size)
    sums = np.fft.irfft(transformed * transformed.conjugate(), n=fft_size)[:size]
    covariance = sums / np.arange(size, 0, -1, dtype=np.float64)
    if not np.isfinite(covariance[0]) or covariance[0] <= 0.0:
        raise ValueError("autocorrelation lag-zero normalization is not finite and positive")
    result = covariance / covariance[0]
    result[0] = 1.0
    stop = size if max_lag is None else int(max_lag) + 1
    return tuple(float(value) for value in result[:stop])


def _samples(values: Sequence[float]) -> np.ndarray:
    try:
        raw = np.asarray(values)
        if np.iscomplexobj(raw):
            raise ValueError("samples must be real-valued")
        samples = np.asarray(values, dtype=np.float64)
    except (TypeError, OverflowError) as exc:
        raise ValueError("samples must be finite real float64 values") from exc
    if samples.ndim != 1 or samples.size == 0:
        raise ValueError("samples must be a non-empty one-dimensional sequence")
    if not np.all(np.isfinite(samples)):
        raise ValueError("samples must contain only finite values")
    return samples


def _stable_mean(values: np.ndarray) -> float:
    with np.errstate(over="ignore", invalid="ignore"):
        differences = values - values[0]
    if np.all(np.isfinite(differences)):
        scale = float(np.max(np.abs(differences)))
        offset = float(np.mean(differences / scale) * scale) if scale else 0.0
        mean = float(values[0] + offset)
    else:
        scale = float(np.max(np.abs(values)))
        mean = float(np.mean(values / scale) * scale) if scale else 0.0
    if not np.isfinite(mean):
        raise ValueError("mean is not a finite representable float64 value")
    return mean
