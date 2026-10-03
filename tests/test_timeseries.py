"""Checks for scalar sampling estimators."""

import math

import numpy as np
import pytest

from httk.analyse.generic.timeseries import BlockAverage, autocorrelation, block_average


def _direct_autocorrelation(values: list[float]) -> tuple[float, ...]:
    shifted = np.asarray(values, dtype=np.float64) - values[0]
    centered = shifted - np.mean(shifted)
    covariances = [
        float(np.dot(centered[: len(values) - lag], centered[lag:]) / (len(values) - lag)) for lag in range(len(values))
    ]
    return tuple(value / covariances[0] for value in covariances)


def test_block_average_known_means_and_sample_error() -> None:
    result = block_average([1, 3, 5, 7, 9, 11], block_size=2)

    assert result.block_means == (2.0, 6.0, 10.0)
    assert result.mean == 6.0
    assert result.standard_deviation == pytest.approx(4.0)
    assert result.standard_error == pytest.approx(4.0 / math.sqrt(3.0))
    assert (result.block_size, result.used_samples, result.dropped_samples) == (2, 6, 0)


def test_block_average_tail_policy_and_retained_mean() -> None:
    values = [1.0, 3.0, 9.0, 11.0, 1000.0]
    before = values.copy()
    with pytest.raises(ValueError, match="not divisible"):
        block_average(values, block_size=2)

    result = block_average(values, block_size=2, remainder="drop")
    assert result.block_means == (2.0, 10.0)
    assert result.mean == 6.0
    assert (result.used_samples, result.dropped_samples) == (4, 1)
    assert values == before


def test_block_average_owns_immutable_tuple_state() -> None:
    source = [1.0, 2.0]
    result = BlockAverage(source, 1.5, 0.7, 0.5, 1, 2, 0)
    source[0] = 99.0
    assert result.block_means == (1.0, 2.0)
    with pytest.raises((AttributeError, TypeError)):
        result.mean = 5.0  # type: ignore[misc]


@pytest.mark.parametrize(
    "values",
    [
        [1.0, 2.0, 4.0],
        [4.0, 1.0, -3.0, 2.0],
        [1.0, -1.0, 1.0, -1.0, 1.0],
        [2.0, 2.0, -1.0, 4.0, 0.0, -2.0],
        [3.0, -4.0, 8.0, 1.0, -2.0, 5.0, 0.0],
    ],
)
def test_autocorrelation_matches_direct_pair_count_oracle(values: list[float]) -> None:
    actual = autocorrelation(values)
    assert actual[0] == 1.0
    assert actual == pytest.approx(_direct_autocorrelation(values), abs=2e-14)


@pytest.mark.parametrize("size", [5, 6, 9, 10])
def test_autocorrelation_zero_padding_prevents_circular_wrap(size: int) -> None:
    values = [0.0] * size
    values[0] = 1.0
    values[-1] = -0.25
    assert autocorrelation(values) == pytest.approx(_direct_autocorrelation(values))


@pytest.mark.parametrize("size", [7, 8, 15, 16])
def test_autocorrelation_matches_direct_oracle_for_random_series(size: int) -> None:
    values = np.random.default_rng(size).normal(size=size).tolist()
    assert autocorrelation(values) == pytest.approx(_direct_autocorrelation(values), abs=2e-14)


def test_autocorrelation_is_offset_and_scale_invariant() -> None:
    values = [0.0, 2.0, -1.0, 4.0, 1.0, -3.0]
    expected = autocorrelation(values)
    assert autocorrelation([1e12 + value for value in values]) == pytest.approx(expected, abs=2e-5)
    assert autocorrelation([value * 1e-300 for value in values]) == pytest.approx(expected)
    assert autocorrelation([value * 1e300 for value in values]) == pytest.approx(expected)
    offset_values = [1e16 + 2.0 * value for value in values]
    assert autocorrelation(offset_values) == pytest.approx(_direct_autocorrelation(offset_values))


def test_autocorrelation_max_lag_includes_requested_lag() -> None:
    values = [1.0, 2.0, 0.0, -1.0, 4.0]
    assert autocorrelation(values, max_lag=0) == (1.0,)
    assert autocorrelation(values, max_lag=2) == pytest.approx(_direct_autocorrelation(values)[:3])


@pytest.mark.parametrize("values", [[], [[1.0, 2.0]], [1.0, np.nan], [1.0, np.inf]])
def test_estimators_reject_invalid_samples(values: list[float]) -> None:
    with pytest.raises(ValueError):
        block_average(values, block_size=1)
    with pytest.raises(ValueError):
        autocorrelation(values)


@pytest.mark.parametrize("values", [[1.0, 1.0], [2.0, 2.0, 2.0]])
def test_autocorrelation_rejects_constant_series(values: list[float]) -> None:
    with pytest.raises(ValueError, match="constant"):
        autocorrelation(values)


def test_estimators_reject_complex_data_explicitly() -> None:
    values = np.array([1.0 + 1.0j, 2.0 + 0.0j])
    with pytest.raises(ValueError, match="real-valued"):
        block_average(values, block_size=1)
    with pytest.raises(ValueError, match="real-valued"):
        autocorrelation(values)


@pytest.mark.parametrize("block_size", [0, -1, 1.5, True])
def test_block_average_requires_positive_integer_size(block_size: float | bool) -> None:
    with pytest.raises(ValueError, match="positive integer"):
        block_average([1.0, 2.0], block_size=block_size)  # type: ignore[arg-type]


def test_block_average_requires_two_complete_blocks() -> None:
    with pytest.raises(ValueError, match="two complete blocks"):
        block_average([1.0, 2.0], block_size=2)


@pytest.mark.parametrize("remainder", ["ignore", None])
def test_block_average_rejects_unknown_remainder_policy(remainder: str | None) -> None:
    with pytest.raises(ValueError, match="remainder"):
        block_average([1.0, 2.0], block_size=1, remainder=remainder)  # type: ignore[arg-type]


@pytest.mark.parametrize("max_lag", [-1, 3, 1.5, True])
def test_autocorrelation_rejects_invalid_lags(max_lag: float | bool) -> None:
    with pytest.raises(ValueError, match="max_lag"):
        autocorrelation([1.0, 2.0, 0.0], max_lag=max_lag)  # type: ignore[arg-type]


def test_statistics_reject_unrepresentable_block_standard_deviation() -> None:
    with pytest.raises(ValueError, match="not finite representable"):
        block_average([1.5e308, 1.5e308, -1.5e308, -1.5e308], block_size=2)


def test_block_statistics_handle_large_and_small_representable_amplitudes() -> None:
    assert block_average([1e308, 1e308, 1e308, 1e308], block_size=2).mean == 1e308
    tiny = block_average([1e-300, 3e-300, 5e-300, 7e-300], block_size=2)
    assert tiny.mean / 1e-300 == pytest.approx(4.0)
    assert tiny.standard_deviation / 1e-300 == pytest.approx(math.sqrt(8.0))


def test_block_standard_deviation_preserves_variation_on_large_offset() -> None:
    result = block_average([1e16, 1e16 + 2.0, 1e16 + 8.0, 1e16 + 10.0], block_size=1)
    assert result.standard_deviation == pytest.approx(math.sqrt(68.0 / 3.0))
