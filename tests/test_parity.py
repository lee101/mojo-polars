from __future__ import annotations

import math

import numpy as np
import polars as upstream
import pytest

import mojopolars as pl
from mojopolars import _kernels


def assert_values(got, expected):
    assert len(got) == len(expected)
    for left, right in zip(got, expected):
        if left is None or right is None:
            assert left is right
        elif isinstance(right, float) and math.isnan(right):
            assert math.isnan(left)
        elif isinstance(right, float):
            assert left == pytest.approx(right, rel=1e-11, abs=1e-12)
        else:
            assert left == right


def assert_frame(got, expected):
    assert got.columns == expected.columns
    for name in got.columns:
        assert_values(got[name].to_list(), expected[name].to_list())


def paired_frame():
    data = {
        "group": ["a", "a", "b", "b", "b"],
        "x": [1.0, 2.0, None, -4.0, 8.0],
        "y": [10.0, 20.0, 30.0, 40.0, 50.0],
        "flag": [True, False, True, True, False],
    }
    return pl.DataFrame(data), upstream.DataFrame(data)


def test_series_construction_and_nulls():
    got = pl.Series("x", [1, None, 3])
    expected = upstream.Series("x", [1, None, 3])
    assert got.name == expected.name
    assert got.null_count == expected.null_count()
    assert_values(got.to_list(), expected.to_list())
    assert got.dtype == pl.Int64


@pytest.mark.parametrize(
    ("operation", "rhs"),
    [
        ("add", 2.5),
        ("sub", -3.0),
        ("mul", 4.0),
        ("truediv", 2.0),
        ("floordiv", 3.0),
        ("mod", 3.0),
        ("pow", 2.0),
    ],
)
def test_series_arithmetic_matches_polars(operation, rhs):
    ours = pl.Series("x", [-5.0, -1.0, None, 2.0, 7.0])
    theirs = upstream.Series("x", [-5.0, -1.0, None, 2.0, 7.0])
    dunder = f"__{operation}__"
    assert_values(getattr(ours, dunder)(rhs).to_list(), getattr(theirs, dunder)(rhs).to_list())


def test_reverse_arithmetic_matches_polars():
    ours = pl.Series("x", [1.0, 2.0, None, 4.0])
    theirs = upstream.Series("x", [1.0, 2.0, None, 4.0])
    assert_values((10.0 - ours).to_list(), (10.0 - theirs).to_list())
    assert_values((10.0 / ours).to_list(), (10.0 / theirs).to_list())


def test_expression_composition_and_scalar_broadcast():
    ours, theirs = paired_frame()
    got = ours.select(
        ((pl.col("x") * 2 + pl.col("y")) / 3).alias("score"),
        pl.col("y").mean().alias("average"),
    )
    expected = theirs.select(
        ((upstream.col("x") * 2 + upstream.col("y")) / 3).alias("score"),
        upstream.col("y").mean().alias("average"),
    )
    assert_frame(got, expected)


def test_fused_expression_simd_tail_preserves_nulls():
    values = [1.0, None, -2.0, 4.0, 8.0, -3.0, 0.5]
    other = [2.0, 3.0, None, -8.0, 1.0, 6.0, -0.25]
    ours = pl.DataFrame({"x": values, "y": other})
    theirs = upstream.DataFrame({"x": values, "y": other})
    assert_frame(
        ours.select(((pl.col("x") * 1.5 + pl.col("y")).abs().sqrt()).alias("z")),
        theirs.select(
            ((upstream.col("x") * 1.5 + upstream.col("y")).abs().sqrt()).alias(
                "z"
            )
        ),
    )


def test_large_fused_paths_cross_parallel_threshold():
    size = 1_000_003
    x = np.linspace(-2.0, 3.0, size, dtype=np.float64)
    y = np.linspace(4.0, -1.0, size, dtype=np.float64)
    ours = pl.DataFrame({"x": x, "y": y})
    theirs = upstream.DataFrame({"x": x, "y": y})
    assert_frame(
        ours.select(((pl.col("x") * 0.75 + pl.col("y")).abs().sqrt()).alias("z")),
        theirs.select(
            (
                (upstream.col("x") * 0.75 + upstream.col("y"))
                .abs()
                .sqrt()
            ).alias("z")
        ),
    )
    assert_frame(
        ours.filter((pl.col("x") > -0.5) & (pl.col("y") <= 2.0)),
        theirs.filter(
            (upstream.col("x") > -0.5) & (upstream.col("y") <= 2.0)
        ),
    )


def test_parallel_scan_window_and_group_paths_keep_tail_rows():
    size = _kernels.PARALLEL_THRESHOLD + 3
    values = np.linspace(-1.0, 3.0, size, dtype=np.float64)
    groups = np.arange(size, dtype=np.int64) % 101
    ours = pl.Series("x", values)
    theirs = upstream.Series("x", values)

    np.testing.assert_allclose(
        ours.cum_sum().to_numpy(),
        theirs.cum_sum().to_numpy(),
        rtol=1e-11,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        ours.cum_sum(reverse=True).to_numpy(),
        theirs.cum_sum(reverse=True).to_numpy(),
        rtol=1e-11,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        ours.rolling_mean(17).to_numpy(),
        theirs.rolling_mean(17).to_numpy(),
        rtol=1e-11,
        atol=1e-12,
        equal_nan=True,
    )

    ours_grouped = pl.DataFrame({"group": groups, "x": values})
    theirs_grouped = upstream.DataFrame({"group": groups, "x": values})
    assert_frame(
        ours_grouped.group_by("group").agg(pl.col("x").sum()).sort("group"),
        theirs_grouped.group_by("group")
        .agg(upstream.col("x").sum())
        .sort("group"),
    )


def test_float64_numpy_input_stays_zero_copy():
    values = np.arange(17, dtype=np.float64)
    series = pl.Series("x", values)
    assert np.shares_memory(series.to_numpy(), values)


def test_raw_ffi_rejects_wrong_dtype_stride_length_and_validity():
    values = np.arange(8, dtype=np.float64)
    valid = np.ones(8, dtype=np.uint8)
    with pytest.raises(TypeError, match="float64"):
        _kernels.unary(values.astype(np.float32), valid, "abs")
    with pytest.raises(ValueError, match="C-contiguous"):
        _kernels.unary(values[::2], valid[:4], "abs")
    with pytest.raises(ValueError, match="length 8"):
        _kernels.unary(values, valid[:-1], "abs")
    invalid = valid.copy()
    invalid[-1] = 2
    with pytest.raises(ValueError, match="only 0 or 1"):
        _kernels.unary(values, invalid, "abs")


def test_int64_precision_limit_is_explicit():
    assert pl.Series([2**53]).to_list() == [2**53]
    with pytest.raises(OverflowError, match="cannot be represented exactly"):
        pl.Series([2**53 + 1])
    with pytest.raises(OverflowError, match="cannot be represented exactly"):
        pl.Series(np.asarray([-(2**53) - 1], dtype=np.int64))


def test_comparisons_boolean_logic_and_filter():
    ours, theirs = paired_frame()
    got = ours.filter((pl.col("x") > 0) & (pl.col("y") <= 50))
    expected = theirs.filter(
        (upstream.col("x") > 0) & (upstream.col("y") <= 50)
    )
    assert_frame(got, expected)
    assert_values(
        ((ours["flag"] | (ours["x"] > 0)) & ~ours["flag"]).to_list(),
        (
            (theirs["flag"] | (theirs["x"] > 0))
            & ~theirs["flag"]
        ).to_list(),
    )


def test_string_key_comparison_filter():
    ours, theirs = paired_frame()
    assert_frame(
        ours.filter(pl.col("group") == "b"),
        theirs.filter(upstream.col("group") == "b"),
    )


@pytest.mark.parametrize("method", ["abs", "sqrt", "exp", "log", "sin", "cos", "tan", "floor", "ceil"])
def test_math_expression_methods(method):
    values = [0.25, 1.0, None, 2.5] if method in ("sqrt", "log") else [-1.25, 0.0, None, 2.5]
    ours = pl.Series("x", values)
    theirs = upstream.Series("x", values)
    assert_values(
        getattr(ours, method)().to_list(),
        getattr(theirs, method)().to_list(),
    )


def test_round_clip_and_cast():
    ours = pl.Series("x", [-1.55, -0.5, None, 1.45, 9.0])
    theirs = upstream.Series("x", [-1.55, -0.5, None, 1.45, 9.0])
    assert_values(ours.round(1).to_list(), theirs.round(1).to_list())
    assert_values(ours.clip(-1, 2).to_list(), theirs.clip(-1, 2).to_list())
    assert_values(
        ours.fill_null(3).cast(pl.Int64).to_list(),
        theirs.fill_null(3).cast(upstream.Int64).to_list(),
    )


def test_null_and_nan_are_distinct():
    values = [1.0, float("nan"), None, 4.0]
    ours = pl.Series("x", values)
    theirs = upstream.Series("x", values)
    assert_values(ours.is_null().to_list(), theirs.is_null().to_list())
    assert_values(ours.is_nan().to_list(), theirs.is_nan().to_list())
    assert_values(ours.fill_null(2).to_list(), theirs.fill_null(2).to_list())
    assert_values(ours.fill_nan(7).to_list(), theirs.fill_nan(7).to_list())
    assert_values(ours.is_not_null().to_list(), theirs.is_not_null().to_list())
    assert_values(ours.is_finite().to_list(), theirs.is_finite().to_list())


@pytest.mark.parametrize(
    "method", ["sum", "mean", "min", "max", "product", "count", "var", "std", "median"]
)
def test_reductions_match_polars(method):
    ours = pl.Series("x", [1.0, 2.0, None, -4.0, 8.0])
    theirs = upstream.Series("x", [1.0, 2.0, None, -4.0, 8.0])
    assert getattr(ours, method)() == pytest.approx(getattr(theirs, method)())


def test_arg_and_unique_reductions():
    ours = pl.Series("x", [3.0, None, -1.0, 3.0, None])
    theirs = upstream.Series("x", [3.0, None, -1.0, 3.0, None])
    assert ours.arg_min() == theirs.arg_min()
    assert ours.arg_max() == theirs.arg_max()
    assert ours.n_unique() == theirs.n_unique()
    assert_frame(
        pl.DataFrame({"x": ours}).select(
            pl.col("x").first().alias("first"),
            pl.col("x").last().alias("last"),
        ),
        upstream.DataFrame({"x": theirs}).select(
            upstream.col("x").first().alias("first"),
            upstream.col("x").last().alias("last"),
        ),
    )


def test_boolean_reductions_preserve_three_valued_logic():
    for values in ([False, None], [True, None], [True, True, None]):
        ours = pl.Series("x", values)
        theirs = upstream.Series("x", values)
        assert ours.any(ignore_nulls=False) == theirs.any(ignore_nulls=False)
        assert ours.all(ignore_nulls=False) == theirs.all(ignore_nulls=False)


@pytest.mark.parametrize("method", ["cum_sum", "cum_prod", "cum_min", "cum_max"])
def test_cumulative_methods(method):
    ours = pl.Series("x", [2.0, None, -1.0, 4.0, 3.0])
    theirs = upstream.Series("x", [2.0, None, -1.0, 4.0, 3.0])
    assert_values(
        getattr(ours, method)().to_list(),
        getattr(theirs, method)().to_list(),
    )
    assert_values(
        getattr(ours, method)(reverse=True).to_list(),
        getattr(theirs, method)(reverse=True).to_list(),
    )


def test_shift_diff_and_pct_change():
    ours = pl.Series("x", [1.0, 2.0, None, 8.0, 16.0])
    theirs = upstream.Series("x", [1.0, 2.0, None, 8.0, 16.0])
    assert_values(ours.shift(2).to_list(), theirs.shift(2).to_list())
    assert_values(ours.shift(-1, fill_value=0).to_list(), theirs.shift(-1, fill_value=0).to_list())
    assert_values(ours.diff().to_list(), theirs.diff().to_list())
    assert_values(ours.pct_change().to_list(), theirs.pct_change().to_list())


@pytest.mark.parametrize(
    "method", ["rolling_sum", "rolling_mean", "rolling_min", "rolling_max", "rolling_var", "rolling_std"]
)
def test_rolling_windows_match_polars(method):
    ours = pl.Series("x", [1.0, 2.0, None, 4.0, 8.0, 16.0])
    theirs = upstream.Series("x", [1.0, 2.0, None, 4.0, 8.0, 16.0])
    ours_result = getattr(ours, method)(3, min_samples=2)
    theirs_result = getattr(theirs, method)(3, min_samples=2)
    assert_values(ours_result.to_list(), theirs_result.to_list())


def test_grouped_aggregations_match_polars():
    ours, theirs = paired_frame()
    got = ours.group_by("group", maintain_order=True).agg(
        pl.col("x").sum(),
        pl.col("y").mean().alias("y_avg"),
        pl.col("x").min().alias("smallest"),
        pl.col("x").max().alias("largest"),
        pl.col("x").count().alias("present"),
        pl.len().alias("rows"),
        pl.col("x").first().alias("first"),
        pl.col("x").last().alias("last"),
        pl.col("x").var().alias("variance"),
        pl.col("x").std().alias("spread"),
    )
    expected = theirs.group_by("group", maintain_order=True).agg(
        upstream.col("x").sum(),
        upstream.col("y").mean().alias("y_avg"),
        upstream.col("x").min().alias("smallest"),
        upstream.col("x").max().alias("largest"),
        upstream.col("x").count().alias("present"),
        upstream.len().alias("rows"),
        upstream.col("x").first().alias("first"),
        upstream.col("x").last().alias("last"),
        upstream.col("x").var().alias("variance"),
        upstream.col("x").std().alias("spread"),
    )
    assert_frame(got, expected)


def test_groupby_convenience_sum():
    ours, theirs = paired_frame()
    got = ours[["group", "x", "y"]].group_by("group", maintain_order=True).sum()
    expected = theirs[["group", "x", "y"]].group_by("group", maintain_order=True).sum()
    assert_frame(got, expected)


def test_numpy_integer_group_labels_match_polars():
    groups = np.asarray([3, 0, 3, 7, 0, 7, 7], dtype=np.int64)
    values = np.asarray([1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0])
    ours = pl.DataFrame({"group": groups, "x": values})
    theirs = upstream.DataFrame({"group": groups, "x": values})
    assert_frame(
        ours.group_by("group").agg(pl.col("x").sum()).sort("group"),
        theirs.group_by("group")
        .agg(upstream.col("x").sum())
        .sort("group"),
    )


def test_multiple_group_keys_match_polars():
    data = {
        "region": ["n", "n", "s", "s", "s"],
        "bucket": [1, 1, 1, 2, 2],
        "x": [1.0, None, 4.0, 8.0, 16.0],
    }
    ours = pl.DataFrame(data)
    theirs = upstream.DataFrame(data)
    assert_frame(
        ours.group_by("region", "bucket", maintain_order=True).agg(
            pl.col("x").sum()
        ),
        theirs.group_by("region", "bucket", maintain_order=True).agg(
            upstream.col("x").sum()
        ),
    )


def test_sort_head_tail_and_dict_rows():
    ours, theirs = paired_frame()
    got = ours.sort("y", descending=True).head(3)
    expected = theirs.sort("y", descending=True).head(3)
    assert_frame(got, expected)
    assert got.to_dicts() == expected.to_dicts()
    assert_frame(ours.tail(2), theirs.tail(2))


@pytest.mark.parametrize("descending", [False, True])
@pytest.mark.parametrize("nulls_last", [False, True])
def test_sort_null_placement_matches_polars(descending, nulls_last):
    ours = pl.Series("x", [2, None, 1, 3, None])
    theirs = upstream.Series("x", [2, None, 1, 3, None])
    assert_values(
        ours.sort(descending=descending, nulls_last=nulls_last).to_list(),
        theirs.sort(descending=descending, nulls_last=nulls_last).to_list(),
    )


def test_quantile_interpolations():
    ours = pl.Series("x", [1.0, 4.0, 9.0, None, 10.0])
    theirs = upstream.Series("x", [1.0, 4.0, 9.0, None, 10.0])
    for interpolation in ("nearest", "linear", "lower", "higher", "midpoint"):
        assert ours.quantile(0.4, interpolation) == pytest.approx(
            theirs.quantile(0.4, interpolation)
        )


def test_with_columns_preserves_unmodified_columns():
    ours, theirs = paired_frame()
    got = ours.with_columns(
        (pl.col("x").fill_null(0) * pl.col("y")).alias("weighted")
    )
    expected = theirs.with_columns(
        (upstream.col("x").fill_null(0) * upstream.col("y")).alias("weighted")
    )
    assert_frame(got, expected)
