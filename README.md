# mojo-polars

`mojo-polars` is a standalone Mojo implementation of the numeric columnar
expression kernels at the center of [Polars](https://pola.rs/), with a
Polars-shaped Python API.

It is not a binding to Polars and it does not load Polars at runtime. The
`polars` dependency in `pixi.toml` is used by the parity tests and benchmarks.
Applications import this project as `mojopolars`:

```python
import mojopolars as pl

frame = pl.DataFrame(
    {
        "team": ["a", "a", "b", "b"],
        "points": [10.0, 14.0, None, 22.0],
        "games": [2.0, 2.0, 1.0, 2.0],
    }
)

result = (
    frame.with_columns(
        (pl.col("points").fill_null(0) / pl.col("games")).alias("per_game")
    )
    .filter(pl.col("per_game") >= 5)
    .group_by("team", maintain_order=True)
    .agg(pl.col("per_game").mean().alias("average"))
)

print(result.to_dict(as_series=False))
# {'team': ['a', 'b'], 'average': [6.0, 11.0]}
```

## Covered subset

The API follows the corresponding Polars names and call signatures for the
covered operations.

| area | implemented |
| --- | --- |
| expression construction | `col`, `lit`, `alias`, arithmetic, comparisons, boolean `&`, `|`, `~` |
| math expressions | `abs`, `sqrt`, `exp`, `log`, `sin`, `cos`, `tan`, `floor`, `ceil`, `round`, `clip` |
| missing data | separate null validity and NaN values, `is_null`, `is_not_null`, `is_nan`, `is_finite`, `fill_null`, `fill_nan` |
| reductions | `sum`, `mean`, `min`, `max`, `product`, `count`, `len`, `n_unique`, `first`, `last`, `var`, `std`, `median`, `quantile`, `arg_min`, `arg_max`, `any`, `all` |
| ordered kernels | `cum_sum`, `cum_prod`, `cum_min`, `cum_max`, `shift`, `diff`, `pct_change` |
| windows | trailing `rolling_sum`, `rolling_mean`, `rolling_min`, `rolling_max`, `rolling_var`, `rolling_std` |
| frame operations | `select`, scalar broadcasting, `with_columns`, `filter`, `sort`, `head`, `tail`, `to_dict`, `to_dicts` |
| grouping | one or more numeric/string keys; grouped `sum`, `mean`, `min`, `max`, `count`, `len`, `first`, `last`, `var`, `std` |

The tests execute the same expressions on `mojopolars` and Polars 1.x and
compare values, null placement, NaN behavior, window results, and grouped
results. The Mojo algorithms include null-aware Welford variance, O(n)
sliding mean/variance, monotonic-deque rolling extrema, cumulative scans, and
dense integer group aggregation.

This is intentionally not the full Polars query engine. It does not implement
lazy plans or optimizer passes, joins, pivoting, file/Arrow IO, streaming,
string expression kernels, temporal/categorical/list/struct dtypes, centered
or weighted rolling windows, or Polars' selector namespace. Strings are
supported as frame values, comparison operands, and group keys, but their
compute path remains in Python. Numeric buffers use float64 storage; tracked
`Int64` and `Boolean` results round-trip naturally, but integer values outside
the exactly representable float64 range are outside this port's contract.

## Install and run

Pixi supplies the pinned Mojo nightly as well as Python, NumPy, pytest, and
upstream Polars:

```bash
pixi install
pixi run build
pixi run test
pixi run bench
```

The build task compiles the single Mojo compilation unit as a shared library:

```text
src/kernels.mojo -> dist/libmojo-polars.so
```

Set `MOJOPOLARS_LIB` to a prebuilt shared library when importing an installed
Python package without the source tree.

## Benchmarks

Measured through `pixi run bench` on an Intel Xeon E5-2697 v4 at 2.30 GHz,
Linux x86_64, with Polars 1.43.1 and 5,000,000 rows. Each result is the best of
four warmed runs. Upstream Polars uses its mature multithreaded Rust engine.
Mojo kernels use native-width SIMD with scalar tails, and large independent
elementwise work uses thresholded CPU parallelism.

| kernel | mojo-polars | polars | result |
| --- | ---: | ---: | ---: |
| sum (5M) | 3.51 ms | 4.14 ms | 1.18x faster |
| sqrt(abs(x * 1.5 + y)) (5M) | 19.72 ms | 42.47 ms | 2.15x faster |
| filter two columns (5M) | 60.84 ms | 19.28 ms | 3.16x slower |
| cum_sum (5M) | 45.95 ms | 42.34 ms | 1.09x slower |
| rolling_mean window=128 (5M) | 95.40 ms | 96.78 ms | 1.01x faster |
| group_by sum 100 groups (5M) | 63.38 ms | 17.26 ms | 3.67x slower |

SIMD reduction and a fused multiply-add/absolute/square-root path now beat
Polars for sum and the composed expression. Filtering computes selected
indexes once for all columns, and dense integer grouping keeps its native
`int64` keys, but Polars still wins those operations with compact Arrow
bitmaps and its mature grouping engine. Cumulative sum is modestly slower in
this run, while rolling mean is effectively tied.

No GPU path is included.

## How it works

Python owns every allocation. A numeric column is a contiguous float64 values
buffer plus a contiguous uint8 validity buffer, so a null is not conflated
with IEEE NaN. Strings use a NumPy object buffer and the same validity
representation.

The ctypes layer passes buffer addresses and lengths as 64-bit integers.
`src/kernels.mojo` reconstructs
`UnsafePointer[..., AnyOrigin[mut=True]]` values inside non-parametric
`@export("name")` functions with `abi("C")` before the return arrow. Results
are written into caller-owned NumPy buffers; Mojo performs no allocation and
there is no cross-language ownership to release.

Expressions form a small immutable tree. `DataFrame.select`,
`with_columns`, `filter`, and `GroupBy.agg` evaluate that tree against columns,
broadcast scalar reductions where Polars does, and dispatch the numeric work
to the shared library. Group keys are factorized in the Python layer; the
value aggregation itself runs in Mojo.

## License

MIT
