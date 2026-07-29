"""Benchmark Mojo expression kernels against Polars on identical columns."""

from __future__ import annotations

import gc
import math
import os
import platform
import sys
import time

import numpy as np
import polars as upstream

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "python"))

import mojopolars as pl  # noqa: E402


def timeit(function, repeat: int = 4) -> float:
    function()
    best = math.inf
    for _ in range(repeat):
        gc.collect()
        start = time.perf_counter()
        function()
        best = min(best, time.perf_counter() - start)
    return best


def cpu_name() -> str:
    try:
        with open("/proc/cpuinfo", encoding="utf-8") as handle:
            for line in handle:
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or "unknown CPU"


def milliseconds(seconds: float) -> str:
    return f"{seconds * 1000:.2f} ms"


def relative(mojo: float, reference: float) -> str:
    ratio = reference / mojo
    return f"{ratio:.2f}x faster" if ratio >= 1 else f"{1 / ratio:.2f}x slower"


def main() -> None:
    rng = np.random.default_rng(20260729)
    n = 5_000_000
    x = rng.normal(loc=2.0, scale=1.0, size=n)
    y = rng.normal(size=n)
    groups = rng.integers(0, 100, size=n, dtype=np.int64)

    ours_x = pl.Series("x", x)
    theirs_x = upstream.Series("x", x)
    ours_frame = pl.DataFrame({"x": x, "y": y})
    theirs_frame = upstream.DataFrame({"x": x, "y": y})
    ours_grouped = pl.DataFrame({"group": groups, "x": x})
    theirs_grouped = upstream.DataFrame({"group": groups, "x": x})

    cases = [
        (
            "sum (5M)",
            ours_x.sum,
            theirs_x.sum,
        ),
        (
            "sqrt(abs(x * 1.5 + y)) (5M)",
            lambda: ours_frame.select(
                (pl.col("x") * 1.5 + pl.col("y")).abs().sqrt().alias("z")
            ),
            lambda: theirs_frame.select(
                (upstream.col("x") * 1.5 + upstream.col("y"))
                .abs()
                .sqrt()
                .alias("z")
            ),
        ),
        (
            "filter two columns (5M)",
            lambda: ours_frame.filter((pl.col("x") > 2) & (pl.col("y") < 0)),
            lambda: theirs_frame.filter(
                (upstream.col("x") > 2) & (upstream.col("y") < 0)
            ),
        ),
        (
            "cum_sum (5M)",
            ours_x.cum_sum,
            theirs_x.cum_sum,
        ),
        (
            "rolling_mean window=128 (5M)",
            lambda: ours_x.rolling_mean(128),
            lambda: theirs_x.rolling_mean(128),
        ),
        (
            "group_by sum 100 groups (5M)",
            lambda: ours_grouped.group_by("group").agg(pl.col("x").sum()),
            lambda: theirs_grouped.group_by("group").agg(upstream.col("x").sum()),
        ),
    ]

    rows = []
    for name, mojo_function, reference_function in cases:
        mojo_time = timeit(mojo_function)
        reference_time = timeit(reference_function)
        rows.append((name, mojo_time, reference_time))

    print(f"Machine: {cpu_name()} ({platform.system()} {platform.machine()})")
    print(f"Polars: {upstream.__version__}; rows: {n:,}")
    print()
    print("| kernel | mojo-polars | polars | result |")
    print("| --- | ---: | ---: | ---: |")
    for name, mojo_time, reference_time in rows:
        print(
            f"| {name} | {milliseconds(mojo_time)} | "
            f"{milliseconds(reference_time)} | {relative(mojo_time, reference_time)} |"
        )


if __name__ == "__main__":
    main()
