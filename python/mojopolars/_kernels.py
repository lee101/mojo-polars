"""Numpy ownership and shape checks around the raw Mojo ABI."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

import numpy as np

from ._lib import lib

BINARY_OPS = {
    "add": 0,
    "sub": 1,
    "mul": 2,
    "truediv": 3,
    "floordiv": 4,
    "mod": 5,
    "pow": 6,
    "eq": 7,
    "ne": 8,
    "lt": 9,
    "le": 10,
    "gt": 11,
    "ge": 12,
    "and": 13,
    "or": 14,
    "min": 15,
    "max": 16,
}
UNARY_OPS = {
    "neg": 0,
    "abs": 1,
    "sqrt": 2,
    "exp": 3,
    "log": 4,
    "sin": 5,
    "cos": 6,
    "tan": 7,
    "floor": 8,
    "ceil": 9,
    "sign": 10,
    "is_nan": 11,
    "is_finite": 12,
    "fill_nan": 13,
    "clip": 14,
    "not": 15,
}
REDUCE_OPS = {
    "sum": 0,
    "mean": 1,
    "min": 2,
    "max": 3,
    "product": 4,
    "count": 5,
    "len": 6,
    "var": 7,
    "std": 8,
    "any": 9,
    "all": 10,
}
CUMULATIVE_OPS = {"cum_sum": 0, "cum_prod": 1, "cum_min": 2, "cum_max": 3}
ROLLING_OPS = {
    "rolling_sum": 0,
    "rolling_mean": 1,
    "rolling_min": 2,
    "rolling_max": 3,
    "rolling_var": 4,
    "rolling_std": 5,
}
GROUP_OPS = {
    "sum": 0,
    "mean": 1,
    "min": 2,
    "max": 3,
    "count": 4,
    "first": 5,
    "last": 6,
    "var": 7,
    "std": 8,
}
SUM_PARALLEL_THRESHOLD = 1_000_000
SUM_TASKS = 8
_SUM_POOL = ThreadPoolExecutor(max_workers=SUM_TASKS)


def addr(array: np.ndarray) -> int:
    address = int(array.ctypes.data)
    if not address:
        raise ValueError("cannot pass a null buffer address to Mojo")
    return address


def _array(
    array: np.ndarray,
    dtype: np.dtype,
    name: str,
    *,
    size: int | None = None,
) -> np.ndarray:
    if not isinstance(array, np.ndarray):
        raise TypeError(f"{name} must be a numpy.ndarray")
    if array.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")
    if array.dtype != dtype:
        raise TypeError(f"{name} must have dtype {dtype}")
    if not array.flags.c_contiguous:
        raise ValueError(f"{name} must be C-contiguous")
    if size is not None and array.size != size:
        raise ValueError(f"{name} must have length {size}")
    addr(array)
    return array


def _values(array: np.ndarray, name: str = "values", *, size=None) -> np.ndarray:
    return _array(array, np.dtype(np.float64), name, size=size)


def _valid(array: np.ndarray, name: str = "validity", *, size=None) -> np.ndarray:
    result = _array(array, np.dtype(np.uint8), name, size=size)
    if result.size and np.any(result > 1):
        raise ValueError(f"{name} must contain only 0 or 1")
    return result


def binary(
    a: np.ndarray,
    av: np.ndarray,
    b: np.ndarray | float,
    bv: np.ndarray | None,
    op: str,
    *,
    scalar_left: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    _values(a, "left values")
    _valid(av, "left validity", size=a.size)
    dst = np.empty_like(a)
    valid = np.empty(a.size, dtype=np.uint8)
    if np.isscalar(b):
        lib().mp_scalar(
            addr(a),
            addr(av),
            float(b),
            addr(dst),
            addr(valid),
            a.size,
            BINARY_OPS[op],
            int(scalar_left),
        )
    else:
        if not isinstance(b, np.ndarray) or a.size != b.size or bv is None:
            raise ValueError("expression operands must have equal length")
        _values(b, "right values", size=a.size)
        _valid(bv, "right validity", size=a.size)
        lib().mp_binary(
            addr(a),
            addr(av),
            addr(b),
            addr(bv),
            addr(dst),
            addr(valid),
            a.size,
            BINARY_OPS[op],
        )
    return dst, valid


def unary(
    a: np.ndarray,
    valid: np.ndarray,
    op: str,
    p0: float = 0.0,
    p1: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    _values(a)
    _valid(valid, size=a.size)
    dst = np.empty_like(a)
    dst_valid = np.empty(a.size, dtype=np.uint8)
    lib().mp_unary(
        addr(a),
        addr(valid),
        addr(dst),
        addr(dst_valid),
        a.size,
        UNARY_OPS[op],
        p0,
        p1,
    )
    return dst, dst_valid


def fused_math(
    a: np.ndarray,
    av: np.ndarray,
    b: np.ndarray,
    bv: np.ndarray,
    scale: float,
) -> tuple[np.ndarray, np.ndarray]:
    _values(a, "left values")
    _valid(av, "left validity", size=a.size)
    _values(b, "right values", size=a.size)
    _valid(bv, "right validity", size=a.size)
    dst = np.empty_like(a)
    valid = np.empty(a.size, dtype=np.uint8)
    lib().mp_fused_math(
        addr(a),
        addr(av),
        addr(b),
        addr(bv),
        scale,
        addr(dst),
        addr(valid),
        a.size,
    )
    return dst, valid


def fused_compare(
    a: np.ndarray,
    av: np.ndarray,
    b: np.ndarray,
    bv: np.ndarray,
    a_scalar: float,
    b_scalar: float,
    a_op: str,
    b_op: str,
    a_scalar_left: bool,
    b_scalar_left: bool,
) -> tuple[np.ndarray, np.ndarray]:
    _values(a, "left values")
    _valid(av, "left validity", size=a.size)
    _values(b, "right values", size=a.size)
    _valid(bv, "right validity", size=a.size)
    dst = np.empty_like(a)
    valid = np.empty(a.size, dtype=np.uint8)
    lib().mp_fused_compare(
        addr(a),
        addr(av),
        addr(b),
        addr(bv),
        a_scalar,
        b_scalar,
        BINARY_OPS[a_op],
        BINARY_OPS[b_op],
        int(a_scalar_left),
        int(b_scalar_left),
        addr(dst),
        addr(valid),
        a.size,
    )
    return dst, valid


def _reduce_raw(
    a: np.ndarray,
    valid: np.ndarray,
    op: str,
    ddof: int,
    all_valid: bool,
) -> float:
    _values(a)
    _valid(valid, size=a.size)
    work = np.empty(8, dtype=np.float64)
    return float(
        lib().mp_reduce(
            addr(a),
            addr(valid),
            addr(work),
            a.size,
            REDUCE_OPS[op],
            ddof,
            int(all_valid),
        )
    )


def reduce(
    a: np.ndarray, valid: np.ndarray, op: str, ddof: int = 1
) -> float:
    all_valid = bool(valid.all())
    if op != "sum" or a.size < SUM_PARALLEL_THRESHOLD:
        return _reduce_raw(a, valid, op, ddof, all_valid)
    chunk = (a.size + SUM_TASKS - 1) // SUM_TASKS
    futures = []
    for task in range(SUM_TASKS):
        start = task * chunk
        end = min(start + chunk, a.size)
        if start < end:
            futures.append(
                _SUM_POOL.submit(
                    _reduce_raw,
                    a[start:end],
                    valid[start:end],
                    op,
                    ddof,
                    all_valid,
                )
            )
    return sum(future.result() for future in futures)


def cast_int(a: np.ndarray) -> np.ndarray:
    _values(a)
    dst = np.empty(a.size, dtype=np.int64)
    lib().mp_cast_int(addr(a), addr(dst), a.size)
    return dst


def arg_reduce(a: np.ndarray, valid: np.ndarray, take_max: bool) -> int | None:
    _values(a)
    _valid(valid, size=a.size)
    index = int(lib().mp_arg_reduce(addr(a), addr(valid), a.size, int(take_max)))
    return None if index < 0 else index


def cumulative(
    a: np.ndarray, valid: np.ndarray, op: str, reverse: bool
) -> tuple[np.ndarray, np.ndarray]:
    _values(a)
    _valid(valid, size=a.size)
    dst = np.empty_like(a)
    dst_valid = np.empty(a.size, dtype=np.uint8)
    lib().mp_cumulative(
        addr(a),
        addr(valid),
        addr(dst),
        addr(dst_valid),
        a.size,
        CUMULATIVE_OPS[op],
        int(reverse),
    )
    return dst, dst_valid


def rolling(
    a: np.ndarray,
    valid: np.ndarray,
    op: str,
    window_size: int,
    min_samples: int,
    ddof: int,
) -> tuple[np.ndarray, np.ndarray]:
    _values(a)
    _valid(valid, size=a.size)
    if window_size <= 0:
        raise ValueError("window_size must be positive")
    if min_samples < 0 or min_samples > window_size:
        raise ValueError("min_samples must be between 0 and window_size")
    if ddof < 0:
        raise ValueError("ddof must be non-negative")
    dst = np.empty_like(a)
    dst_valid = np.empty(a.size, dtype=np.uint8)
    queue = np.empty(a.size, dtype=np.int64)
    lib().mp_rolling(
        addr(a),
        addr(valid),
        addr(dst),
        addr(dst_valid),
        a.size,
        addr(queue),
        window_size,
        min_samples,
        ROLLING_OPS[op],
        ddof,
    )
    return dst, dst_valid


def filter_values(
    a: np.ndarray,
    valid: np.ndarray,
    predicate: np.ndarray,
    predicate_valid: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    _values(a)
    _valid(valid, size=a.size)
    _values(predicate, "predicate", size=a.size)
    _valid(predicate_valid, "predicate validity", size=a.size)
    dst = np.empty_like(a)
    dst_valid = np.empty(a.size, dtype=np.uint8)
    size = int(
        lib().mp_filter(
            addr(a),
            addr(valid),
            addr(predicate),
            addr(predicate_valid),
            addr(dst),
            addr(dst_valid),
            a.size,
        )
    )
    return dst[:size].copy(), dst_valid[:size].copy()


def group_reduce(
    a: np.ndarray,
    valid: np.ndarray,
    labels: np.ndarray,
    ngroups: int,
    op: str,
    ddof: int,
) -> tuple[np.ndarray, np.ndarray]:
    _values(a)
    _valid(valid, size=a.size)
    _array(labels, np.dtype(np.int64), "group labels", size=a.size)
    if ngroups < 0:
        raise ValueError("ngroups must be non-negative")
    if labels.size and (labels.min() < 0 or labels.max() >= ngroups):
        raise ValueError("group labels must be in [0, ngroups)")
    if ddof < 0:
        raise ValueError("ddof must be non-negative")
    dst = np.empty(ngroups, dtype=np.float64)
    work = np.empty(ngroups, dtype=np.float64)
    dst_valid = np.empty(ngroups, dtype=np.uint8)
    counts = np.empty(ngroups, dtype=np.int64)
    lib().mp_group_reduce(
        addr(a),
        addr(valid),
        addr(labels),
        addr(dst),
        addr(work),
        addr(dst_valid),
        addr(counts),
        a.size,
        ngroups,
        GROUP_OPS[op],
        ddof,
    )
    return dst, dst_valid
