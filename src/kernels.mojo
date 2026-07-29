"""Numeric columnar expression kernels exposed through a compact C ABI."""

from std.algorithm import sync_parallelize
from std.ffi import external_call
from std.math import ceil, cos, exp, floor, isnan, pow, sin, sqrt, tan
from std.sys.info import simd_width_of as simdwidthof

comptime FPtr = UnsafePointer[Float64, AnyOrigin[mut=True]]
comptime IPtr = UnsafePointer[Int64, AnyOrigin[mut=True]]
comptime BPtr = UnsafePointer[UInt8, AnyOrigin[mut=True]]
comptime PARALLEL_THRESHOLD = 1_000_000
comptime SUM_PARALLEL_THRESHOLD = 10_000_000
comptime PARALLEL_TASKS = 8


def fp(addr: Int) -> FPtr:
    return FPtr(unsafe_from_address=addr)


def ip(addr: Int) -> IPtr:
    return IPtr(unsafe_from_address=addr)


def bp(addr: Int) -> BPtr:
    return BPtr(unsafe_from_address=addr)


def nan_value() -> Float64:
    var zero = 0.0
    return zero / zero


def precise_log(value: Float64) -> Float64:
    return external_call["log", Float64](value)


def binary_value(a: Float64, b: Float64, op: Int) -> Float64:
    if op == 0:
        return a + b
    if op == 1:
        return a - b
    if op == 2:
        return a * b
    if op == 3:
        return a / b
    if op == 4:
        return floor(a / b)
    if op == 5:
        return a - floor(a / b) * b
    if op == 6:
        return pow(a, b)
    if op == 7:
        return 1.0 if a == b else 0.0
    if op == 8:
        return 1.0 if a != b else 0.0
    if op == 9:
        return 1.0 if a < b else 0.0
    if op == 10:
        return 1.0 if a <= b else 0.0
    if op == 11:
        return 1.0 if a > b else 0.0
    if op == 12:
        return 1.0 if a >= b else 0.0
    if op == 13:
        return 1.0 if a != 0.0 and b != 0.0 else 0.0
    if op == 14:
        return 1.0 if a != 0.0 or b != 0.0 else 0.0
    if op == 15:
        return min(a, b)
    return max(a, b)


def binary_fast_range[op: Int](
    a: FPtr,
    av: BPtr,
    b: FPtr,
    bv: BPtr,
    dst: FPtr,
    dv: BPtr,
    start: Int,
    end: Int,
):
    comptime W = simdwidthof[DType.float64]()
    var i = start
    while i + W <= end:
        var va = a.load[width=W](i)
        var vb = b.load[width=W](i)
        var result: SIMD[DType.float64, W]
        comptime if op == 0:
            result = va + vb
        elif op == 2:
            result = va * vb
        elif op == 9:
            result = va.lt(vb).select(
                SIMD[DType.float64, W](1.0), SIMD[DType.float64, W](0.0)
            )
        elif op == 10:
            result = va.le(vb).select(
                SIMD[DType.float64, W](1.0), SIMD[DType.float64, W](0.0)
            )
        elif op == 11:
            result = va.gt(vb).select(
                SIMD[DType.float64, W](1.0), SIMD[DType.float64, W](0.0)
            )
        elif op == 12:
            result = va.ge(vb).select(
                SIMD[DType.float64, W](1.0), SIMD[DType.float64, W](0.0)
            )
        else:
            var a_true = av.load[width=W](i).ne(0) & va.ne(0.0)
            var b_true = bv.load[width=W](i).ne(0) & vb.ne(0.0)
            result = (a_true & b_true).select(
                SIMD[DType.float64, W](1.0), SIMD[DType.float64, W](0.0)
            )
        dst.store(i, result)
        comptime if op == 13:
            var a_valid = av.load[width=W](i).ne(0)
            var b_valid = bv.load[width=W](i).ne(0)
            var valid = (a_valid & b_valid) | (a_valid & va.eq(0.0)) | (
                b_valid & vb.eq(0.0)
            )
            dv.store(
                i,
                valid.select(
                    SIMD[DType.uint8, W](1), SIMD[DType.uint8, W](0)
                ),
            )
        else:
            dv.store(i, av.load[width=W](i) & bv.load[width=W](i))
        i += W
    while i < end:
        var valid = av[i] != 0 and bv[i] != 0
        comptime if op == 13:
            valid = valid or (av[i] != 0 and a[i] == 0.0) or (
                bv[i] != 0 and b[i] == 0.0
            )
        dv[i] = UInt8(1) if valid else UInt8(0)
        if valid:
            comptime if op == 0:
                dst[i] = a[i] + b[i]
            elif op == 2:
                dst[i] = a[i] * b[i]
            elif op == 9:
                dst[i] = 1.0 if a[i] < b[i] else 0.0
            elif op == 10:
                dst[i] = 1.0 if a[i] <= b[i] else 0.0
            elif op == 11:
                dst[i] = 1.0 if a[i] > b[i] else 0.0
            elif op == 12:
                dst[i] = 1.0 if a[i] >= b[i] else 0.0
            else:
                dst[i] = (
                    1.0
                    if av[i] != 0 and a[i] != 0.0
                    and bv[i] != 0 and b[i] != 0.0
                    else 0.0
                )
        i += 1


@no_inline
def binary_fast_dispatch(
    op: Int,
    a: FPtr,
    av: BPtr,
    b: FPtr,
    bv: BPtr,
    dst: FPtr,
    dv: BPtr,
    start: Int,
    end: Int,
):
    if op == 0:
        binary_fast_range[0](a, av, b, bv, dst, dv, start, end)
    elif op == 2:
        binary_fast_range[2](a, av, b, bv, dst, dv, start, end)
    elif op == 9:
        binary_fast_range[9](a, av, b, bv, dst, dv, start, end)
    elif op == 10:
        binary_fast_range[10](a, av, b, bv, dst, dv, start, end)
    elif op == 11:
        binary_fast_range[11](a, av, b, bv, dst, dv, start, end)
    elif op == 12:
        binary_fast_range[12](a, av, b, bv, dst, dv, start, end)
    else:
        binary_fast_range[13](a, av, b, bv, dst, dv, start, end)


@no_inline
def binary_fast(
    op: Int,
    a_addr: Int,
    av_addr: Int,
    b_addr: Int,
    bv_addr: Int,
    dst_addr: Int,
    dv_addr: Int,
    n: Int,
):
    if n < PARALLEL_THRESHOLD:
        binary_fast_dispatch(
            op,
            fp(a_addr),
            bp(av_addr),
            fp(b_addr),
            bp(bv_addr),
            fp(dst_addr),
            bp(dv_addr),
            0,
            n,
        )
        return
    var chunk = (n + PARALLEL_TASKS - 1) // PARALLEL_TASKS

    @__copy_capture(
        op, a_addr, av_addr, b_addr, bv_addr, dst_addr, dv_addr, n, chunk
    )
    @parameter
    def work(task: Int):
        var start = task * chunk
        var end = min(start + chunk, n)
        if start < end:
            binary_fast_dispatch(
                op,
                fp(a_addr),
                bp(av_addr),
                fp(b_addr),
                bp(bv_addr),
                fp(dst_addr),
                bp(dv_addr),
                start,
                end,
            )

    sync_parallelize[work](PARALLEL_TASKS)


def scalar_fast_range[op: Int](
    a: FPtr,
    av: BPtr,
    scalar: Float64,
    dst: FPtr,
    dv: BPtr,
    start: Int,
    end: Int,
    scalar_left: Bool,
):
    comptime W = simdwidthof[DType.float64]()
    var i = start
    while i + W <= end:
        var va = a.load[width=W](i)
        var vb = SIMD[DType.float64, W](scalar)
        var left = vb if scalar_left else va
        var right = va if scalar_left else vb
        var result: SIMD[DType.float64, W]
        comptime if op == 0:
            result = left + right
        elif op == 2:
            result = left * right
        elif op == 9:
            result = left.lt(right).select(
                SIMD[DType.float64, W](1.0), SIMD[DType.float64, W](0.0)
            )
        elif op == 10:
            result = left.le(right).select(
                SIMD[DType.float64, W](1.0), SIMD[DType.float64, W](0.0)
            )
        elif op == 11:
            result = left.gt(right).select(
                SIMD[DType.float64, W](1.0), SIMD[DType.float64, W](0.0)
            )
        else:
            result = left.ge(right).select(
                SIMD[DType.float64, W](1.0), SIMD[DType.float64, W](0.0)
            )
        dst.store(i, result)
        dv.store(i, av.load[width=W](i))
        i += W
    while i < end:
        dv[i] = av[i]
        if av[i] != 0:
            var left = scalar if scalar_left else a[i]
            var right = a[i] if scalar_left else scalar
            comptime if op == 0:
                dst[i] = left + right
            elif op == 2:
                dst[i] = left * right
            elif op == 9:
                dst[i] = 1.0 if left < right else 0.0
            elif op == 10:
                dst[i] = 1.0 if left <= right else 0.0
            elif op == 11:
                dst[i] = 1.0 if left > right else 0.0
            else:
                dst[i] = 1.0 if left >= right else 0.0
        i += 1


@no_inline
def scalar_fast_dispatch(
    op: Int,
    a: FPtr,
    av: BPtr,
    scalar: Float64,
    dst: FPtr,
    dv: BPtr,
    start: Int,
    end: Int,
    scalar_left: Bool,
):
    if op == 0:
        scalar_fast_range[0](
            a, av, scalar, dst, dv, start, end, scalar_left
        )
    elif op == 2:
        scalar_fast_range[2](
            a, av, scalar, dst, dv, start, end, scalar_left
        )
    elif op == 9:
        scalar_fast_range[9](
            a, av, scalar, dst, dv, start, end, scalar_left
        )
    elif op == 10:
        scalar_fast_range[10](
            a, av, scalar, dst, dv, start, end, scalar_left
        )
    elif op == 11:
        scalar_fast_range[11](
            a, av, scalar, dst, dv, start, end, scalar_left
        )
    else:
        scalar_fast_range[12](
            a, av, scalar, dst, dv, start, end, scalar_left
        )


@no_inline
def scalar_fast(
    op: Int,
    a_addr: Int,
    av_addr: Int,
    scalar: Float64,
    dst_addr: Int,
    dv_addr: Int,
    n: Int,
    scalar_left: Bool,
):
    if n < PARALLEL_THRESHOLD:
        scalar_fast_dispatch(
            op,
            fp(a_addr),
            bp(av_addr),
            scalar,
            fp(dst_addr),
            bp(dv_addr),
            0,
            n,
            scalar_left,
        )
        return
    var chunk = (n + PARALLEL_TASKS - 1) // PARALLEL_TASKS

    @__copy_capture(
        op,
        a_addr,
        av_addr,
        scalar,
        dst_addr,
        dv_addr,
        n,
        chunk,
        scalar_left,
    )
    @parameter
    def work(task: Int):
        var start = task * chunk
        var end = min(start + chunk, n)
        if start < end:
            scalar_fast_dispatch(
                op,
                fp(a_addr),
                bp(av_addr),
                scalar,
                fp(dst_addr),
                bp(dv_addr),
                start,
                end,
                scalar_left,
            )

    sync_parallelize[work](PARALLEL_TASKS)


def unary_fast_range[op: Int](
    a: FPtr, av: BPtr, dst: FPtr, dv: BPtr, start: Int, end: Int
):
    comptime W = simdwidthof[DType.float64]()
    var i = start
    while i + W <= end:
        var value = a.load[width=W](i)
        comptime if op == 1:
            dst.store(i, abs(value))
        else:
            dst.store(i, sqrt(value))
        dv.store(i, av.load[width=W](i))
        i += W
    while i < end:
        dv[i] = av[i]
        if av[i] != 0:
            comptime if op == 1:
                dst[i] = abs(a[i])
            else:
                dst[i] = sqrt(a[i])
        i += 1


@no_inline
def unary_fast_dispatch(
    op: Int,
    a: FPtr,
    av: BPtr,
    dst: FPtr,
    dv: BPtr,
    start: Int,
    end: Int,
):
    if op == 1:
        unary_fast_range[1](a, av, dst, dv, start, end)
    else:
        unary_fast_range[2](a, av, dst, dv, start, end)


@no_inline
def unary_fast(
    op: Int,
    a_addr: Int, av_addr: Int, dst_addr: Int, dv_addr: Int, n: Int
):
    if n < PARALLEL_THRESHOLD:
        unary_fast_dispatch(
            op, fp(a_addr), bp(av_addr), fp(dst_addr), bp(dv_addr), 0, n
        )
        return
    var chunk = (n + PARALLEL_TASKS - 1) // PARALLEL_TASKS

    @__copy_capture(op, a_addr, av_addr, dst_addr, dv_addr, n, chunk)
    @parameter
    def work(task: Int):
        var start = task * chunk
        var end = min(start + chunk, n)
        if start < end:
            unary_fast_dispatch(
                op,
                fp(a_addr),
                bp(av_addr),
                fp(dst_addr),
                bp(dv_addr),
                start,
                end,
            )

    sync_parallelize[work](PARALLEL_TASKS)


def fused_math_range(
    a: FPtr,
    av: BPtr,
    b: FPtr,
    bv: BPtr,
    scale: Float64,
    dst: FPtr,
    dv: BPtr,
    start: Int,
    end: Int,
):
    comptime W = simdwidthof[DType.float64]()
    var i = start
    var factor = SIMD[DType.float64, W](scale)
    while i + W <= end:
        var result = sqrt(abs(a.load[width=W](i) * factor + b.load[width=W](i)))
        dst.store(i, result)
        dv.store(i, av.load[width=W](i) & bv.load[width=W](i))
        i += W
    while i < end:
        var valid = av[i] != 0 and bv[i] != 0
        dv[i] = UInt8(1) if valid else UInt8(0)
        if valid:
            dst[i] = sqrt(abs(a[i] * scale + b[i]))
        i += 1


@no_inline
def fused_math(
    a_addr: Int,
    av_addr: Int,
    b_addr: Int,
    bv_addr: Int,
    scale: Float64,
    dst_addr: Int,
    dv_addr: Int,
    n: Int,
):
    if n < PARALLEL_THRESHOLD:
        fused_math_range(
            fp(a_addr),
            bp(av_addr),
            fp(b_addr),
            bp(bv_addr),
            scale,
            fp(dst_addr),
            bp(dv_addr),
            0,
            n,
        )
        return
    var chunk = (n + PARALLEL_TASKS - 1) // PARALLEL_TASKS

    @__copy_capture(
        a_addr,
        av_addr,
        b_addr,
        bv_addr,
        scale,
        dst_addr,
        dv_addr,
        n,
        chunk,
    )
    @parameter
    def work(task: Int):
        var start = task * chunk
        var end = min(start + chunk, n)
        if start < end:
            fused_math_range(
                fp(a_addr),
                bp(av_addr),
                fp(b_addr),
                bp(bv_addr),
                scale,
                fp(dst_addr),
                bp(dv_addr),
                start,
                end,
            )

    sync_parallelize[work](PARALLEL_TASKS)


def compare_vector[W: Int](
    value: SIMD[DType.float64, W],
    scalar: SIMD[DType.float64, W],
    op: Int,
    scalar_left: Bool,
) -> SIMD[DType.bool, W]:
    var left = scalar if scalar_left else value
    var right = value if scalar_left else scalar
    if op == 7:
        return left.eq(right)
    if op == 8:
        return left.ne(right)
    if op == 9:
        return left.lt(right)
    if op == 10:
        return left.le(right)
    if op == 11:
        return left.gt(right)
    return left.ge(right)


def compare_scalar(
    value: Float64, scalar: Float64, op: Int, scalar_left: Bool
) -> Bool:
    var left = scalar if scalar_left else value
    var right = value if scalar_left else scalar
    if op == 7:
        return left == right
    if op == 8:
        return left != right
    if op == 9:
        return left < right
    if op == 10:
        return left <= right
    if op == 11:
        return left > right
    return left >= right


def fused_compare_range(
    a: FPtr,
    av: BPtr,
    b: FPtr,
    bv: BPtr,
    a_scalar: Float64,
    b_scalar: Float64,
    a_op: Int,
    b_op: Int,
    a_scalar_left: Bool,
    b_scalar_left: Bool,
    dst: FPtr,
    dv: BPtr,
    start: Int,
    end: Int,
):
    comptime W = simdwidthof[DType.float64]()
    var i = start
    var zero = SIMD[DType.float64, W](0.0)
    var one = SIMD[DType.float64, W](1.0)
    var asimd = SIMD[DType.float64, W](a_scalar)
    var bsimd = SIMD[DType.float64, W](b_scalar)
    while i + W <= end:
        var predicate = compare_vector(
            a.load[width=W](i), asimd, a_op, a_scalar_left
        ) & compare_vector(b.load[width=W](i), bsimd, b_op, b_scalar_left)
        dst.store(i, predicate.select(one, zero))
        dv.store(i, av.load[width=W](i) & bv.load[width=W](i))
        i += W
    while i < end:
        var valid = av[i] != 0 and bv[i] != 0
        dv[i] = UInt8(1) if valid else UInt8(0)
        if valid:
            dst[i] = (
                1.0
                if compare_scalar(a[i], a_scalar, a_op, a_scalar_left)
                and compare_scalar(b[i], b_scalar, b_op, b_scalar_left)
                else 0.0
            )
        i += 1


@no_inline
def fused_compare(
    a_addr: Int,
    av_addr: Int,
    b_addr: Int,
    bv_addr: Int,
    a_scalar: Float64,
    b_scalar: Float64,
    a_op: Int,
    b_op: Int,
    a_scalar_left: Bool,
    b_scalar_left: Bool,
    dst_addr: Int,
    dv_addr: Int,
    n: Int,
):
    if n < PARALLEL_THRESHOLD:
        fused_compare_range(
            fp(a_addr),
            bp(av_addr),
            fp(b_addr),
            bp(bv_addr),
            a_scalar,
            b_scalar,
            a_op,
            b_op,
            a_scalar_left,
            b_scalar_left,
            fp(dst_addr),
            bp(dv_addr),
            0,
            n,
        )
        return
    var chunk = (n + PARALLEL_TASKS - 1) // PARALLEL_TASKS

    @__copy_capture(
        a_addr,
        av_addr,
        b_addr,
        bv_addr,
        a_scalar,
        b_scalar,
        a_op,
        b_op,
        a_scalar_left,
        b_scalar_left,
        dst_addr,
        dv_addr,
        n,
        chunk,
    )
    @parameter
    def work(task: Int):
        var start = task * chunk
        var end = min(start + chunk, n)
        if start < end:
            fused_compare_range(
                fp(a_addr),
                bp(av_addr),
                fp(b_addr),
                bp(bv_addr),
                a_scalar,
                b_scalar,
                a_op,
                b_op,
                a_scalar_left,
                b_scalar_left,
                fp(dst_addr),
                bp(dv_addr),
                start,
                end,
            )

    sync_parallelize[work](PARALLEL_TASKS)


def sum_range(a: FPtr, av: BPtr, start: Int, end: Int) -> Float64:
    comptime W = simdwidthof[DType.float64]()
    var totals = SIMD[DType.float64, W](0.0)
    var i = start
    while i + W <= end:
        var values = a.load[width=W](i)
        var present = av.load[width=W](i).ne(0)
        totals += present.select(values, SIMD[DType.float64, W](0.0))
        i += W
    var total = totals.reduce_add()
    while i < end:
        if av[i] != 0:
            total += a[i]
        i += 1
    return total


def sum_all_range(a: FPtr, start: Int, end: Int) -> Float64:
    comptime W = simdwidthof[DType.float64]()
    var total0 = SIMD[DType.float64, W](0.0)
    var total1 = SIMD[DType.float64, W](0.0)
    var total2 = SIMD[DType.float64, W](0.0)
    var total3 = SIMD[DType.float64, W](0.0)
    var i = start
    while i + 4 * W <= end:
        total0 += a.load[width=W](i)
        total1 += a.load[width=W](i + W)
        total2 += a.load[width=W](i + 2 * W)
        total3 += a.load[width=W](i + 3 * W)
        i += 4 * W
    while i + W <= end:
        total0 += a.load[width=W](i)
        i += W
    var total = (total0 + total1 + total2 + total3).reduce_add()
    while i < end:
        total += a[i]
        i += 1
    return total


@no_inline
def parallel_sum(
    a_addr: Int, av_addr: Int, work_addr: Int, n: Int, all_valid: Bool
) -> Float64:
    if n < SUM_PARALLEL_THRESHOLD:
        if all_valid:
            return sum_all_range(fp(a_addr), 0, n)
        return sum_range(fp(a_addr), bp(av_addr), 0, n)
    var chunk = (n + PARALLEL_TASKS - 1) // PARALLEL_TASKS

    @__copy_capture(a_addr, av_addr, work_addr, n, chunk, all_valid)
    @parameter
    def work(task: Int):
        var start = task * chunk
        var end = min(start + chunk, n)
        if start >= end:
            fp(work_addr)[task] = 0.0
        elif all_valid:
            fp(work_addr)[task] = sum_all_range(fp(a_addr), start, end)
        else:
            fp(work_addr)[task] = sum_range(
                fp(a_addr), bp(av_addr), start, end
            )

    sync_parallelize[work](PARALLEL_TASKS)
    var total = 0.0
    for task in range(PARALLEL_TASKS):
        total += fp(work_addr)[task]
    return total


def cast_int_range(a: FPtr, dst: IPtr, start: Int, end: Int):
    comptime W = simdwidthof[DType.float64]()
    var i = start
    while i + W <= end:
        dst.store(i, a.load[width=W](i).cast[DType.int64]())
        i += W
    while i < end:
        dst[i] = Int64(a[i])
        i += 1


@no_inline
def cast_int(a_addr: Int, dst_addr: Int, n: Int):
    if n < PARALLEL_THRESHOLD:
        cast_int_range(fp(a_addr), ip(dst_addr), 0, n)
        return
    var chunk = (n + PARALLEL_TASKS - 1) // PARALLEL_TASKS

    @__copy_capture(a_addr, dst_addr, n, chunk)
    @parameter
    def work(task: Int):
        var start = task * chunk
        var end = min(start + chunk, n)
        if start < end:
            cast_int_range(
                fp(a_addr), ip(dst_addr), start, end
            )

    sync_parallelize[work](PARALLEL_TASKS)


@export("mp_fused_math")
def mp_fused_math(
    a_addr: Int,
    av_addr: Int,
    b_addr: Int,
    bv_addr: Int,
    scale: Float64,
    dst_addr: Int,
    dv_addr: Int,
    n: Int,
) abi("C"):
    fused_math(
        a_addr, av_addr, b_addr, bv_addr, scale, dst_addr, dv_addr, n
    )


@export("mp_fused_compare")
def mp_fused_compare(
    a_addr: Int,
    av_addr: Int,
    b_addr: Int,
    bv_addr: Int,
    a_scalar: Float64,
    b_scalar: Float64,
    a_op: Int,
    b_op: Int,
    a_scalar_left: Int,
    b_scalar_left: Int,
    dst_addr: Int,
    dv_addr: Int,
    n: Int,
) abi("C"):
    fused_compare(
        a_addr,
        av_addr,
        b_addr,
        bv_addr,
        a_scalar,
        b_scalar,
        a_op,
        b_op,
        a_scalar_left != 0,
        b_scalar_left != 0,
        dst_addr,
        dv_addr,
        n,
    )


@export("mp_cast_int")
def mp_cast_int(a_addr: Int, dst_addr: Int, n: Int) abi("C"):
    cast_int(a_addr, dst_addr, n)


@export("mp_binary")
def mp_binary(
    a_addr: Int,
    av_addr: Int,
    b_addr: Int,
    bv_addr: Int,
    dst_addr: Int,
    dv_addr: Int,
    n: Int,
    op: Int,
) abi("C"):
    var a = fp(a_addr)
    var av = bp(av_addr)
    var b = fp(b_addr)
    var bv = bp(bv_addr)
    var dst = fp(dst_addr)
    var dv = bp(dv_addr)
    if op == 0:
        binary_fast(op, a_addr, av_addr, b_addr, bv_addr, dst_addr, dv_addr, n)
        return
    if op == 2:
        binary_fast(op, a_addr, av_addr, b_addr, bv_addr, dst_addr, dv_addr, n)
        return
    if op == 9:
        binary_fast(op, a_addr, av_addr, b_addr, bv_addr, dst_addr, dv_addr, n)
        return
    if op == 10:
        binary_fast(op, a_addr, av_addr, b_addr, bv_addr, dst_addr, dv_addr, n)
        return
    if op == 11:
        binary_fast(op, a_addr, av_addr, b_addr, bv_addr, dst_addr, dv_addr, n)
        return
    if op == 12:
        binary_fast(op, a_addr, av_addr, b_addr, bv_addr, dst_addr, dv_addr, n)
        return
    if op == 13:
        binary_fast(op, a_addr, av_addr, b_addr, bv_addr, dst_addr, dv_addr, n)
        return
    comptime W = simdwidthof[DType.float64]()
    var i = 0
    while i + W <= n:
        var va = a.load[width=W](i)
        var vb = b.load[width=W](i)
        var result: SIMD[DType.float64, W]
        if op == 0:
            result = va + vb
        elif op == 1:
            result = va - vb
        elif op == 2:
            result = va * vb
        elif op == 3:
            result = va / vb
        elif op == 4:
            result = floor(va / vb)
        elif op == 5:
            result = va - floor(va / vb) * vb
        elif op == 6:
            result = pow(va, vb)
        elif op == 7:
            result = va.eq(vb).select(
                SIMD[DType.float64, W](1.0), SIMD[DType.float64, W](0.0)
            )
        elif op == 8:
            result = va.ne(vb).select(
                SIMD[DType.float64, W](1.0), SIMD[DType.float64, W](0.0)
            )
        elif op == 9:
            result = va.lt(vb).select(
                SIMD[DType.float64, W](1.0), SIMD[DType.float64, W](0.0)
            )
        elif op == 10:
            result = va.le(vb).select(
                SIMD[DType.float64, W](1.0), SIMD[DType.float64, W](0.0)
            )
        elif op == 11:
            result = va.gt(vb).select(
                SIMD[DType.float64, W](1.0), SIMD[DType.float64, W](0.0)
            )
        elif op == 12:
            result = va.ge(vb).select(
                SIMD[DType.float64, W](1.0), SIMD[DType.float64, W](0.0)
            )
        elif op == 13:
            result = (va.ne(0.0) & vb.ne(0.0)).select(
                SIMD[DType.float64, W](1.0), SIMD[DType.float64, W](0.0)
            )
        elif op == 14:
            result = (va.ne(0.0) | vb.ne(0.0)).select(
                SIMD[DType.float64, W](1.0), SIMD[DType.float64, W](0.0)
            )
        elif op == 15:
            result = min(va, vb)
        else:
            result = max(va, vb)
        if op == 14:
            var a_valid = av.load[width=W](i).ne(0)
            var b_valid = bv.load[width=W](i).ne(0)
            var a_true = a_valid & va.ne(0.0)
            var b_true = b_valid & vb.ne(0.0)
            result = (a_true | b_true).select(
                SIMD[DType.float64, W](1.0), SIMD[DType.float64, W](0.0)
            )
            dv.store(
                i,
                ((a_valid & b_valid) | a_true | b_true).select(
                    SIMD[DType.uint8, W](1), SIMD[DType.uint8, W](0)
                ),
            )
        else:
            dv.store(
                i,
                av.load[width=W](i) & bv.load[width=W](i),
            )
        dst.store(i, result)
        i += W
    while i < n:
        var valid = av[i] != 0 and bv[i] != 0
        if op == 14:
            valid = valid or (av[i] != 0 and a[i] != 0.0) or (
                bv[i] != 0 and b[i] != 0.0
            )
        dv[i] = UInt8(1) if valid else UInt8(0)
        if valid:
            if op == 14:
                dst[i] = (
                    1.0
                    if av[i] != 0 and a[i] != 0.0
                    or bv[i] != 0 and b[i] != 0.0
                    else 0.0
                )
            else:
                dst[i] = binary_value(a[i], b[i], op)
        i += 1


@export("mp_scalar")
def mp_scalar(
    a_addr: Int,
    av_addr: Int,
    scalar: Float64,
    dst_addr: Int,
    dv_addr: Int,
    n: Int,
    op: Int,
    scalar_left: Int,
) abi("C"):
    var a = fp(a_addr)
    var av = bp(av_addr)
    var dst = fp(dst_addr)
    var dv = bp(dv_addr)
    if op == 0:
        scalar_fast(
            op,
            a_addr,
            av_addr,
            scalar,
            dst_addr,
            dv_addr,
            n,
            scalar_left != 0,
        )
        return
    if op == 2:
        scalar_fast(
            op,
            a_addr,
            av_addr,
            scalar,
            dst_addr,
            dv_addr,
            n,
            scalar_left != 0,
        )
        return
    if op == 9:
        scalar_fast(
            op,
            a_addr,
            av_addr,
            scalar,
            dst_addr,
            dv_addr,
            n,
            scalar_left != 0,
        )
        return
    if op == 10:
        scalar_fast(
            op,
            a_addr,
            av_addr,
            scalar,
            dst_addr,
            dv_addr,
            n,
            scalar_left != 0,
        )
        return
    if op == 11:
        scalar_fast(
            op,
            a_addr,
            av_addr,
            scalar,
            dst_addr,
            dv_addr,
            n,
            scalar_left != 0,
        )
        return
    if op == 12:
        scalar_fast(
            op,
            a_addr,
            av_addr,
            scalar,
            dst_addr,
            dv_addr,
            n,
            scalar_left != 0,
        )
        return
    comptime W = simdwidthof[DType.float64]()
    var i = 0
    while i + W <= n:
        var va = a.load[width=W](i)
        var vb = SIMD[DType.float64, W](scalar)
        var left = vb if scalar_left != 0 else va
        var right = va if scalar_left != 0 else vb
        var result: SIMD[DType.float64, W]
        if op == 0:
            result = left + right
        elif op == 1:
            result = left - right
        elif op == 2:
            result = left * right
        elif op == 3:
            result = left / right
        elif op == 4:
            result = floor(left / right)
        elif op == 5:
            result = left - floor(left / right) * right
        elif op == 6:
            result = pow(left, right)
        elif op == 7:
            result = left.eq(right).select(
                SIMD[DType.float64, W](1.0), SIMD[DType.float64, W](0.0)
            )
        elif op == 8:
            result = left.ne(right).select(
                SIMD[DType.float64, W](1.0), SIMD[DType.float64, W](0.0)
            )
        elif op == 9:
            result = left.lt(right).select(
                SIMD[DType.float64, W](1.0), SIMD[DType.float64, W](0.0)
            )
        elif op == 10:
            result = left.le(right).select(
                SIMD[DType.float64, W](1.0), SIMD[DType.float64, W](0.0)
            )
        elif op == 11:
            result = left.gt(right).select(
                SIMD[DType.float64, W](1.0), SIMD[DType.float64, W](0.0)
            )
        elif op == 12:
            result = left.ge(right).select(
                SIMD[DType.float64, W](1.0), SIMD[DType.float64, W](0.0)
            )
        elif op == 13:
            result = (left.ne(0.0) & right.ne(0.0)).select(
                SIMD[DType.float64, W](1.0), SIMD[DType.float64, W](0.0)
            )
        elif op == 14:
            result = (left.ne(0.0) | right.ne(0.0)).select(
                SIMD[DType.float64, W](1.0), SIMD[DType.float64, W](0.0)
            )
        elif op == 15:
            result = min(left, right)
        else:
            result = max(left, right)
        if op == 13:
            var valid = av.load[width=W](i).ne(0) | vb.eq(0.0)
            result = (av.load[width=W](i).ne(0) & va.ne(0.0) & vb.ne(0.0)).select(
                SIMD[DType.float64, W](1.0), SIMD[DType.float64, W](0.0)
            )
            dv.store(i, valid.select(SIMD[DType.uint8, W](1), SIMD[DType.uint8, W](0)))
        elif op == 14:
            var valid = av.load[width=W](i).ne(0) | vb.ne(0.0)
            result = (
                vb.ne(0.0) | (av.load[width=W](i).ne(0) & va.ne(0.0))
            ).select(
                SIMD[DType.float64, W](1.0), SIMD[DType.float64, W](0.0)
            )
            dv.store(i, valid.select(SIMD[DType.uint8, W](1), SIMD[DType.uint8, W](0)))
        else:
            dv.store(i, av.load[width=W](i))
        dst.store(i, result)
        i += W
    while i < n:
        var valid = av[i] != 0
        if op == 13:
            valid = valid or scalar == 0.0
        elif op == 14:
            valid = valid or scalar != 0.0
        dv[i] = UInt8(1) if valid else UInt8(0)
        if valid:
            if op == 13:
                dst[i] = (
                    1.0
                    if av[i] != 0 and a[i] != 0.0 and scalar != 0.0
                    else 0.0
                )
                i += 1
                continue
            if op == 14:
                dst[i] = (
                    1.0
                    if scalar != 0.0 or av[i] != 0 and a[i] != 0.0
                    else 0.0
                )
                i += 1
                continue
            dst[i] = (
                binary_value(scalar, a[i], op)
                if scalar_left != 0
                else binary_value(a[i], scalar, op)
            )
        i += 1


@export("mp_unary")
def mp_unary(
    a_addr: Int,
    av_addr: Int,
    dst_addr: Int,
    dv_addr: Int,
    n: Int,
    op: Int,
    p0: Float64,
    p1: Float64,
) abi("C"):
    var a = fp(a_addr)
    var av = bp(av_addr)
    var dst = fp(dst_addr)
    var dv = bp(dv_addr)
    if op == 1:
        unary_fast(op, a_addr, av_addr, dst_addr, dv_addr, n)
        return
    if op == 2:
        unary_fast(op, a_addr, av_addr, dst_addr, dv_addr, n)
        return
    comptime W = simdwidthof[DType.float64]()
    var i = 0
    if op != 4 and op != 12:
        while i + W <= n:
            var value = a.load[width=W](i)
            var result: SIMD[DType.float64, W]
            if op == 0:
                result = -value
            elif op == 1:
                result = abs(value)
            elif op == 2:
                result = sqrt(value)
            elif op == 3:
                result = exp(value)
            elif op == 5:
                result = sin(value)
            elif op == 6:
                result = cos(value)
            elif op == 7:
                result = tan(value)
            elif op == 8:
                result = floor(value)
            elif op == 9:
                result = ceil(value)
            elif op == 10:
                result = value.lt(0.0).select(
                    SIMD[DType.float64, W](-1.0),
                    value.gt(0.0).select(
                        SIMD[DType.float64, W](1.0),
                        SIMD[DType.float64, W](0.0),
                    ),
                )
            elif op == 11:
                result = isnan(value).select(
                    SIMD[DType.float64, W](1.0),
                    SIMD[DType.float64, W](0.0),
                )
            elif op == 13:
                result = isnan(value).select(
                    SIMD[DType.float64, W](p0), value
                )
            elif op == 14:
                result = min(max(value, p0), p1)
            elif op == 15:
                result = value.ne(0.0).select(
                    SIMD[DType.float64, W](0.0),
                    SIMD[DType.float64, W](1.0),
                )
            else:
                result = value
            dst.store(i, result)
            dv.store(i, av.load[width=W](i))
            i += W
    while i < n:
        dv[i] = av[i]
        if av[i] == 0:
            i += 1
            continue
        var value = a[i]
        if op == 0:
            dst[i] = -value
        elif op == 1:
            dst[i] = abs(value)
        elif op == 2:
            dst[i] = sqrt(value)
        elif op == 3:
            dst[i] = exp(value)
        elif op == 4:
            dst[i] = precise_log(value)
        elif op == 5:
            dst[i] = sin(value)
        elif op == 6:
            dst[i] = cos(value)
        elif op == 7:
            dst[i] = tan(value)
        elif op == 8:
            dst[i] = floor(value)
        elif op == 9:
            dst[i] = ceil(value)
        elif op == 10:
            dst[i] = -1.0 if value < 0.0 else (1.0 if value > 0.0 else 0.0)
        elif op == 11:
            dst[i] = 1.0 if isnan(value) else 0.0
        elif op == 12:
            dst[i] = 0.0 if isnan(value) or abs(value) == 1.0 / 0.0 else 1.0
        elif op == 13:
            dst[i] = p0 if isnan(value) else value
        elif op == 14:
            dst[i] = min(max(value, p0), p1)
        elif op == 15:
            dst[i] = 0.0 if value != 0.0 else 1.0
        else:
            dst[i] = value
        i += 1


@export("mp_reduce")
def mp_reduce(
    a_addr: Int,
    av_addr: Int,
    work_addr: Int,
    n: Int,
    op: Int,
    ddof: Int,
    all_valid: Int,
) abi("C") -> Float64:
    var a = fp(a_addr)
    var av = bp(av_addr)
    if op == 0:
        return parallel_sum(
            a_addr, av_addr, work_addr, n, all_valid != 0
        )
    if op == 4:
        var product = 1.0
        for i in range(n):
            if av[i] != 0:
                product *= a[i]
        return product
    if op == 5:
        var count = 0
        for i in range(n):
            if av[i] != 0:
                count += 1
        return Float64(count)
    if op == 6:
        return Float64(n)
    if op == 9 or op == 10:
        var count = 0
        var truth = 0
        for i in range(n):
            if av[i] != 0:
                count += 1
                if a[i] != 0.0:
                    truth += 1
        return (
            1.0 if truth > 0 else 0.0
        ) if op == 9 else (1.0 if truth == count else 0.0)
    var count = 0
    var best = 0.0
    var mean = 0.0
    var moment2 = 0.0
    for i in range(n):
        if av[i] == 0:
            continue
        var value = a[i]
        count += 1
        if count == 1 or value < best:
            if op == 2:
                best = value
        if count == 1 or value > best:
            if op == 3:
                best = value
        var delta = value - mean
        mean += delta / Float64(count)
        moment2 += delta * (value - mean)
    if op == 1:
        return mean if count > 0 else nan_value()
    if op == 2 or op == 3:
        return best if count > 0 else nan_value()
    if op == 7 or op == 8:
        if count <= ddof:
            return nan_value()
        var variance = moment2 / Float64(count - ddof)
        return sqrt(max(variance, 0.0)) if op == 8 else variance
    return nan_value()


@export("mp_arg_reduce")
def mp_arg_reduce(a_addr: Int, av_addr: Int, n: Int, take_max: Int) abi("C") -> Int:
    var a = fp(a_addr)
    var av = bp(av_addr)
    var found = False
    var best = 0.0
    var index = -1
    for i in range(n):
        if av[i] == 0:
            continue
        var value = a[i]
        if not found or (take_max != 0 and value > best) or (
            take_max == 0 and value < best
        ):
            found = True
            best = value
            index = i
    return index


@export("mp_cumulative")
def mp_cumulative(
    a_addr: Int,
    av_addr: Int,
    dst_addr: Int,
    dv_addr: Int,
    n: Int,
    op: Int,
    reverse: Int,
) abi("C"):
    var a = fp(a_addr)
    var av = bp(av_addr)
    var dst = fp(dst_addr)
    var dv = bp(dv_addr)
    var acc = 1.0 if op == 1 else 0.0
    var seen = False
    for step in range(n):
        var i = n - 1 - step if reverse != 0 else step
        dv[i] = av[i]
        if av[i] == 0:
            continue
        var value = a[i]
        if op == 0:
            acc += value
        elif op == 1:
            acc *= value
        elif op == 2:
            acc = value if not seen else min(acc, value)
        else:
            acc = value if not seen else max(acc, value)
        seen = True
        dst[i] = acc


@export("mp_rolling")
def mp_rolling(
    a_addr: Int,
    av_addr: Int,
    dst_addr: Int,
    dv_addr: Int,
    n: Int,
    queue_addr: Int,
    window: Int,
    min_periods: Int,
    op: Int,
    ddof: Int,
) abi("C"):
    var a = fp(a_addr)
    var av = bp(av_addr)
    var dst = fp(dst_addr)
    var dv = bp(dv_addr)
    var queue = ip(queue_addr)
    var total = 0.0
    var mean = 0.0
    var moment2 = 0.0
    var count = 0
    var head = 0
    var tail = 0
    for i in range(n):
        var value = a[i]
        if av[i] != 0:
            count += 1
            total += value
            var delta = value - mean
            mean += delta / Float64(count)
            moment2 += delta * (value - mean)
        if i >= window and av[i - window] != 0:
            var old = a[i - window]
            total -= old
            if count == 1:
                count = 0
                mean = 0.0
                moment2 = 0.0
            else:
                var new_count = count - 1
                var new_mean = (Float64(count) * mean - old) / Float64(new_count)
                moment2 -= (old - mean) * (old - new_mean)
                mean = new_mean
                count = new_count
                if moment2 < 0.0:
                    moment2 = 0.0
        if op == 2 or op == 3:
            var first = i - window + 1
            while head < tail and Int(queue[head]) < first:
                head += 1
            if av[i] != 0:
                while head < tail:
                    var queued = a[Int(queue[tail - 1])]
                    var remove = queued >= value if op == 2 else queued <= value
                    if not remove:
                        break
                    tail -= 1
                queue[tail] = Int64(i)
                tail += 1
        var valid = count >= min_periods
        if op == 4 or op == 5:
            valid = valid and count > ddof
        dv[i] = UInt8(1) if valid else UInt8(0)
        if not valid:
            continue
        if op == 0:
            dst[i] = total
        elif op == 1:
            dst[i] = mean
        elif op == 2 or op == 3:
            dst[i] = a[Int(queue[head])]
        else:
            var variance = max(moment2, 0.0) / Float64(count - ddof)
            dst[i] = sqrt(variance) if op == 5 else variance


@export("mp_filter")
def mp_filter(
    a_addr: Int,
    av_addr: Int,
    pred_addr: Int,
    pv_addr: Int,
    dst_addr: Int,
    dv_addr: Int,
    n: Int,
) abi("C") -> Int:
    var a = fp(a_addr)
    var av = bp(av_addr)
    var pred = fp(pred_addr)
    var pv = bp(pv_addr)
    var dst = fp(dst_addr)
    var dv = bp(dv_addr)
    var written = 0
    for i in range(n):
        if pv[i] != 0 and pred[i] != 0.0:
            dst[written] = a[i]
            dv[written] = av[i]
            written += 1
    return written


@export("mp_group_reduce")
def mp_group_reduce(
    a_addr: Int,
    av_addr: Int,
    labels_addr: Int,
    dst_addr: Int,
    work_addr: Int,
    dv_addr: Int,
    counts_addr: Int,
    n: Int,
    ngroups: Int,
    op: Int,
    ddof: Int,
) abi("C"):
    var a = fp(a_addr)
    var av = bp(av_addr)
    var labels = ip(labels_addr)
    var dst = fp(dst_addr)
    var work = fp(work_addr)
    var dv = bp(dv_addr)
    var counts = ip(counts_addr)
    for g in range(ngroups):
        dst[g] = 0.0
        work[g] = 0.0
        dv[g] = UInt8(0)
        counts[g] = 0
    for i in range(n):
        var g = Int(labels[i])
        if g < 0 or g >= ngroups:
            continue
        if op == 5:
            if work[g] == 0.0:
                work[g] = 1.0
                dst[g] = a[i]
                dv[g] = av[i]
            continue
        if op == 6:
            work[g] = 1.0
            dst[g] = a[i]
            dv[g] = av[i]
            continue
        if av[i] == 0:
            continue
        var value = a[i]
        counts[g] += 1
        if op == 0 or op == 1:
            dst[g] += value
        elif op == 2:
            if counts[g] == 1 or value < dst[g]:
                dst[g] = value
        elif op == 3:
            if counts[g] == 1 or value > dst[g]:
                dst[g] = value
        elif op == 4:
            dst[g] = Float64(counts[g])
        elif op == 7 or op == 8:
            var delta = value - dst[g]
            dst[g] += delta / Float64(counts[g])
            work[g] += delta * (value - dst[g])
    for g in range(ngroups):
        if op == 5 or op == 6:
            continue
        elif op == 0:
            dv[g] = UInt8(1)
        elif op == 4:
            dst[g] = Float64(counts[g])
            dv[g] = UInt8(1)
        elif counts[g] > 0 and op != 7 and op != 8:
            if op == 1:
                dst[g] /= Float64(counts[g])
            dv[g] = UInt8(1)
        elif op == 7 or op == 8:
            if Int(counts[g]) > ddof:
                dst[g] = work[g] / Float64(Int(counts[g]) - ddof)
                if op == 8:
                    dst[g] = sqrt(max(dst[g], 0.0))
                dv[g] = UInt8(1)
            else:
                dv[g] = UInt8(0)
