"""Polars-shaped Series, Expr, DataFrame, and GroupBy objects."""

from __future__ import annotations

import builtins
import math
import operator
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from functools import cmp_to_key
from typing import Any

import numpy as np

from . import _kernels as K


class DataType:
    def __init__(self, name: str):
        self.name = name

    def __repr__(self) -> str:
        return self.name

    def __hash__(self) -> int:
        return hash(self.name)

    def __eq__(self, other: object) -> bool:
        return isinstance(other, DataType) and self.name == other.name


Float64 = DataType("Float64")
Int64 = DataType("Int64")
Boolean = DataType("Boolean")
String = DataType("String")
Utf8 = String


def _dtype_kind(dtype: DataType | type | str | None) -> str | None:
    if dtype in (Float64, float, "float", "float64"):
        return "float"
    if dtype in (Int64, int, "int", "int64"):
        return "int"
    if dtype in (Boolean, bool, "bool", "boolean"):
        return "bool"
    if dtype in (String, str, "str", "string"):
        return "str"
    return None


def _dtype_for(kind: str) -> DataType:
    return {"float": Float64, "int": Int64, "bool": Boolean, "str": String}[kind]


def _as_list(values: Any) -> list[Any]:
    if isinstance(values, Series):
        return values.to_list()
    if isinstance(values, np.ndarray):
        return values.tolist()
    if isinstance(values, (str, bytes)) or not isinstance(values, Iterable):
        return [values]
    return list(values)


class Series:
    def __init__(
        self,
        name: str | Sequence[Any] | np.ndarray | None = None,
        values: Sequence[Any] | np.ndarray | None = None,
        dtype: DataType | type | str | None = None,
        *,
        strict: bool = True,
        nan_to_null: bool = False,
        _valid: np.ndarray | None = None,
        _kind: str | None = None,
    ):
        del strict
        if values is None and name is not None and not isinstance(name, str):
            values, name = name, ""
        if name is None:
            name = ""
        kind = _kind or _dtype_kind(dtype)
        array = values if isinstance(values, np.ndarray) else None
        if array is not None and array.ndim != 1:
            array = np.ravel(array)
        if array is not None and array.dtype.kind != "O":
            if kind is None:
                if array.dtype.kind == "b":
                    kind = "bool"
                elif array.dtype.kind in "iu":
                    kind = "int"
                elif array.dtype.kind in "SU":
                    kind = "str"
                else:
                    kind = "float"
            self._name = str(name)
            self._kind = kind
            if kind == "int" and array.dtype.kind in "iu":
                exact_limit = 1 << 53
                if array.size and (
                    np.any(array > exact_limit) or np.any(array < -exact_limit)
                ):
                    raise OverflowError(
                        "Int64 values outside [-2**53, 2**53] cannot be "
                        "represented exactly by mojo-polars"
                    )
            if _valid is None:
                if nan_to_null and array.dtype.kind == "f":
                    valid = (~np.isnan(array)).astype(np.uint8)
                else:
                    valid = np.ones(array.size, dtype=np.uint8)
            else:
                valid = np.ascontiguousarray(_valid, dtype=np.uint8)
            self._valid = valid
            if kind == "str":
                self._values = np.asarray(array, dtype=object)
            else:
                self._values = np.ascontiguousarray(array, dtype=np.float64)
            self._native_int = (
                np.ascontiguousarray(array, dtype=np.int64)
                if kind == "int" and array.dtype.kind in "iu"
                else None
            )
            return
        raw = [] if values is None else _as_list(values)
        nonnull = [value for value in raw if value is not None]
        if kind is None:
            if nonnull and builtins.all(
                isinstance(value, (bool, np.bool_)) for value in nonnull
            ):
                kind = "bool"
            elif nonnull and builtins.all(
                isinstance(value, (int, np.integer))
                and not isinstance(value, (bool, np.bool_))
                for value in nonnull
            ):
                kind = "int"
            elif nonnull and builtins.all(
                isinstance(value, (str, np.str_)) for value in nonnull
            ):
                kind = "str"
            else:
                kind = "float"
        self._name = str(name)
        self._kind = kind
        if kind == "int" and any(
            abs(int(value)) > (1 << 53) for value in nonnull
        ):
            raise OverflowError(
                "Int64 values outside [-2**53, 2**53] cannot be represented "
                "exactly by mojo-polars"
            )
        if _valid is None:
            valid = np.fromiter(
                (
                    value is not None
                    and not (
                        nan_to_null
                        and isinstance(value, (float, np.floating))
                        and math.isnan(float(value))
                    )
                    for value in raw
                ),
                dtype=np.uint8,
                count=len(raw),
            )
        else:
            valid = np.ascontiguousarray(_valid, dtype=np.uint8)
        self._valid = valid
        if kind == "str":
            self._values = np.asarray(
                ["" if value is None else str(value) for value in raw], dtype=object
            )
        else:
            self._values = np.ascontiguousarray(
                [0.0 if value is None else float(value) for value in raw],
                dtype=np.float64,
            )
        self._native_int = None

    @classmethod
    def _from_arrays(
        cls,
        name: str,
        values: np.ndarray,
        valid: np.ndarray,
        kind: str = "float",
        native_int: np.ndarray | None = None,
    ) -> Series:
        instance = cls.__new__(cls)
        instance._name = name
        instance._kind = kind
        instance._values = np.ascontiguousarray(
            values, dtype=object if kind == "str" else np.float64
        )
        instance._valid = np.ascontiguousarray(valid, dtype=np.uint8)
        instance._native_int = native_int
        return instance

    @property
    def name(self) -> str:
        return self._name

    @property
    def dtype(self) -> DataType:
        return _dtype_for(self._kind)

    @property
    def shape(self) -> tuple[int]:
        return (len(self),)

    @property
    def null_count(self) -> int:
        return int((self._valid == 0).sum())

    def __len__(self) -> int:
        return self._values.size

    def __iter__(self):
        return iter(self.to_list())

    def __repr__(self) -> str:
        return f"Series({self._name!r}, {self.to_list()!r}, dtype={self.dtype!r})"

    def __array__(self, dtype=None, copy=None) -> np.ndarray:
        result = self.to_numpy()
        if dtype is not None:
            result = result.astype(dtype, copy=False)
        if copy:
            result = result.copy()
        return result

    def __getitem__(self, item):
        if isinstance(item, slice):
            return Series._from_arrays(
                self.name,
                self._values[item],
                self._valid[item],
                self._kind,
                (
                    None
                    if self._native_int is None
                    else self._native_int[item]
                ),
            )
        index = int(item)
        if not self._valid[index]:
            return None
        return self._python_value(self._values[index])

    def _python_value(self, value):
        if self._kind == "int":
            return int(value)
        if self._kind == "bool":
            return bool(value)
        if self._kind == "float":
            return float(value)
        return str(value)

    def to_list(self) -> list[Any]:
        return [
            self._python_value(value) if valid else None
            for value, valid in zip(self._values, self._valid)
        ]

    def to_numpy(
        self,
        *,
        writable: bool = False,
        allow_copy: bool = True,
        use_pyarrow: bool | None = None,
        zero_copy_only: bool | None = None,
    ) -> np.ndarray:
        del writable, allow_copy, use_pyarrow, zero_copy_only
        if self._kind == "str":
            return np.asarray(self.to_list(), dtype=object)
        if self.null_count:
            result = self._values.copy()
            result[self._valid == 0] = np.nan
            return result
        if self._kind == "int":
            return self._values.astype(np.int64)
        if self._kind == "bool":
            return self._values.astype(bool)
        return self._values

    def item(self, index: int | None = None):
        if index is None:
            if len(self) != 1:
                raise ValueError("can only call item on a Series of length 1")
            index = 0
        return self[index]

    def rename(self, name: str) -> Series:
        return Series._from_arrays(
            name,
            self._values,
            self._valid,
            self._kind,
            self._native_int,
        )

    def clone(self) -> Series:
        return Series._from_arrays(
            self.name,
            self._values.copy(),
            self._valid.copy(),
            self._kind,
            (
                None
                if self._native_int is None
                else self._native_int.copy()
            ),
        )

    def _expr(self) -> Expr:
        return Expr("series", (self,), output_name=self.name)

    def _eager(self, expression: Expr):
        result = expression._eval({})
        return result.item() if expression._op == "agg" else result

    def _binary(self, other, op: str, reverse: bool = False) -> Series:
        expression = (
            _to_expr(other)._binary(self._expr(), op)
            if reverse
            else self._expr()._binary(other, op)
        )
        return expression._eval({})

    def __add__(self, other):
        return self._binary(other, "add")

    def __radd__(self, other):
        return self._binary(other, "add", True)

    def __sub__(self, other):
        return self._binary(other, "sub")

    def __rsub__(self, other):
        return self._binary(other, "sub", True)

    def __mul__(self, other):
        return self._binary(other, "mul")

    def __rmul__(self, other):
        return self._binary(other, "mul", True)

    def __truediv__(self, other):
        return self._binary(other, "truediv")

    def __rtruediv__(self, other):
        return self._binary(other, "truediv", True)

    def __floordiv__(self, other):
        return self._binary(other, "floordiv")

    def __rfloordiv__(self, other):
        return self._binary(other, "floordiv", True)

    def __mod__(self, other):
        return self._binary(other, "mod")

    def __rmod__(self, other):
        return self._binary(other, "mod", True)

    def __pow__(self, other):
        return self._binary(other, "pow")

    def __rpow__(self, other):
        return self._binary(other, "pow", True)

    def __eq__(self, other):
        return self._binary(other, "eq")

    def __ne__(self, other):
        return self._binary(other, "ne")

    def __lt__(self, other):
        return self._binary(other, "lt")

    def __le__(self, other):
        return self._binary(other, "le")

    def __gt__(self, other):
        return self._binary(other, "gt")

    def __ge__(self, other):
        return self._binary(other, "ge")

    def __and__(self, other):
        return self._binary(other, "and")

    def __or__(self, other):
        return self._binary(other, "or")

    def __invert__(self):
        return self._expr().__invert__()._eval({})

    def __neg__(self):
        return self._expr().__neg__()._eval({})

    def __abs__(self):
        return self.abs()

    def abs(self) -> Series:
        return self._expr().abs()._eval({})

    def sqrt(self) -> Series:
        return self._expr().sqrt()._eval({})

    def exp(self) -> Series:
        return self._expr().exp()._eval({})

    def log(self, base: float = math.e) -> Series:
        return self._expr().log(base)._eval({})

    def sin(self) -> Series:
        return self._expr().sin()._eval({})

    def cos(self) -> Series:
        return self._expr().cos()._eval({})

    def tan(self) -> Series:
        return self._expr().tan()._eval({})

    def floor(self) -> Series:
        return self._expr().floor()._eval({})

    def ceil(self) -> Series:
        return self._expr().ceil()._eval({})

    def round(self, decimals: int = 0, mode: str = "half_to_even") -> Series:
        return self._expr().round(decimals, mode)._eval({})

    def clip(self, lower_bound=None, upper_bound=None) -> Series:
        return self._expr().clip(lower_bound, upper_bound)._eval({})

    def is_null(self) -> Series:
        return self._expr().is_null()._eval({})

    def is_not_null(self) -> Series:
        return self._expr().is_not_null()._eval({})

    def is_nan(self) -> Series:
        return self._expr().is_nan()._eval({})

    def is_finite(self) -> Series:
        return self._expr().is_finite()._eval({})

    def fill_null(self, value: Any) -> Series:
        return self._expr().fill_null(value)._eval({})

    def fill_nan(self, value: float) -> Series:
        return self._expr().fill_nan(value)._eval({})

    def cast(self, dtype: DataType | type | str, *, strict: bool = True) -> Series:
        return self._expr().cast(dtype, strict=strict)._eval({})

    def sum(self):
        return self._expr().sum()._eval({}).item()

    def mean(self):
        return self._expr().mean()._eval({}).item()

    def min(self):
        return self._expr().min()._eval({}).item()

    def max(self):
        return self._expr().max()._eval({}).item()

    def product(self):
        return self._expr().product()._eval({}).item()

    def count(self) -> int:
        return int(self._expr().count()._eval({}).item())

    def n_unique(self) -> int:
        return int(self._expr().n_unique()._eval({}).item())

    def var(self, ddof: int = 1):
        return self._expr().var(ddof)._eval({}).item()

    def std(self, ddof: int = 1):
        return self._expr().std(ddof)._eval({}).item()

    def median(self):
        return self._expr().median()._eval({}).item()

    def quantile(self, quantile: float, interpolation: str = "nearest"):
        return self._expr().quantile(quantile, interpolation)._eval({}).item()

    def arg_min(self):
        return K.arg_reduce(self._numeric(), self._valid, False)

    def arg_max(self):
        return K.arg_reduce(self._numeric(), self._valid, True)

    def any(self, *, ignore_nulls: bool = True) -> bool | None:
        return self._expr().any(ignore_nulls=ignore_nulls)._eval({}).item()

    def all(self, *, ignore_nulls: bool = True) -> bool | None:
        return self._expr().all(ignore_nulls=ignore_nulls)._eval({}).item()

    def cum_sum(self, *, reverse: bool = False) -> Series:
        return self._expr().cum_sum(reverse=reverse)._eval({})

    def cum_prod(self, *, reverse: bool = False) -> Series:
        return self._expr().cum_prod(reverse=reverse)._eval({})

    def cum_min(self, *, reverse: bool = False) -> Series:
        return self._expr().cum_min(reverse=reverse)._eval({})

    def cum_max(self, *, reverse: bool = False) -> Series:
        return self._expr().cum_max(reverse=reverse)._eval({})

    def shift(self, n: int = 1, *, fill_value=None) -> Series:
        return self._expr().shift(n, fill_value=fill_value)._eval({})

    def diff(self, n: int = 1, null_behavior: str = "ignore") -> Series:
        return self._expr().diff(n, null_behavior)._eval({})

    def pct_change(self, n: int = 1) -> Series:
        return self._expr().pct_change(n)._eval({})

    def rolling_sum(
        self,
        window_size: int,
        weights=None,
        *,
        min_samples: int | None = None,
        center: bool = False,
    ) -> Series:
        return self._expr().rolling_sum(
            window_size, weights, min_samples=min_samples, center=center
        )._eval({})

    def rolling_mean(
        self,
        window_size: int,
        weights=None,
        *,
        min_samples: int | None = None,
        center: bool = False,
    ) -> Series:
        return self._expr().rolling_mean(
            window_size, weights, min_samples=min_samples, center=center
        )._eval({})

    def rolling_min(
        self, window_size: int, *, min_samples: int | None = None, center: bool = False
    ) -> Series:
        return self._expr().rolling_min(
            window_size, min_samples=min_samples, center=center
        )._eval({})

    def rolling_max(
        self, window_size: int, *, min_samples: int | None = None, center: bool = False
    ) -> Series:
        return self._expr().rolling_max(
            window_size, min_samples=min_samples, center=center
        )._eval({})

    def rolling_var(
        self,
        window_size: int,
        weights=None,
        *,
        min_samples: int | None = None,
        center: bool = False,
        ddof: int = 1,
    ) -> Series:
        return self._expr().rolling_var(
            window_size,
            weights,
            min_samples=min_samples,
            center=center,
            ddof=ddof,
        )._eval({})

    def rolling_std(
        self,
        window_size: int,
        weights=None,
        *,
        min_samples: int | None = None,
        center: bool = False,
        ddof: int = 1,
    ) -> Series:
        return self._expr().rolling_std(
            window_size,
            weights,
            min_samples=min_samples,
            center=center,
            ddof=ddof,
        )._eval({})

    def filter(self, predicate: Series) -> Series:
        if len(predicate) != len(self):
            raise ValueError("filter predicate length does not match Series")
        indexes = np.flatnonzero(
            (predicate._valid != 0) & (predicate._values != 0)
        )
        selected_valid = (
            np.ones(indexes.size, dtype=np.uint8)
            if self._valid.all()
            else self._valid[indexes]
        )
        return Series._from_arrays(
            self.name,
            self._values[indexes],
            selected_valid,
            self._kind,
        )

    def sort(
        self, *, descending: bool = False, nulls_last: bool = False, multithreaded=True
    ) -> Series:
        del multithreaded
        valid_indexes = sorted(
            (index for index in range(len(self)) if self._valid[index]),
            key=lambda index: self._values[index],
            reverse=descending,
        )
        null_indexes = [index for index in range(len(self)) if not self._valid[index]]
        indexes = valid_indexes + null_indexes if nulls_last else null_indexes + valid_indexes
        return Series._from_arrays(
            self.name, self._values[indexes], self._valid[indexes], self._kind
        )

    def head(self, n: int = 10) -> Series:
        return self[:n]

    def tail(self, n: int = 10) -> Series:
        return self[-n:]

    def _numeric(self) -> np.ndarray:
        if self._kind == "str":
            raise TypeError("string columns are not supported by numeric expression kernels")
        return self._values


_COMPARISONS = {"eq", "ne", "lt", "le", "gt", "ge", "and", "or"}


@dataclass(frozen=True)
class Expr:
    _op: str
    _args: tuple[Any, ...] = ()
    _kwargs: tuple[tuple[str, Any], ...] = ()
    _output_name: str = ""

    def __init__(
        self,
        op: str,
        args: tuple[Any, ...] = (),
        kwargs: Mapping[str, Any] | None = None,
        output_name: str = "",
    ):
        object.__setattr__(self, "_op", op)
        object.__setattr__(self, "_args", args)
        object.__setattr__(self, "_kwargs", tuple((kwargs or {}).items()))
        object.__setattr__(self, "_output_name", output_name)

    @property
    def _kw(self) -> dict[str, Any]:
        return dict(self._kwargs)

    def __repr__(self) -> str:
        return f"Expr({self._op!r}, {self._args!r})"

    def alias(self, name: str) -> Expr:
        return Expr("alias", (self,), {"name": name}, output_name=name)

    def _binary(self, other, op: str) -> Expr:
        other = _to_expr(other)
        return Expr(
            op,
            (self, other),
            output_name=self._output_name or other._output_name,
        )

    def __add__(self, other):
        return self._binary(other, "add")

    def __radd__(self, other):
        return _to_expr(other)._binary(self, "add")

    def __sub__(self, other):
        return self._binary(other, "sub")

    def __rsub__(self, other):
        return _to_expr(other)._binary(self, "sub")

    def __mul__(self, other):
        return self._binary(other, "mul")

    def __rmul__(self, other):
        return _to_expr(other)._binary(self, "mul")

    def __truediv__(self, other):
        return self._binary(other, "truediv")

    def __rtruediv__(self, other):
        return _to_expr(other)._binary(self, "truediv")

    def __floordiv__(self, other):
        return self._binary(other, "floordiv")

    def __rfloordiv__(self, other):
        return _to_expr(other)._binary(self, "floordiv")

    def __mod__(self, other):
        return self._binary(other, "mod")

    def __rmod__(self, other):
        return _to_expr(other)._binary(self, "mod")

    def __pow__(self, other):
        return self._binary(other, "pow")

    def __rpow__(self, other):
        return _to_expr(other)._binary(self, "pow")

    def __eq__(self, other):
        return self._binary(other, "eq")

    def __ne__(self, other):
        return self._binary(other, "ne")

    def __lt__(self, other):
        return self._binary(other, "lt")

    def __le__(self, other):
        return self._binary(other, "le")

    def __gt__(self, other):
        return self._binary(other, "gt")

    def __ge__(self, other):
        return self._binary(other, "ge")

    def __and__(self, other):
        return self._binary(other, "and")

    def __rand__(self, other):
        return _to_expr(other)._binary(self, "and")

    def __or__(self, other):
        return self._binary(other, "or")

    def __ror__(self, other):
        return _to_expr(other)._binary(self, "or")

    def __invert__(self):
        return Expr("unary", (self,), {"op": "not"}, self._output_name)

    def __neg__(self):
        return Expr("unary", (self,), {"op": "neg"}, self._output_name)

    def __abs__(self):
        return self.abs()

    def abs(self) -> Expr:
        return Expr("unary", (self,), {"op": "abs"}, self._output_name)

    def sqrt(self) -> Expr:
        return Expr("unary", (self,), {"op": "sqrt"}, self._output_name)

    def exp(self) -> Expr:
        return Expr("unary", (self,), {"op": "exp"}, self._output_name)

    def log(self, base: float = math.e) -> Expr:
        result = Expr("unary", (self,), {"op": "log"}, self._output_name)
        return result if base == math.e else result / math.log(base)

    def sin(self) -> Expr:
        return Expr("unary", (self,), {"op": "sin"}, self._output_name)

    def cos(self) -> Expr:
        return Expr("unary", (self,), {"op": "cos"}, self._output_name)

    def tan(self) -> Expr:
        return Expr("unary", (self,), {"op": "tan"}, self._output_name)

    def floor(self) -> Expr:
        return Expr("unary", (self,), {"op": "floor"}, self._output_name)

    def ceil(self) -> Expr:
        return Expr("unary", (self,), {"op": "ceil"}, self._output_name)

    def round(self, decimals: int = 0, mode: str = "half_to_even") -> Expr:
        return Expr(
            "round", (self,), {"decimals": decimals, "mode": mode}, self._output_name
        )

    def clip(self, lower_bound=None, upper_bound=None) -> Expr:
        return Expr(
            "clip",
            (self,),
            {"lower": lower_bound, "upper": upper_bound},
            self._output_name,
        )

    def is_null(self) -> Expr:
        return Expr("is_null", (self,), output_name=self._output_name)

    def is_not_null(self) -> Expr:
        return Expr("is_not_null", (self,), output_name=self._output_name)

    def is_nan(self) -> Expr:
        return Expr("unary", (self,), {"op": "is_nan"}, self._output_name)

    def is_finite(self) -> Expr:
        return Expr("unary", (self,), {"op": "is_finite"}, self._output_name)

    def fill_null(self, value: Any) -> Expr:
        return Expr("fill_null", (self,), {"value": value}, self._output_name)

    def fill_nan(self, value: float) -> Expr:
        return Expr("unary", (self,), {"op": "fill_nan", "p0": value}, self._output_name)

    def cast(self, dtype: DataType | type | str, *, strict: bool = True) -> Expr:
        return Expr(
            "cast", (self,), {"dtype": dtype, "strict": strict}, self._output_name
        )

    def _agg(self, op: str, **kwargs) -> Expr:
        return Expr("agg", (self,), {"op": op, **kwargs}, self._output_name)

    def sum(self) -> Expr:
        return self._agg("sum")

    def mean(self) -> Expr:
        return self._agg("mean")

    def min(self) -> Expr:
        return self._agg("min")

    def max(self) -> Expr:
        return self._agg("max")

    def product(self) -> Expr:
        return self._agg("product")

    def count(self) -> Expr:
        return self._agg("count")

    def len(self) -> Expr:
        return self._agg("len")

    def n_unique(self) -> Expr:
        return self._agg("n_unique")

    def first(self) -> Expr:
        return self._agg("first")

    def last(self) -> Expr:
        return self._agg("last")

    def var(self, ddof: int = 1) -> Expr:
        return self._agg("var", ddof=ddof)

    def std(self, ddof: int = 1) -> Expr:
        return self._agg("std", ddof=ddof)

    def median(self) -> Expr:
        return self._agg("median")

    def quantile(self, quantile: float, interpolation: str = "nearest") -> Expr:
        return self._agg("quantile", quantile=quantile, interpolation=interpolation)

    def any(self, *, ignore_nulls: bool = True) -> Expr:
        return self._agg("any", ignore_nulls=ignore_nulls)

    def all(self, *, ignore_nulls: bool = True) -> Expr:
        return self._agg("all", ignore_nulls=ignore_nulls)

    def _cum(self, op: str, reverse: bool) -> Expr:
        return Expr("cumulative", (self,), {"op": op, "reverse": reverse}, self._output_name)

    def cum_sum(self, *, reverse: bool = False) -> Expr:
        return self._cum("cum_sum", reverse)

    def cum_prod(self, *, reverse: bool = False) -> Expr:
        return self._cum("cum_prod", reverse)

    def cum_min(self, *, reverse: bool = False) -> Expr:
        return self._cum("cum_min", reverse)

    def cum_max(self, *, reverse: bool = False) -> Expr:
        return self._cum("cum_max", reverse)

    def shift(self, n: int = 1, *, fill_value=None) -> Expr:
        return Expr(
            "shift", (self,), {"n": n, "fill_value": fill_value}, self._output_name
        )

    def diff(self, n: int = 1, null_behavior: str = "ignore") -> Expr:
        return Expr(
            "diff", (self,), {"n": n, "null_behavior": null_behavior}, self._output_name
        )

    def pct_change(self, n: int = 1) -> Expr:
        return Expr("pct_change", (self,), {"n": n}, self._output_name)

    def _rolling(
        self,
        op: str,
        window_size: int,
        weights,
        min_samples: int | None,
        center: bool,
        ddof: int = 1,
    ) -> Expr:
        if weights is not None:
            raise NotImplementedError("weighted rolling windows are outside the covered subset")
        if center:
            raise NotImplementedError("centered rolling windows are outside the covered subset")
        return Expr(
            "rolling",
            (self,),
            {
                "op": op,
                "window_size": window_size,
                "min_samples": window_size if min_samples is None else min_samples,
                "ddof": ddof,
            },
            self._output_name,
        )

    def rolling_sum(
        self, window_size, weights=None, *, min_samples=None, center=False
    ) -> Expr:
        return self._rolling(
            "rolling_sum", window_size, weights, min_samples, center
        )

    def rolling_mean(
        self, window_size, weights=None, *, min_samples=None, center=False
    ) -> Expr:
        return self._rolling(
            "rolling_mean", window_size, weights, min_samples, center
        )

    def rolling_min(self, window_size, *, min_samples=None, center=False) -> Expr:
        return self._rolling(
            "rolling_min", window_size, None, min_samples, center
        )

    def rolling_max(self, window_size, *, min_samples=None, center=False) -> Expr:
        return self._rolling(
            "rolling_max", window_size, None, min_samples, center
        )

    def rolling_var(
        self,
        window_size,
        weights=None,
        *,
        min_samples=None,
        center=False,
        ddof=1,
    ) -> Expr:
        return self._rolling(
            "rolling_var", window_size, weights, min_samples, center, ddof
        )

    def rolling_std(
        self,
        window_size,
        weights=None,
        *,
        min_samples=None,
        center=False,
        ddof=1,
    ) -> Expr:
        return self._rolling(
            "rolling_std", window_size, weights, min_samples, center, ddof
        )

    def _eval(self, frame: Mapping[str, Series]) -> Series:
        kw = self._kw
        fused_math = _match_fused_math(self)
        if fused_math is not None:
            a_name, b_name, scale = fused_math
            a = frame[a_name]
            b = frame[b_name]
            values, valid = K.fused_math(
                a._numeric(),
                a._valid,
                b._numeric(),
                b._valid,
                scale,
            )
            return Series._from_arrays(self._output_name, values, valid)
        fused_compare = _match_fused_compare(self)
        if fused_compare is not None:
            (
                a_name,
                b_name,
                a_scalar,
                b_scalar,
                a_op,
                b_op,
                a_scalar_left,
                b_scalar_left,
            ) = fused_compare
            a = frame[a_name]
            b = frame[b_name]
            values, valid = K.fused_compare(
                a._numeric(),
                a._valid,
                b._numeric(),
                b._valid,
                a_scalar,
                b_scalar,
                a_op,
                b_op,
                a_scalar_left,
                b_scalar_left,
            )
            return Series._from_arrays(
                self._output_name, values, valid, "bool"
            )
        if self._op == "col":
            return frame[self._args[0]]
        if self._op == "series":
            return self._args[0]
        if self._op == "lit":
            return Series(self._output_name, [self._args[0]])
        if self._op == "alias":
            return self._args[0]._eval(frame).rename(kw["name"])
        if self._op in K.BINARY_OPS:
            left = self._args[0]._eval(frame)
            right = self._args[1]._eval(frame)
            if left._kind == "str" or right._kind == "str":
                left, right = _broadcast_pair(left, right)
                return _object_binary(left, right, self._op, self._output_name)
            if len(left) == len(right):
                values, valid = K.binary(
                    left._numeric(),
                    left._valid,
                    right._numeric(),
                    right._valid,
                    self._op,
                )
            elif len(right) == 1:
                if right._valid[0]:
                    values, valid = K.binary(
                        left._numeric(),
                        left._valid,
                        float(right._values[0]),
                        None,
                        self._op,
                    )
                else:
                    values = np.empty(len(left), dtype=np.float64)
                    valid = np.zeros(len(left), dtype=np.uint8)
            elif len(left) == 1:
                if left._valid[0]:
                    values, valid = K.binary(
                        right._numeric(),
                        right._valid,
                        float(left._values[0]),
                        None,
                        self._op,
                        scalar_left=True,
                    )
                else:
                    values = np.empty(len(right), dtype=np.float64)
                    valid = np.zeros(len(right), dtype=np.uint8)
            else:
                raise ValueError("expression operands must have equal length")
            kind = "bool" if self._op in _COMPARISONS else "float"
            return Series._from_arrays(self._output_name, values, valid, kind)
        child = self._args[0]._eval(frame)
        if self._op == "unary":
            values, valid = K.unary(
                child._numeric(),
                child._valid,
                kw["op"],
                float(kw.get("p0", 0.0)),
                float(kw.get("p1", 0.0)),
            )
            kind = "bool" if kw["op"] in ("is_nan", "is_finite", "not") else "float"
            return Series._from_arrays(self._output_name, values, valid, kind)
        if self._op in ("is_null", "is_not_null"):
            values = (child._valid == (0 if self._op == "is_null" else 1)).astype(
                np.float64
            )
            return Series._from_arrays(
                self._output_name, values, np.ones(len(child), dtype=np.uint8), "bool"
            )
        if self._op == "fill_null":
            value = kw["value"]
            result = child.clone()
            missing = result._valid == 0
            result._values[missing] = value
            result._valid[missing] = 1
            return result.rename(self._output_name)
        if self._op == "round":
            if kw["mode"] not in ("half_to_even", "half_away_from_zero"):
                raise ValueError("unsupported rounding mode")
            result = child.clone()
            if kw["mode"] == "half_to_even":
                result._values = np.round(result._values, kw["decimals"])
            else:
                scale = 10.0 ** kw["decimals"]
                values = result._values * scale
                result._values = np.copysign(np.floor(np.abs(values) + 0.5), values) / scale
            return result.rename(self._output_name)
        if self._op == "clip":
            lower = -np.inf if kw["lower"] is None else float(kw["lower"])
            upper = np.inf if kw["upper"] is None else float(kw["upper"])
            values, valid = K.unary(child._numeric(), child._valid, "clip", lower, upper)
            return Series._from_arrays(self._output_name, values, valid, child._kind)
        if self._op == "cast":
            kind = _dtype_kind(kw["dtype"])
            if kind is None:
                raise TypeError(f"unsupported dtype: {kw['dtype']!r}")
            if kind == "str":
                return Series(self._output_name, child.to_list(), dtype=String)
            values = child._numeric().copy()
            if kind == "int":
                values = np.trunc(values)
            elif kind == "bool":
                values = (values != 0).astype(np.float64)
            return Series._from_arrays(self._output_name, values, child._valid.copy(), kind)
        if self._op == "agg":
            return _aggregate(child, kw["op"], kw, self._output_name)
        if self._op == "cumulative":
            values, valid = K.cumulative(
                child._numeric(), child._valid, kw["op"], kw["reverse"]
            )
            return Series._from_arrays(self._output_name, values, valid, child._kind)
        if self._op == "rolling":
            if kw["window_size"] <= 0:
                raise ValueError("window_size must be positive")
            values, valid = K.rolling(
                child._numeric(),
                child._valid,
                kw["op"],
                kw["window_size"],
                kw["min_samples"],
                kw["ddof"],
            )
            return Series._from_arrays(self._output_name, values, valid, "float")
        if self._op == "shift":
            return _shift(child, kw["n"], kw["fill_value"]).rename(self._output_name)
        if self._op == "diff":
            if kw["null_behavior"] != "ignore":
                raise NotImplementedError("only null_behavior='ignore' is covered")
            shifted = _shift(child, kw["n"], None)
            values, valid = K.binary(
                child._numeric(),
                child._valid,
                shifted._numeric(),
                shifted._valid,
                "sub",
            )
            return Series._from_arrays(self._output_name, values, valid, "float")
        if self._op == "pct_change":
            shifted = _shift(child, kw["n"], None)
            delta, valid = K.binary(
                child._numeric(),
                child._valid,
                shifted._numeric(),
                shifted._valid,
                "sub",
            )
            values, valid = K.binary(
                delta, valid, shifted._numeric(), shifted._valid, "truediv"
            )
            return Series._from_arrays(self._output_name, values, valid, "float")
        raise NotImplementedError(f"expression operation {self._op!r}")


def _match_fused_math(expression: Expr) -> tuple[str, str, float] | None:
    if expression._op != "unary" or expression._kw.get("op") != "sqrt":
        return None
    absolute = expression._args[0]
    if absolute._op != "unary" or absolute._kw.get("op") != "abs":
        return None
    addition = absolute._args[0]
    if addition._op != "add":
        return None
    for multiply, other in (
        (addition._args[0], addition._args[1]),
        (addition._args[1], addition._args[0]),
    ):
        if multiply._op != "mul" or other._op != "col":
            continue
        left, right = multiply._args
        if left._op == "col" and right._op == "lit":
            column, literal = left, right
        elif right._op == "col" and left._op == "lit":
            column, literal = right, left
        else:
            continue
        if isinstance(literal._args[0], (int, float, np.number)):
            return (
                str(column._args[0]),
                str(other._args[0]),
                float(literal._args[0]),
            )
    return None


def _match_scalar_comparison(
    expression: Expr,
) -> tuple[str, float, str, bool] | None:
    if expression._op not in {"eq", "ne", "lt", "le", "gt", "ge"}:
        return None
    left, right = expression._args
    if left._op == "col" and right._op == "lit":
        column, literal, scalar_left = left, right, False
    elif right._op == "col" and left._op == "lit":
        column, literal, scalar_left = right, left, True
    else:
        return None
    if not isinstance(literal._args[0], (int, float, np.number)):
        return None
    return (
        str(column._args[0]),
        float(literal._args[0]),
        expression._op,
        scalar_left,
    )


def _match_fused_compare(
    expression: Expr,
) -> tuple[str, str, float, float, str, str, bool, bool] | None:
    if expression._op != "and":
        return None
    left = _match_scalar_comparison(expression._args[0])
    right = _match_scalar_comparison(expression._args[1])
    if left is None or right is None:
        return None
    return (
        left[0],
        right[0],
        left[1],
        right[1],
        left[2],
        right[2],
        left[3],
        right[3],
    )


def _to_expr(value: Any) -> Expr:
    if isinstance(value, Expr):
        return value
    if isinstance(value, Series):
        return value._expr()
    return Expr("lit", (value,))


def col(name: str) -> Expr:
    return Expr("col", (name,), output_name=name)


def lit(value: Any, dtype: DataType | type | str | None = None) -> Expr:
    expression = Expr("lit", (value,))
    return expression if dtype is None else expression.cast(dtype)


def _broadcast(series: Series, size: int) -> Series:
    if len(series) == size:
        return series
    if len(series) != 1:
        raise ValueError("expression result has an incompatible length")
    return Series._from_arrays(
        series.name,
        np.repeat(series._values, size),
        np.repeat(series._valid, size),
        series._kind,
    )


def _broadcast_pair(left: Series, right: Series) -> tuple[Series, Series]:
    size = max(len(left), len(right))
    return _broadcast(left, size), _broadcast(right, size)


def _object_binary(left: Series, right: Series, op: str, name: str) -> Series:
    if op not in ("eq", "ne", "lt", "le", "gt", "ge"):
        raise TypeError("string columns only support comparisons")
    function = {
        "eq": operator.eq,
        "ne": operator.ne,
        "lt": operator.lt,
        "le": operator.le,
        "gt": operator.gt,
        "ge": operator.ge,
    }[op]
    valid = (left._valid & right._valid).astype(np.uint8)
    values = np.zeros(len(left), dtype=np.float64)
    for i in range(len(left)):
        if valid[i]:
            values[i] = float(function(left._values[i], right._values[i]))
    return Series._from_arrays(name, values, valid, "bool")


def _aggregate(child: Series, op: str, kw: dict[str, Any], name: str) -> Series:
    if op == "n_unique":
        valid_count = int(child._valid.sum())
        values = {
            child._python_value(value)
            for value, valid in zip(child._values, child._valid)
            if valid
        }
        result = len(values) + int(valid_count != len(child))
        return Series(name, [result], dtype=Int64)
    if op in ("first", "last"):
        if not len(child):
            return Series(name, [None], dtype=child.dtype)
        index = 0 if op == "first" else len(child) - 1
        value = (
            child._python_value(child._values[index])
            if child._valid[index]
            else None
        )
        return Series(name, [value], dtype=child.dtype)
    if op in ("median", "quantile"):
        values = child._numeric()[child._valid != 0]
        if not values.size:
            return Series(name, [None], dtype=Float64)
        if op == "median":
            result = float(np.median(values))
        else:
            method = {
                "nearest": "nearest",
                "higher": "higher",
                "lower": "lower",
                "midpoint": "midpoint",
                "linear": "linear",
            }[kw["interpolation"]]
            result = float(np.quantile(values, kw["quantile"], method=method))
        return Series(name, [result], dtype=Float64)
    ddof = int(kw.get("ddof", 1))
    result = K.reduce(child._numeric(), child._valid, op, ddof)
    valid_count = (
        int(child._valid.sum())
        if op in ("mean", "min", "max", "var", "std")
        else len(child)
    )
    is_valid = not (
        (op in ("mean", "min", "max") and valid_count == 0)
        or (op in ("var", "std") and valid_count <= ddof)
    )
    if op in ("any", "all") and not kw.get("ignore_nulls", True) and child.null_count:
        decisive = result != 0.0 if op == "any" else result == 0.0
        is_valid = decisive
    kind = "int" if op in ("count", "len") else "bool" if op in ("any", "all") else "float"
    return Series._from_arrays(
        name,
        np.asarray([result], dtype=np.float64),
        np.asarray([is_valid], dtype=np.uint8),
        kind,
    )


def _shift(child: Series, n: int, fill_value) -> Series:
    size = len(child)
    values = np.empty_like(child._values)
    valid = np.empty_like(child._valid)
    if abs(n) >= size:
        valid.fill(0 if fill_value is None else 1)
        values.fill("" if child._kind == "str" and fill_value is None else (fill_value or 0))
    elif n >= 0:
        values[n:] = child._values[: size - n] if n else child._values
        valid[n:] = child._valid[: size - n] if n else child._valid
        if n:
            values[:n] = "" if child._kind == "str" and fill_value is None else (fill_value or 0)
            valid[:n] = 0 if fill_value is None else 1
    else:
        k = -n
        values[: size - k] = child._values[k:]
        valid[: size - k] = child._valid[k:]
        values[size - k :] = "" if child._kind == "str" and fill_value is None else (fill_value or 0)
        valid[size - k :] = 0 if fill_value is None else 1
    return Series._from_arrays(child.name, values, valid, child._kind)


def _flatten(items: tuple[Any, ...]) -> list[Any]:
    if len(items) == 1 and isinstance(items[0], (list, tuple)):
        return list(items[0])
    return list(items)


class DataFrame:
    def __init__(
        self,
        data: Mapping[str, Any] | Sequence[Series] | None = None,
        schema=None,
        *,
        schema_overrides=None,
        strict: bool = True,
        orient=None,
        infer_schema_length=100,
        nan_to_null: bool = False,
    ):
        del schema, schema_overrides, strict, orient, infer_schema_length
        data = {} if data is None else data
        if isinstance(data, Mapping):
            columns = {
                str(name): (
                    values.rename(str(name))
                    if isinstance(values, Series)
                    else Series(str(name), values, nan_to_null=nan_to_null)
                )
                for name, values in data.items()
            }
        else:
            columns = {series.name: series for series in data}
        lengths = {len(series) for series in columns.values()}
        if len(lengths) > 1:
            raise ValueError("columns have different lengths")
        self._columns = columns

    @property
    def columns(self) -> list[str]:
        return list(self._columns)

    @property
    def dtypes(self) -> list[DataType]:
        return [series.dtype for series in self._columns.values()]

    @property
    def schema(self) -> dict[str, DataType]:
        return {name: series.dtype for name, series in self._columns.items()}

    @property
    def height(self) -> int:
        return len(next(iter(self._columns.values()))) if self._columns else 0

    @property
    def width(self) -> int:
        return len(self._columns)

    @property
    def shape(self) -> tuple[int, int]:
        return self.height, self.width

    def __len__(self) -> int:
        return self.height

    def __repr__(self) -> str:
        return f"DataFrame({self.to_dict(as_series=False)!r})"

    def __getitem__(self, item):
        if isinstance(item, str):
            return self._columns[item]
        return DataFrame({name: self._columns[name] for name in item})

    def get_column(self, name: str) -> Series:
        return self._columns[name]

    def select(self, *exprs, **named_exprs) -> DataFrame:
        expressions = [
            col(item) if isinstance(item, str) else _to_expr(item)
            for item in _flatten(exprs)
        ]
        expressions.extend(_to_expr(value).alias(name) for name, value in named_exprs.items())
        results = [expression._eval(self._columns) for expression in expressions]
        size = max((len(result) for result in results), default=0)
        columns: dict[str, Series] = {}
        for result in results:
            result = _broadcast(result, size)
            if result.name in columns:
                raise ValueError(f"duplicate output name: {result.name!r}")
            columns[result.name] = result
        return DataFrame(columns)

    def with_columns(self, *exprs, **named_exprs) -> DataFrame:
        expressions = [
            col(item) if isinstance(item, str) else _to_expr(item)
            for item in _flatten(exprs)
        ]
        expressions.extend(_to_expr(value).alias(name) for name, value in named_exprs.items())
        columns = dict(self._columns)
        for expression in expressions:
            result = _broadcast(expression._eval(self._columns), self.height)
            columns[result.name] = result
        return DataFrame(columns)

    def filter(self, *predicates, **constraints) -> DataFrame:
        expressions = [_to_expr(item) for item in predicates]
        expressions.extend(col(name) == value for name, value in constraints.items())
        if not expressions:
            return DataFrame(self._columns)
        predicate = expressions[0]
        for expression in expressions[1:]:
            predicate = predicate & expression
        mask = _broadcast(predicate._eval(self._columns), self.height)
        indexes = np.flatnonzero(
            (mask._valid != 0) & (mask._values != 0)
        )
        return DataFrame(
            {
                name: Series._from_arrays(
                    name,
                    series._values[indexes],
                    (
                        np.ones(indexes.size, dtype=np.uint8)
                        if series._valid.all()
                        else series._valid[indexes]
                    ),
                    series._kind,
                )
                for name, series in self._columns.items()
            }
        )

    def group_by(self, *by, maintain_order: bool = False, **named_by) -> GroupBy:
        keys = [
            col(item) if isinstance(item, str) else _to_expr(item)
            for item in _flatten(by)
        ]
        keys.extend(_to_expr(value).alias(name) for name, value in named_by.items())
        return GroupBy(self, keys, maintain_order)

    groupby = group_by

    def sort(
        self,
        by,
        *more_by,
        descending: bool | Sequence[bool] = False,
        nulls_last: bool | Sequence[bool] = False,
        multithreaded: bool = True,
        maintain_order: bool = False,
    ) -> DataFrame:
        del multithreaded, maintain_order
        names = [by, *more_by] if not isinstance(by, (list, tuple)) else list(by)
        desc = [descending] * len(names) if isinstance(descending, bool) else list(descending)
        null_last = [nulls_last] * len(names) if isinstance(nulls_last, bool) else list(nulls_last)
        indexes = list(range(self.height))

        def compare(left: int, right: int) -> int:
            for name, reverse, nl in zip(names, desc, null_last):
                series = self._columns[name]
                left_valid = bool(series._valid[left])
                right_valid = bool(series._valid[right])
                if left_valid != right_valid:
                    if not left_valid:
                        return 1 if nl else -1
                    return -1 if nl else 1
                if not left_valid:
                    continue
                left_value = series._values[left]
                right_value = series._values[right]
                if left_value < right_value:
                    return 1 if reverse else -1
                if left_value > right_value:
                    return -1 if reverse else 1
            return 0

        indexes.sort(key=cmp_to_key(compare))
        return DataFrame(
            {
                name: Series._from_arrays(
                    name, series._values[indexes], series._valid[indexes], series._kind
                )
                for name, series in self._columns.items()
            }
        )

    def head(self, n: int = 5) -> DataFrame:
        return DataFrame({name: series[:n] for name, series in self._columns.items()})

    def tail(self, n: int = 5) -> DataFrame:
        return DataFrame({name: series[-n:] for name, series in self._columns.items()})

    def to_dict(self, *, as_series: bool = True) -> dict[str, Any]:
        if as_series:
            return {name: series.clone() for name, series in self._columns.items()}
        return {name: series.to_list() for name, series in self._columns.items()}

    def to_dicts(self) -> list[dict[str, Any]]:
        return [
            {name: series[index] for name, series in self._columns.items()}
            for index in range(self.height)
        ]


class GroupBy:
    def __init__(self, frame: DataFrame, keys: list[Expr], maintain_order: bool):
        self._frame = frame
        self._key_exprs = keys
        self._maintain_order = maintain_order

    def _labels(self) -> tuple[list[Series], np.ndarray, list[tuple[Any, ...]]]:
        key_series = [
            _broadcast(expression._eval(self._frame._columns), self._frame.height)
            for expression in self._key_exprs
        ]
        if len(key_series) == 1 and key_series[0].null_count == 0:
            series = key_series[0]
            if series._kind == "int" and not self._maintain_order and len(series):
                integer_values = (
                    series._native_int
                    if series._native_int is not None
                    else K.cast_int(series._values)
                )
                lower = int(integer_values.min())
                upper = int(integer_values.max())
                if lower >= 0 and upper <= 1_000_000:
                    present = np.bincount(integer_values, minlength=upper + 1) != 0
                    unique = np.flatnonzero(present)
                    if lower == 0 and unique.size == upper + 1:
                        groups = [(int(value),) for value in unique]
                        return key_series, integer_values, groups
                    remap = np.full(upper + 1, -1, dtype=np.int64)
                    remap[unique] = np.arange(unique.size, dtype=np.int64)
                    labels = np.ascontiguousarray(remap[integer_values])
                    groups = [(int(value),) for value in unique]
                    return key_series, labels, groups
            unique, first_indexes, inverse = np.unique(
                series._values, return_index=True, return_inverse=True
            )
            if self._maintain_order:
                order = np.argsort(first_indexes)
                remap = np.empty(order.size, dtype=np.int64)
                remap[order] = np.arange(order.size, dtype=np.int64)
                inverse = remap[inverse]
                unique = unique[order]
            labels = np.ascontiguousarray(inverse, dtype=np.int64)
            groups = [(series._python_value(value),) for value in unique]
            return key_series, labels, groups
        lookup: dict[tuple[Any, ...], int] = {}
        groups: list[tuple[Any, ...]] = []
        labels = np.empty(self._frame.height, dtype=np.int64)
        for row in range(self._frame.height):
            key = tuple(series[row] for series in key_series)
            if key not in lookup:
                lookup[key] = len(groups)
                groups.append(key)
            labels[row] = lookup[key]
        if not self._maintain_order:
            order = sorted(range(len(groups)), key=lambda index: repr(groups[index]))
            remap = np.empty(len(groups), dtype=np.int64)
            for new, old in enumerate(order):
                remap[old] = new
            labels = np.ascontiguousarray(remap[labels])
            groups = [groups[index] for index in order]
        return key_series, labels, groups

    def agg(self, *aggs, **named_aggs) -> DataFrame:
        expressions = [_to_expr(item) for item in _flatten(aggs)]
        expressions.extend(_to_expr(value).alias(name) for name, value in named_aggs.items())
        key_series, labels, groups = self._labels()
        ngroups = len(groups)
        columns: dict[str, Series] = {}
        for key_index, series in enumerate(key_series):
            columns[series.name] = Series(
                series.name,
                [group[key_index] for group in groups],
                dtype=series.dtype,
            )
        for expression in expressions:
            alias = expression._output_name
            base = expression
            if expression._op == "alias":
                base = expression._args[0]
            if base._op != "agg":
                raise TypeError("GroupBy.agg expects aggregation expressions")
            kw = base._kw
            op = kw["op"]
            child = base._args[0]._eval(self._frame._columns)
            if op == "len":
                source = np.ones(self._frame.height, dtype=np.float64)
                valid = np.ones(self._frame.height, dtype=np.uint8)
                values, out_valid = K.group_reduce(
                    source, valid, labels, ngroups, "count", 1
                )
                kind = "int"
            elif op in K.GROUP_OPS:
                values, out_valid = K.group_reduce(
                    child._numeric(),
                    child._valid,
                    labels,
                    ngroups,
                    op,
                    int(kw.get("ddof", 1)),
                )
                kind = "int" if op == "count" else child._kind
            else:
                raise NotImplementedError(f"grouped {op!r} is outside the covered subset")
            columns[alias] = Series._from_arrays(alias, values, out_valid, kind)
        return DataFrame(columns)

    def _all_columns(self, op: str) -> DataFrame:
        key_names = {expression._output_name for expression in self._key_exprs}
        expressions = [
            getattr(col(name), op)()
            for name in self._frame.columns
            if name not in key_names and self._frame[name]._kind != "str"
        ]
        return self.agg(expressions)

    def sum(self) -> DataFrame:
        return self._all_columns("sum")

    def mean(self) -> DataFrame:
        return self._all_columns("mean")

    def min(self) -> DataFrame:
        return self._all_columns("min")

    def max(self) -> DataFrame:
        return self._all_columns("max")

    def count(self) -> DataFrame:
        return self._all_columns("count")

    def len(self, name: str = "len") -> DataFrame:
        return self.agg(Expr("agg", (lit(1),), {"op": "len"}, name))
