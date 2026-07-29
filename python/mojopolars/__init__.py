"""A Mojo-backed numeric subset of the Polars expression API."""

from .core import (
    Boolean,
    DataFrame,
    DataType,
    Expr,
    Float64,
    Int64,
    Series,
    String,
    Utf8,
    col,
    lit,
)

__version__ = "0.1.0"


def _column_agg(name, operation):
    return getattr(col(name), operation)()


def sum(name):
    return _column_agg(name, "sum")


def mean(name):
    return _column_agg(name, "mean")


def min(name):
    return _column_agg(name, "min")


def max(name):
    return _column_agg(name, "max")


def first(name):
    return _column_agg(name, "first")


def last(name):
    return _column_agg(name, "last")


def len():
    return Expr("agg", (lit(1),), {"op": "len"}, "len")


__all__ = [
    "Boolean",
    "DataFrame",
    "DataType",
    "Expr",
    "Float64",
    "Int64",
    "Series",
    "String",
    "Utf8",
    "col",
    "first",
    "last",
    "len",
    "lit",
    "max",
    "mean",
    "min",
    "sum",
]
