"""ctypes loader for the Mojo expression kernels."""

from __future__ import annotations

import ctypes
import os
import shutil
import subprocess

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
LIB = os.environ.get("MOJOPOLARS_LIB") or os.path.join(
    ROOT, "dist", "libmojo-polars.so"
)
SOURCE = os.path.join(ROOT, "src", "kernels.mojo")

I = ctypes.c_int64
F = ctypes.c_double

_SIGNATURES = {
    "mp_binary": ([I] * 8, None),
    "mp_scalar": ([I, I, F, I, I, I, I, I], None),
    "mp_unary": ([I, I, I, I, I, I, F, F], None),
    "mp_fused_math": ([I, I, I, I, F, I, I, I], None),
    "mp_fused_compare": (
        [I, I, I, I, F, F, I, I, I, I, I, I, I],
        None,
    ),
    "mp_cast_int": ([I, I, I], None),
    "mp_reduce": ([I, I, I, I, I, I, I], F),
    "mp_arg_reduce": ([I, I, I, I], I),
    "mp_cumulative": ([I, I, I, I, I, I, I], None),
    "mp_rolling": ([I] * 10, None),
    "mp_filter": ([I] * 7, I),
    "mp_group_reduce": ([I] * 11, None),
}


class BuildError(RuntimeError):
    pass


def build(force: bool = False) -> str:
    if os.environ.get("MOJOPOLARS_LIB") and os.path.exists(LIB):
        return LIB
    if (
        not force
        and os.path.exists(LIB)
        and os.path.getmtime(LIB) >= os.path.getmtime(SOURCE)
    ):
        return LIB
    pixi = shutil.which("pixi")
    if pixi:
        cmd = [pixi, "run", "--manifest-path", os.path.join(ROOT, "pixi.toml"), "build"]
    else:
        cmd = ["bash", os.path.join(ROOT, "build", "build.sh")]
    proc = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, timeout=1800)
    if proc.returncode or not os.path.exists(LIB):
        raise BuildError((proc.stderr or proc.stdout).strip()[:4000])
    return LIB


_LIBRARY: ctypes.CDLL | None = None


def lib() -> ctypes.CDLL:
    global _LIBRARY
    if _LIBRARY is None:
        _LIBRARY = ctypes.CDLL(build())
        for name, (argtypes, restype) in _SIGNATURES.items():
            function = getattr(_LIBRARY, name)
            function.argtypes = argtypes
            function.restype = restype
    return _LIBRARY
