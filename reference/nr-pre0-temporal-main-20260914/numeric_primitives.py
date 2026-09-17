"""Finite transparent E4M3/F16 array primitives used by the local preblock body.

No source parser, VM, native code, dynamic loader, or external module import is
used.  Operations are the pinned CPU numerical policy: float64 accumulation
followed by explicit RN binary16 boundaries; E4M3FN NaN encodings fail closed.
"""
from __future__ import annotations

from collections.abc import Callable, Mapping
import numpy as np


class UnsupportedNonFinite(ValueError):
    pass


def _e4m3_value(code: int) -> float:
    if code in (0x7F, 0xFF):
        raise UnsupportedNonFinite("E4M3FN NaN operand")
    sign = -1.0 if code & 0x80 else 1.0
    exponent, fraction = (code >> 3) & 15, code & 7
    return sign * ((fraction / 512.0) if exponent == 0
                   else (8 + fraction) * 2.0 ** (exponent - 10))


E4M3_LUT = np.array([_e4m3_value(code) if code not in (0x7F, 0xFF) else np.nan
                      for code in range(256)], dtype=np.float64)
E4_POSITIVE = E4M3_LUT[:0x7F].copy()


def decode_e4m3(packed: bytes, offsets: np.ndarray) -> np.ndarray:
    if type(packed) is not bytes:
        raise ValueError("immutable packed bytes required")
    raw = np.frombuffer(packed, dtype=np.uint8)
    codes = raw[offsets]
    if np.any((codes == 0x7F) | (codes == 0xFF)):
        raise UnsupportedNonFinite("E4M3FN NaN input/weight is not admitted")
    return E4M3_LUT[codes]


def _rn_half(values: np.ndarray, stage: str) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    if not np.all(np.isfinite(values)):
        raise UnsupportedNonFinite(f"nonfinite f64 before {stage}")
    with np.errstate(over="ignore", invalid="ignore"):
        rounded = values.astype(np.float16)
    if not np.all(np.isfinite(rounded)):
        raise UnsupportedNonFinite(f"binary16 overflow/nonfinite at {stage}")
    return rounded


def fp8_rn_satfinite(halves: np.ndarray) -> np.ndarray:
    if halves.dtype != np.float16 or not np.all(np.isfinite(halves)):
        raise UnsupportedNonFinite("finite binary16 activation required for E4M3 pack")
    bits = halves.view(np.uint16)
    sign = ((bits >> 8) & 0x80).astype(np.uint8)
    value = np.abs(halves.astype(np.float64))
    upper = np.searchsorted(E4_POSITIVE, value, side="left")
    result = np.empty(value.shape, dtype=np.uint8)
    result[upper == 0] = 0
    result[upper == len(E4_POSITIVE)] = 0x7E
    middle = (upper > 0) & (upper < len(E4_POSITIVE))
    high = upper[middle]
    low = high - 1
    low_distance = value[middle] - E4_POSITIVE[low]
    high_distance = E4_POSITIVE[high] - value[middle]
    choose_low = ((low_distance < high_distance) |
                  ((low_distance == high_distance) & ((low & 1) == 0)))
    result[middle] = np.where(choose_low, low, high).astype(np.uint8)
    return result | sign


def _readonly_snapshot(array: np.ndarray) -> np.ndarray:
    frozen = np.frombuffer(np.ascontiguousarray(array).tobytes(), dtype=array.dtype)
    return frozen.reshape(array.shape)


def _call_hook(hooks: Mapping[str, Callable[[np.ndarray], None]] | None,
               name: str, array: np.ndarray) -> None:
    if hooks is None:
        return
    if not isinstance(hooks, Mapping):
        raise ValueError("hooks must be a mapping")
    callback = hooks.get(name)
    if callback is not None:
        if not callable(callback):
            raise ValueError(f"hook {name!r} is not callable")
        callback(_readonly_snapshot(array))
