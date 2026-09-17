"""PREBLOCK0 temporal forward built from transparent NumPy arrays, never a VM.

The learned body is the already recovered typed array backbone copied into this
review (`embedding`, `ffn`, `qkv`, `normalized`, `attention`).  This module only
feeds it temporal logical-half features and writes ordinary NumPy output arrays.
There is no PTX parsing, register machine, native call, resource write, or
history recurrence here.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
import struct
from typing import Callable

import numpy as np

import attention
import embedding
import ffn
import normalized
import qkv
from temporal_preblock import (
    ActualFrontendState,
    ApproxPolicy,
    FeatureTile,
    NonFrontendSelectorProof,
    SamplerPolicy,
    ValidatedInvocation,
    preprocess_temporal,
    validate_actual_frontend,
    validate_nonfrontend_selector,
)

HERE = Path(__file__).resolve().parent
SLAB_SHA256 = "5fe2ab86289f7b21253d00368ce7a95f005ec3c8476abd7db22550b918962a5c"
Y, X = np.indices((8, 8))
# PTX tensor row is 4x4-tile order, not source row-major pixel order.
TILE = 16 * ((Y // 4) * 2 + X // 4) + 4 * (Y % 4) + X % 4
PI = ffn.pi(np.arange(32))


@dataclass(frozen=True)
class TemporalPrediction:
    """Pure output arrays plus an explicit non-mutating state transition."""
    skip_C_packed: np.ndarray       # uint8, exact +216 sparse-store payload layout
    primary_A_planar: np.ndarray    # uint8, [2,H/2,W/2,16], exact +248 layout
    feature_tiles: dict[tuple[int, int], FeatureTile]
    counter_before: int
    counter_after: int
    temporal_family: str
    history_write: None = None

    @property
    def history_write_contract(self) -> str:
        return "not modeled: caller owns external copy/direct-history state and synchronization"


def _readonly(array: np.ndarray) -> np.ndarray:
    array.setflags(write=False)
    return array


def _geometry(args264: bytes) -> tuple[int, int]:
    """Bounded normal-output family; no padding/suffix geometry is invented."""
    if type(args264) is not bytes or len(args264) != 264:
        raise ValueError("one exact 264-byte ABI aggregate required")
    output_h, output_w = struct.unpack_from("<2i", args264, 240)
    low_h, low_w = struct.unpack_from("<2i", args264, 256)
    source_h, source_w = struct.unpack_from("<2i", args264, 208)
    if not (8 <= output_h <= 128 and 8 <= output_w <= 128 and
            output_h % 8 == output_w % 8 == 0 and
            (low_h, low_w) == (output_h // 2, output_w // 2)):
        raise ValueError("bounded normal +216/+248 output geometry required")
    # Every output CTA must remain in the original single-upper-mirror input domain.
    if source_h < 2 or source_w < 2 or output_h > 2 * source_h - 1 or output_w > 2 * source_w - 1:
        raise ValueError("output grid would need unsupported input reflection/padding")
    return output_h, output_w


def _reduce_half(final_half: np.ndarray) -> np.ndarray:
    """Original lane-dependent balanced 2x2 half reduction, not a mean shortcut."""
    grid = final_half[TILE]
    output = np.empty((4, 4, 32), dtype=np.float16)
    rn = ffn.num._rn_half
    for y in range(4):
        for x in range(4):
            dy, dx = y % 2, x // 2
            a = grid[2 * y + dy, 2 * x + dx]
            b = grid[2 * y + dy, 2 * x + 1 - dx]
            c = grid[2 * y + 1 - dy, 2 * x + dx]
            d = grid[2 * y + 1 - dy, 2 * x + 1 - dx]
            ab = rn(a.astype(np.float64) + b.astype(np.float64), "temporal downsample pair AB")
            cd = rn(c.astype(np.float64) + d.astype(np.float64), "temporal downsample pair CD")
            total = rn(ab.astype(np.float64) + cd.astype(np.float64), "temporal downsample pair sum")
            output[y, x] = rn(total.astype(np.float64) * 0.25, "temporal downsample quarter")
    return output


def _feature_input(tile: FeatureTile) -> np.ndarray:
    raw = np.asarray(tile.logical, dtype=np.uint16)
    if raw.shape != (8, 8, 16):
        raise AssertionError("temporal feature tile shape")
    # logical is [x,y,C]; learnt embedding expects PTX 4x4-tile row order.
    grid = raw.transpose(1, 0, 2).copy().view(np.float16)
    output = np.empty((64, 16), dtype=np.float16)
    output[TILE.ravel()] = grid.reshape(64, 16)
    if not np.all(np.isfinite(output)):
        raise ValueError("temporal feature conversion produced nonfinite half")
    return output


class Preblock0TemporalArrayReference:
    """Stateless pure-array PREBLOCK0 forward for a validated temporal launch."""

    def __init__(self, slab: bytes):
        if type(slab) is not bytes or len(slab) != 21696:
            raise ValueError("exact 21696-byte block0 slab required")
        if hashlib.sha256(slab).hexdigest() != SLAB_SHA256:
            raise ValueError("block0 slab identity mismatch")
        self._slab = slab

    @classmethod
    def from_local_test_data(cls) -> "Preblock0TemporalArrayReference":
        return cls((HERE / "block0-slab.bin").read_bytes())

    @staticmethod
    def local_numeric_policies() -> tuple[normalized.RsqrtHalfDomain, attention.ReciprocalHalfDomain]:
        """Load pinned CPU primitive tables once; `forward` itself performs no I/O."""
        return (
            normalized.RsqrtHalfDomain((HERE / "rsqrt-f32.bin").read_bytes()),
            attention.ReciprocalHalfDomain((HERE / "reciprocal-f32.bin").read_bytes()),
        )

    def _forward_window(self, args264: bytes, invocation: ValidatedInvocation, *, cta: tuple[int, int],
                        sampler: SamplerPolicy, approx: ApproxPolicy,
                        rsqrt: normalized.RsqrtHalfDomain,
                        reciprocal: attention.ReciprocalHalfDomain) -> tuple[FeatureTile, np.ndarray, np.ndarray]:
        tile = preprocess_temporal(args264, invocation=invocation, sampler=sampler, approx=approx, cta=cta)
        embedded = embedding.forward(_feature_input(tile), self._slab)
        feed_forward = ffn.forward(embedded["output_half"], self._slab)
        raw = qkv.forward_raw(feed_forward["physical"].tobytes(), self._slab)
        norm = normalized.forward(raw["warp_bank_view"], self._slab, rsqrt=rsqrt)
        attended = attention.forward(norm["Q"], norm["K"], norm["V"],
                                     feed_forward["prequantization"], self._slab,
                                     reciprocal=reciprocal)
        high = ffn.num.fp8_rn_satfinite(attended["prequantization"])
        low = _reduce_half(attended["prequantization"])
        low_codes = ffn.num.fp8_rn_satfinite(low)
        low_a = low_codes[:, :, PI]
        packed_low = low_a.reshape(4, 4, 2, 16).transpose(2, 0, 1, 3).copy()
        packed_high = np.empty(2048, dtype=np.uint8)
        packed_high[ffn.C_MAP] = high
        return tile, packed_high, packed_low

    def _forward(self, args264: bytes, invocation: ValidatedInvocation, *, sampler: SamplerPolicy,
                 approx: ApproxPolicy, rsqrt: normalized.RsqrtHalfDomain,
                 reciprocal: attention.ReciprocalHalfDomain) -> TemporalPrediction:
        if not isinstance(sampler, SamplerPolicy) or not isinstance(approx, ApproxPolicy):
            raise ValueError("explicit sampler and nonnative ingress policy required")
        if not isinstance(rsqrt, normalized.RsqrtHalfDomain) or not isinstance(reciprocal, attention.ReciprocalHalfDomain):
            raise ValueError("explicit typed learned numeric primitives required")
        output_h, output_w = _geometry(args264)
        high = np.empty(output_h * output_w * 32, dtype=np.uint8)
        low = np.empty((2, output_h // 2, output_w // 2, 16), dtype=np.uint8)
        high_seen = np.zeros(high.size, dtype=bool)
        low_seen = np.zeros((output_h // 2, output_w // 2), dtype=bool)
        tiles: dict[tuple[int, int], FeatureTile] = {}
        for cy in range(output_h // 8):
            for cx in range(output_w // 8):
                tile, local_high, local_low = self._forward_window(
                    args264, invocation, cta=(cx, cy), sampler=sampler, approx=approx,
                    rsqrt=rsqrt, reciprocal=reciprocal)
                yy, xx = 8 * cy + Y, 8 * cx + X
                global_m = 16 * ((yy // 4) * (output_w // 4) + xx // 4) + 4 * (yy % 4) + xx % 4
                destinations = ffn.axes.c_offset(global_m[:, :, None], np.arange(32)[None, None, :], 32)
                if high_seen[destinations].any():
                    raise AssertionError("overlapping +216 output bytes")
                high[destinations] = local_high[ffn.C_MAP][TILE]
                high_seen[destinations] = True
                ys, xs = slice(4 * cy, 4 * cy + 4), slice(4 * cx, 4 * cx + 4)
                if low_seen[ys, xs].any():
                    raise AssertionError("overlapping +248 output bytes")
                low[:, ys, xs] = local_low
                low_seen[ys, xs] = True
                tiles[cx, cy] = tile
        if not high_seen.all() or not low_seen.all():
            raise AssertionError("unfilled output array; no zero fill is permitted")
        return TemporalPrediction(_readonly(high), _readonly(low), tiles,
                                  invocation.counter_before, invocation.counter_after,
                                  invocation.family)

    def forward_actual(self, args264: bytes, *, state: ActualFrontendState, sampler: SamplerPolicy,
                       approx: ApproxPolicy, rsqrt: normalized.RsqrtHalfDomain,
                       reciprocal: attention.ReciprocalHalfDomain) -> TemporalPrediction:
        """Accepted actual-host temporal family: T1/T2 pair, T3 null, T4 optional."""
        return self._forward(args264, validate_actual_frontend(args264, state), sampler=sampler,
                             approx=approx, rsqrt=rsqrt, reciprocal=reciprocal)

    def forward_nonfrontend_selector(self, args264: bytes, *, proof: NonFrontendSelectorProof,
                                     sampler: SamplerPolicy, approx: ApproxPolicy,
                                     rsqrt: normalized.RsqrtHalfDomain,
                                     reciprocal: attention.ReciprocalHalfDomain) -> TemporalPrediction:
        """Source-math-only nonfrontend T3 test; never labels T3 as DLSSNR.Depth."""
        return self._forward(args264, validate_nonfrontend_selector(args264, proof), sampler=sampler,
                             approx=approx, rsqrt=rsqrt, reciprocal=reciprocal)
