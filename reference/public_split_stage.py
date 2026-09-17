"""Pure CPU stage for padded public split blocks 23..30.

Dependency modules, LUTs, and block files are loaded in the constructor only.
``forward`` operates solely on the caller-provided block22 byte boundary.
"""
from __future__ import annotations

from pathlib import Path
import importlib.util
import sys
import types
from typing import Any

import numpy as np


BLOCK22_LOW_BYTES = 16 * 20 * 512
BLOCK30_LAYER3_BYTES = 16 * 20 * 512
HEAD_8X12_BYTES = 8 * 12 * 1024
N96_REPACK_BYTES = 96 * 1024


def _default_reviews_root() -> Path:
    return Path(__file__).resolve().parent


def _load_module(name: str, path: Path) -> types.ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load review dependency: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


class PaddedPublicSplitStage:
    """Split blocks 23..30 for 16x20 -> logical 8x10 / allocated 8x12.

    ``reviews_root`` defaults to the current local ``reviews`` directory and
    is resolved at construction.  No path is retained as a forward input.
    """

    def __init__(self, reviews_root: str | Path | None = None) -> None:
        self.reviews_root = Path(reviews_root) if reviews_root is not None else _default_reviews_root()
        chain_path = self.reviews_root / "nr-exact-split23-30-h16-main-20260914" / "chain.py"
        if not chain_path.is_file():
            raise FileNotFoundError(f"padded split dependency is missing: {chain_path}")

        # The source chain imports these names unqualified.  Clear stale
        # variants before the one local review is loaded into this stage.
        for name in ("candidate", "candidate_v3", "split_standard_candidate", "block30_candidate"):
            sys.modules.pop(name, None)
        self._split = _load_module("_padded_public_split_chain", chain_path)
        self._candidate = sys.modules["candidate_v3"]
        self._install_patches()
        self._model = self._split.ExactSplit23to30H16()

    def _install_patches(self) -> None:
        split, candidate = self._split, self._candidate

        def geometry(h: int, w: int, x: int, y: int) -> tuple[int, int]:
            if any(type(z) is not int for z in (h, w, x, y)) or not (
                8 <= h <= 32 and 8 <= w <= 32 and x in (-4, 0) and y in (-4, 0)
            ):
                raise ValueError("padded split geometry")
            return (w - x + 7) // 8, (h - y + 7) // 8

        seen: set[int] = set()

        def walk(value: Any, depth: int = 0) -> None:
            if id(value) in seen or depth > 8:
                return
            seen.add(id(value))
            if isinstance(value, types.ModuleType) and hasattr(value, "geometry"):
                value.geometry = geometry
            children = (
                vars(value).values()
                if isinstance(value, types.ModuleType)
                else value.values()
                if isinstance(value, dict)
                else value
                if isinstance(value, (list, tuple))
                else ()
            )
            for child in children:
                if isinstance(child, (types.ModuleType, dict, list, tuple)):
                    walk(child, depth + 1)

        walk(split)

        # Same proved SM120 HFMA2 tree used by the padded encoder pass.
        def snorm(values: np.ndarray, rsqrt: Any) -> dict[str, np.ndarray]:
            rn = candidate.num._rn_half
            square24 = rn(values[..., 24:32].astype(np.float64) ** 2, "sq24")
            square16 = rn(values[..., 16:24].astype(np.float64) ** 2, "sq16")
            pair_a = rn(values[..., 8:16].astype(np.float64) * values[..., 8:16].astype(np.float64) + square24.astype(np.float64), "HFMA8+24")
            pair_b = rn(values[..., :8].astype(np.float64) * values[..., :8].astype(np.float64) + square16.astype(np.float64), "HFMA0+16")
            local = rn(pair_a.astype(np.float64) + pair_b.astype(np.float64), "local").reshape(64, 16, 4, 2)
            butterfly2 = rn(local.astype(np.float64) + local[:, :, np.arange(4) ^ 2, :].astype(np.float64), "b2")
            butterfly1 = rn(butterfly2.astype(np.float64) + butterfly2[:, :, np.arange(4) ^ 1, :].astype(np.float64), "b1")
            denominator = rn(butterfly1[..., 0].astype(np.float64) + butterfly1[..., 1].astype(np.float64), "den")
            epsilon = np.array([948045311], np.uint32).view(np.float32).astype(np.float16)[0]
            floor = np.maximum(denominator, epsilon).astype(np.float16)
            raw, factor = rsqrt.evaluate(floor)
            normalized = rn(values.astype(np.float64) * factor[..., 0, None].astype(np.float64), "factor")
            return {
                "square24": square24, "square16": square16, "pairA": pair_a, "pairB": pair_b,
                "denominator": denominator, "floor": floor, "raw": raw, "normalized": normalized,
            }

        candidate._norm_tree = snorm

    def forward(self, block22_low: bytes) -> dict[str, bytes]:
        """Run split23..30 from a supplied 16x20 block22-low byte tensor."""
        if type(block22_low) is not bytes or len(block22_low) != BLOCK22_LOW_BYTES:
            raise ValueError(f"block22_low must be exactly {BLOCK22_LOW_BYTES} bytes")

        self._install_patches()
        split, model = self._split, self._model
        layer0 = model.b23.layer0(block22_low, 16, 20)
        layer1 = model.b23.layer1(layer0["output"], block22_low, 16, 20)
        # 16x20 is not a full pair of 8x8 tiles: this padded, zero-filled
        # final window is intentional and only scatters its valid 16x20 area.
        layer2 = self._block23_attention_partial(model.b23, layer1["output"], model.rs, model.rc)
        layer3 = model.b23.layer3(layer2["output"], layer1["output"], 16, 20)
        value = layer3["output"]
        for block in range(24, 30):
            value = model.standard[block].forward_shifted(
                value, 16, 20, split.ORIGINS[block], rsqrt=model.rs, reciprocal=model.rc
            )["output"]

        final0 = model.b30.layer0(value, 16, 20)
        final1 = model.b30.layer1(final0["output"], value, 16, 20)
        final2 = model.b30.layer2_shifted(final1["output"], 16, 20, split.ORIGINS[30], rsqrt=model.rs, reciprocal=model.rc)
        final3 = model.b30.layer3(final2["output"], final1["output"], 16, 20)
        pool = self._pool_8x10_in_8x12(model, final3["prequantization"])
        head = model.b30.head(pool, 8, 12)["output"]
        repack = bytes(split.repack.run_repack(head, 8, 12))
        return {"block30_layer3": final3["output"], "head_8x12": head, "n96_repack": repack}

    def _block23_attention_partial(self, model: Any, source: bytes, rsqrt: Any, reciprocal: Any) -> dict[str, Any]:
        split = self._split
        global_codes = split.unpack_a(source, 16, 20)
        result = np.empty((16 * 20, 512), np.uint8)
        seen: set[tuple[int, int]] = set()
        windows: list[dict[str, Any]] = []
        for oy in range(0, 16, 8):
            for ox in range(0, 20, 8):
                raw = np.zeros(8 * 8 * 512, np.uint8)
                for ly in range(8):
                    for lx in range(8):
                        yy, xx = oy + ly, ox + lx
                        if yy < 16 and xx < 20:
                            for channel in range(512):
                                raw[split.packed_a_offset(8, 8, ly, lx, channel)] = global_codes[yy * 20 + xx, channel]
                window = model.layer2(raw.tobytes(), 8, 8, rsqrt=rsqrt, reciprocal=reciprocal)
                windows.append(window)
                for ly in range(8):
                    for lx in range(8):
                        yy, xx = oy + ly, ox + lx
                        if yy < 16 and xx < 20:
                            if (yy, xx) in seen:
                                raise ValueError("duplicate block23 output")
                            seen.add((yy, xx))
                            result[yy * 20 + xx] = window["logical"][8 * ly + lx]
        if len(seen) != 16 * 20:
            raise ValueError("missing block23 output")
        return {"output": split.pack_c(result, 16, 20), "logical": result, "windows": windows, "origin": (0, 0)}

    def _pool_8x10_in_8x12(self, model: Any, prequantization: np.ndarray) -> bytes:
        """Pool logical 8x10 values into a zero-cleared 8x12 FP8 allocation."""
        pool_fn = model.b30.layer3_pool.__func__
        num, pack_c = pool_fn.__globals__["num"], pool_fn.__globals__["pack_c"]
        values = prequantization.reshape(16, 20, 512)
        # Columns 10:12 remain zero before quantization and packing.
        low = np.zeros((8, 12, 512), np.float16)
        rn = num._rn_half
        for y in range(8):
            for x in range(10):
                pair0 = rn(values[2 * y, 2 * x].astype(np.float64) + values[2 * y, 2 * x + 1].astype(np.float64), "pool pair0")
                pair1 = rn(values[2 * y + 1, 2 * x].astype(np.float64) + values[2 * y + 1, 2 * x + 1].astype(np.float64), "pool pair1")
                low[y, x] = rn(rn(pair0.astype(np.float64) + pair1.astype(np.float64), "pool sum").astype(np.float64) * 0.25, "pool quarter")
        return pack_c(num.fp8_rn_satfinite(low.reshape(-1, 512)), 8, 12)
