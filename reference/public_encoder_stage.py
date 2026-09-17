"""Pure CPU stage for the captured padded public encoder boundary.

The constructor is the only place that resolves review dependencies or reads
weights/slabs.  ``forward`` consumes supplied byte boundaries only; it does
not consult captures, native artifacts, files, the network, or a process.
"""
from __future__ import annotations

from pathlib import Path
import importlib.util
import sys
import types
from typing import Any

import numpy as np


PRIMARY_BYTES = 256 * 288 * 32
SKIP_BYTES = 512 * 576 * 32
BLOCK22_HIGH_BYTES = 32 * 36 * 256
BLOCK22_LOW_BYTES = 16 * 20 * 512
SKIP4_BYTES = 256 * 288 * 32
SKIP8_BYTES = 128 * 144 * 64
SKIP14_BYTES = 64 * 72 * 128


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


class PaddedPublicEncoderStage:
    """Encoder blocks 1..22 for the public 512x576 padded topology.

    ``reviews_root`` selects the directory containing the local ``nr-*``
    reviews.  ``checkpoint_path`` defaults to the checked-in local review
    weights.  Both are resolved during construction, never by ``forward``.
    """

    def __init__(
        self,
        reviews_root: str | Path | None = None,
        checkpoint_path: str | Path | None = None,
    ) -> None:
        self.reviews_root = Path(reviews_root) if reviews_root is not None else _default_reviews_root()
        chain_path = self.reviews_root / "nr-exact-encoder0-22-h512-main-20260914" / "chain.py"
        if checkpoint_path is None:
            checkpoint_path = (
                self.reviews_root
                / "weights"
                / "dlssnr-310.8-weights.safetensors"
            )
        self.checkpoint_path = Path(checkpoint_path)
        if not chain_path.is_file():
            raise FileNotFoundError(f"padded encoder dependency is missing: {chain_path}")
        if not self.checkpoint_path.is_file():
            raise FileNotFoundError(f"encoder checkpoint is missing: {self.checkpoint_path}")

        # The source family has deliberately short module names; load it only
        # while constructing this stage and retain the constructed pure model.
        self._enc = _load_module("_padded_public_encoder_chain", chain_path)
        self._install_patches()
        self._model = self._enc.ExactEncoder0to22H512(self.checkpoint_path)

    def _install_patches(self) -> None:
        enc = self._enc

        def geometry(h: int, w: int, x: int, y: int) -> tuple[int, int]:
            if any(type(z) is not int for z in (h, w, x, y)) or not (
                8 <= h <= 576 and 8 <= w <= 576 and x in (-4, 0) and y in (-4, 0)
            ):
                raise ValueError("padded public geometry")
            return (w - x + 7) // 8, (h - y + 7) // 8

        def source_shape(h: int, w: int, extent: tuple[int, int]) -> tuple[int, int]:
            eh, ew = extent
            eh = eh if eh > 0 else h
            ew = ew if ew > 0 else w
            if not (1 <= eh <= 576 and 1 <= ew <= 576):
                raise ValueError("padded public extent")
            return eh, ew

        seen: set[int] = set()

        def walk(value: Any, depth: int = 0) -> None:
            if id(value) in seen or depth > 9:
                return
            seen.add(id(value))
            if isinstance(value, types.ModuleType):
                if hasattr(value, "geometry"):
                    value.geometry = geometry
                if hasattr(value, "source_shape"):
                    value.source_shape = source_shape
                if hasattr(value, "shape"):
                    value.shape = source_shape
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

        walk(enc)

        # Preserve the SM120 HFMA2 reduction tree rather than using the
        # source family's generic reduction expression.
        def norm_patch(module: types.ModuleType) -> None:
            def tree(values: np.ndarray, rsqrt: Any) -> dict[str, np.ndarray]:
                rn = module.num._rn_half
                square24 = rn(values[..., 24:32].astype(np.float64) ** 2, "sq24")
                square16 = rn(values[..., 16:24].astype(np.float64) ** 2, "sq16")
                pair_a = rn(
                    values[..., 8:16].astype(np.float64) * values[..., 8:16].astype(np.float64)
                    + square24.astype(np.float64),
                    "HFMA8+24",
                )
                pair_b = rn(
                    values[..., :8].astype(np.float64) * values[..., :8].astype(np.float64)
                    + square16.astype(np.float64),
                    "HFMA0+16",
                )
                local = rn(pair_a.astype(np.float64) + pair_b.astype(np.float64), "local").reshape(
                    values.shape[0], values.shape[1], 4, 2
                )
                butterfly2 = rn(local.astype(np.float64) + local[:, :, np.arange(4) ^ 2, :].astype(np.float64), "b2")
                butterfly1 = rn(butterfly2.astype(np.float64) + butterfly2[:, :, np.arange(4) ^ 1, :].astype(np.float64), "b1")
                denominator = rn(butterfly1[..., 0].astype(np.float64) + butterfly1[..., 1].astype(np.float64), "den")
                epsilon = np.array([948045311], np.uint32).view(np.float32).astype(np.float16)[0]
                floored = np.maximum(denominator, epsilon).astype(np.float16)
                rsqrt_f32_bits, factor_half = rsqrt.evaluate(floored)
                normalized = rn(values.astype(np.float64) * factor_half[..., 0, None].astype(np.float64), "factor")
                return {
                    "square24": square24, "square16": square16, "pair_a": pair_a, "pair_b": pair_b,
                    "local": local, "butterfly2": butterfly2, "butterfly1": butterfly1,
                    "denominator": denominator, "epsilon": epsilon, "floored": floored,
                    "rsqrt_f32_bits": rsqrt_f32_bits, "factor_half": factor_half, "normalized": normalized,
                }

            module.tree = tree

        for module in list(sys.modules.values()):
            if (
                isinstance(module, types.ModuleType)
                and hasattr(module, "tree")
                and hasattr(module, "num")
                and str(getattr(module, "__file__", "")).endswith("/normalized.py")
            ):
                norm_patch(module)
        norm_patch(enc.E.B1["model"].normalized)
        norm_patch(enc.E.B23["model"].normalized)
        for module in (enc.E.D4, enc.E.B5, enc.E.B67, enc.E.D8, enc.E.B9, enc.E.B1013, enc.E.D14, enc.E.B15, enc.E.B1621, enc.E.D22):
            norm_patch(module["normalized"])

        def ingress15(features: bytes, *, h: int, w: int, x: int, y: int, cta: tuple[int, int], source_extent: tuple[int, int] = (0, 0)) -> np.ndarray:
            eh, ew = source_extent
            eh = eh if eh > 0 else h
            ew = ew if ew > 0 else w
            if not (1 <= eh <= 40 and 1 <= ew <= 40) or type(features) is not bytes or len(features) != eh * ew * 256:
                raise ValueError("padded public planar source")
            cx, cy = cta
            source = np.frombuffer(features, np.uint8)
            output = np.zeros(16384, np.uint8)
            for wy in range(8):
                for j in range(16):
                    for lane in range(32):
                        yy = y + 8 * cy + 4 * (j // 8) + 2 * (j % 2) + lane // 16
                        xx = x + 8 * cx + 4 * ((j % 8) // 4) + (lane // 4) % 4
                        valid = (eh == 1 or 0 <= yy < eh) and (ew == 1 or 0 <= xx < ew)
                        sy, sx = (0 if eh == 1 else yy), (0 if ew == 1 else xx)
                        group = 2 * wy + (j % 4) // 2
                        stage = 4096 * (j // 4) + 512 * wy + 16 * lane + 4 * (j % 4)
                        if valid:
                            offset = 16 * ((group * eh + sy) * ew + sx) + 4 * (lane % 4)
                            output[stage : stage + 4] = source[offset : offset + 4]
            return output

        enc.E.B15["nn"].ingress = ingress15

    def forward(self, primary_bytes: bytes, skip_bytes: bytes) -> dict[str, bytes]:
        """Run blocks 1..22 from supplied public-pre0 byte boundaries only."""
        if type(primary_bytes) is not bytes or len(primary_bytes) != PRIMARY_BYTES:
            raise ValueError(f"primary_bytes must be exactly {PRIMARY_BYTES} bytes")
        if type(skip_bytes) is not bytes or len(skip_bytes) != SKIP_BYTES:
            raise ValueError(f"skip_bytes must be exactly {SKIP_BYTES} bytes")

        # Other isolated stage constructors may have rebound short-name module
        # globals; restore this stage's proved geometry immediately before use.
        self._install_patches()
        enc, model = self._enc, self._model
        value = model.b1.forward_packed(primary_bytes, h=256, w=288)["output_packed"]
        for block, origin in ((2, (-4, -4)), (3, (-4, 0))):
            entry = model.b23.blocks[block]
            value = enc.E.B23["model"].nn.forward(value, entry._slab, h=256, w=288, x=origin[0], y=origin[1], rsqrt=entry._rs, reciprocal=entry._rc)["output"]

        result: dict[str, bytes] = {}
        for tag, modules, slabs, rsqrt, reciprocal in model.modules:
            if tag == "d4":
                down = modules["nn"].forward(value, slabs[4], h=256, w=288, rsqrt=rsqrt, reciprocal=reciprocal)
                result["skip4"] = down["high_C"]
                value = down["low_A"]
            elif tag == "b5":
                value = modules["nn"].forward(value, slabs[5], h=128, w=144, rsqrt=rsqrt, reciprocal=reciprocal)["output"]
            elif tag == "b67":
                for block, origin in ((6, (-4, -4)), (7, (-4, 0))):
                    value = modules["nn"].forward(value, slabs[block], h=128, w=144, x=origin[0], y=origin[1], rsqrt=rsqrt, reciprocal=reciprocal)["output"]
            elif tag == "d8":
                down = modules["nn"].forward(value, slabs[8], h=128, w=144, rsqrt=rsqrt, reciprocal=reciprocal)
                result["skip8"] = down["high_C"]
                value = down["low_A"]
            elif tag == "b9":
                value = modules["nn"].forward(value, slabs[9], h=64, w=72, rsqrt=rsqrt, reciprocal=reciprocal)["output"]
            elif tag == "b1013":
                for block, origin in ((10, (-4, -4)), (11, (-4, 0)), (12, (0, -4)), (13, (0, 0))):
                    value = modules["nn"].forward(value, slabs[block], h=64, w=72, x=origin[0], y=origin[1], rsqrt=rsqrt, reciprocal=reciprocal)["output"]
            elif tag == "d14":
                down = modules["nn"].forward(value, slabs[14], h=64, w=72, rsqrt=rsqrt, reciprocal=reciprocal)
                result["skip14"] = down["high_C"]
                value = down["low_A"]
            elif tag == "b15":
                value = modules["nn"].forward(value, slabs[15], h=32, w=36, rsqrt=rsqrt, reciprocal=reciprocal)
            elif tag == "b1621":
                for block, origin in ((16, (-4, -4)), (17, (-4, 0)), (18, (0, -4)), (19, (0, 0)), (20, (-4, -4)), (21, (-4, 0))):
                    value = modules["nn"].forward(value, slabs[block], h=32, w=36, x=origin[0], y=origin[1], rsqrt=rsqrt, reciprocal=reciprocal)["output"]
            elif tag == "d22":
                down = self._block22_padded(modules["nn"], value, slabs[22], rsqrt, reciprocal)
                result["block22_high"] = down["high_C"]
                result["block22_low"] = down["low_A"]
            else:
                raise AssertionError(f"unexpected encoder module: {tag}")
        return result

    @staticmethod
    def _block22_padded(nn: Any, features: bytes, slab: bytes, rsqrt: Any, reciprocal: Any) -> dict[str, bytes]:
        """Block22's 16x20 allocation has only logical columns 0..17."""
        h, w, x, y, low_h, low_w = 32, 36, 0, -4, 16, 20
        core, downsample = nn.core, nn.downsample
        grid_x, grid_y = core.geometry(h, w, x, y)
        if type(features) is not bytes or len(features) != h * w * 256:
            raise ValueError("block22 padded span")
        high = bytearray(h * w * 256)
        high_seen: set[int] = set()
        # Allocation is explicitly zero-cleared; columns 18:20 must remain so.
        low = np.zeros((32, low_h, low_w, 16), np.uint8)
        low_seen: set[tuple[int, int]] = set()
        for cy in range(grid_y):
            for cx in range(grid_x):
                ingress = core.ingress(features, h, w, x, y, cx, cy)
                window = core.forward_window(ingress, slab, rsqrt=rsqrt, reciprocal=reciprocal)
                reduced = downsample.forward(window["attention"]["prequantization"], slab)
                for index, offset in enumerate(core.tile_sources(h, w, x, y, cx, cy)):
                    if offset is not None:
                        if offset in high_seen:
                            raise ValueError("duplicate block22 high output")
                        high_seen.add(offset)
                        high[offset : offset + 4096] = window["physical"][index * 4096 : (index + 1) * 4096].tobytes()
                oy, ox = (y + 8 * cy) // 2, (x + 8 * cx) // 2
                for ly in range(4):
                    for lx in range(4):
                        yy, xx = oy + ly, ox + lx
                        if 0 <= yy < h // 2 and 0 <= xx < w // 2:
                            if (yy, xx) in low_seen:
                                raise ValueError("duplicate block22 low output")
                            low_seen.add((yy, xx))
                            low[:, yy, xx] = reduced["local_planar"][:, ly, lx]
        if high_seen != set(range(0, len(high), 4096)) or len(low_seen) != (h // 2) * (w // 2):
            raise ValueError("missing logical block22 output before padded clear")
        return {"high_C": bytes(high), "low_A": low.tobytes()}
