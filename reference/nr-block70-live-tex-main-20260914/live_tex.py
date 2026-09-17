"""Transparent array/F32-bit reconstruction of block70's optional live TEX tail.

This is the direct post-head RGB/history part of the original block70 PTX:
* path 0: original.ptx lines 13897..14122;
* path 1: original.ptx lines 16025..16257.

The final forward below is ordinary Python/NumPy arrays plus :mod:`f32bits`.
It does *not* import or execute a PTX VM.  TEX itself remains an explicit
strict callback boundary: ``(slot, sampler_mode, u_f32_bits, v_f32_bits) ->
RGBA_f32_bits``.  There is intentionally no nearest/linear image-array helper
here, because such a helper would be an unproved substitute for hardware.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Protocol, runtime_checkable
import struct

import numpy as np

import f32bits as f

# Literal bit patterns in the two PTX paths.
ZERO = 0x00000000
ONE = 0x3F800000
TWO = 0x40000000
HALF = 0x3F000000
NEG_HALF = 0xBF000000
NEG_ONE = 0xBF800000
EIGHT = 0x41000000
A_DECODE_SCALE = 0x3E000000       # +1/8
A_DECODE_BIAS = 0xBD800000        # -1/16
CUBIC_1P5 = 0x3FC00000
CUBIC_2P5 = 0x40200000
SIGMOID_NEG_LOG2E = 0xBFB8AA3B

# Host sampler evidence is recorded here only as an ABI contract for callback
# dispatch.  No software sampler implementation is inferred from these modes.
SAMPLER_MODE_A = 2          # proved Vulkan mode2: normalized/nearest/clamp
SAMPLER_MODE_HISTORY_B = 0  # proved Vulkan mode0: normalized/linear/clamp
SAMPLER_MODE_MOTION_C = 2  # proved Vulkan mode2: normalized/nearest/clamp


class TextureContractError(ValueError):
    """The caller did not provide an exact finite F32-bit fetch contract."""


@runtime_checkable
class StrictTextureFetch(Protocol):
    def fetch(self, slot: str, sampler_mode: int, u_f32_bits: int,
              v_f32_bits: int) -> tuple[int, int, int, int]:
        """Return RGBA F32 bits for the *already computed* exact coordinates."""


@dataclass(frozen=True)
class LiveTextures:
    """Optional A/B/C bindings.

    ``history_b`` is parameter +88 (the five-tap history fetch, mode0) and
    ``motion_c`` is parameter +96 (the motion fetch, mode2), matching the
    original PTX's non-monotonic register loads.  ``None`` represents the
    original null-object gate; it is not synthesized into a texture value.
    """

    a: StrictTextureFetch | None = None
    history_b: StrictTextureFetch | None = None
    motion_c: StrictTextureFetch | None = None

    @staticmethod
    def _checked_fetch(callback: StrictTextureFetch, slot: str, mode: int,
                       u_bits: int, v_bits: int) -> tuple[int, int, int, int]:
        if not isinstance(callback, StrictTextureFetch):
            raise TextureContractError("binding lacks strict .fetch callback")
        # Coordinate bits are passed unchanged.  In particular, this layer does
        # not convert them to pixel indices or apply a sampler approximation.
        answer = callback.fetch(slot, mode, f.u32(u_bits), f.u32(v_bits))
        if type(answer) is not tuple or len(answer) != 4:
            raise TextureContractError("fetch must return an RGBA tuple of four F32 bits")
        out: list[int] = []
        for value in answer:
            if not f.is_finite_word(value):
                raise TextureContractError("fetch result must be finite unsigned F32 bit words")
            out.append(f.u32(value))
        return tuple(out)  # type: ignore[return-value]

    def fetch_a(self, u_bits: int, v_bits: int) -> tuple[int, int, int, int]:
        if self.a is None:
            raise TextureContractError("A absent")
        return self._checked_fetch(self.a, "A", SAMPLER_MODE_A, u_bits, v_bits)

    def fetch_motion_c(self, u_bits: int, v_bits: int) -> tuple[int, int, int, int]:
        if self.motion_c is None:
            raise TextureContractError("C/motion absent")
        return self._checked_fetch(self.motion_c, "C_motion", SAMPLER_MODE_MOTION_C,
                                   u_bits, v_bits)

    def fetch_history_b(self, u_bits: int, v_bits: int) -> tuple[int, int, int, int]:
        if self.history_b is None:
            raise TextureContractError("B/history absent")
        return self._checked_fetch(self.history_b, "B_history", SAMPLER_MODE_HISTORY_B,
                                   u_bits, v_bits)


@dataclass(frozen=True)
class FrameConfig:
    """Named F32-bit projection of block70's relevant 184-byte host ABI.

    The spelling follows the parameter offsets rather than guessing semantic
    meanings for the unknown host transform fields:

    * A: +64/+72/+80;
    * motion C and history transform: +112/+120/+128/+136/+144/+152/+160/+168;
    * source bounds: +172/+176.

    ``blend_half_bits`` represents the loaded 16-bit scalar at non-null +104;
    ``None`` represents a null +104 pointer, whose PTX default is exactly 1.
    """

    source_width: int
    source_height: int
    origin_x: int = 0
    origin_y: int = 0
    rgb_multiplier_bits: int = 0x3D000000  # 1/32
    decode_enabled_u32: int = 1
    blend_half_bits: int | None = None

    # +64/+72/+80: A coordinate sequence
    a_base_x_bits: int = ZERO
    a_base_y_bits: int = ZERO
    a_scale_x_bits: int = ONE
    a_scale_y_bits: int = ONE
    a_post_x_bits: int = ONE
    a_post_y_bits: int = ONE

    # +112..+168: exact register-use names for C-motion / B-history sequence
    p112_low_u32: int = 0
    p112_high_bits: int = ZERO
    p120_low_bits: int = ZERO
    p120_high_bits: int = ONE
    p128_low_bits: int = ONE
    p128_high_bits: int = ONE
    p136_low_bits: int = ONE
    p136_high_bits: int = ZERO
    p144_low_bits: int = ZERO
    p144_high_bits: int = ONE
    p152_low_bits: int = ONE
    p152_high_bits: int = ONE
    p160_low_bits: int = ONE
    p160_high_bits: int = ONE
    p168_bits: int = ONE

    def __post_init__(self) -> None:
        if (type(self.source_width) is not int or type(self.source_height) is not int
                or not 0 < self.source_width <= 0x7FFFFFFF
                or not 0 < self.source_height <= 0x7FFFFFFF):
            raise ValueError("source dimensions must be positive signed-32 bounds")
        if type(self.origin_x) is not int or type(self.origin_y) is not int:
            raise TypeError("origins must be integer coordinates")
        if self.origin_x not in (0, -4) or self.origin_y not in (0, -4):
            raise ValueError("only the pinned host ABI origins 0/-4 are admitted")
        if type(self.decode_enabled_u32) is not int or not 0 <= self.decode_enabled_u32 <= 0xFFFFFFFF:
            raise ValueError("decode flag must be uint32")
        if type(self.p112_low_u32) is not int or not 0 <= self.p112_low_u32 <= 0xFFFFFFFF:
            raise ValueError("+112 low word must be uint32")
        if self.blend_half_bits is not None and (type(self.blend_half_bits) is not int
                                                 or not 0 <= self.blend_half_bits <= 0xFFFF):
            raise ValueError("blend must be None or raw uint16")
        f32_fields = (
            "rgb_multiplier_bits", "p112_high_bits", "p120_low_bits", "p120_high_bits",
            "p128_low_bits", "p128_high_bits", "p136_low_bits", "p136_high_bits",
            "p144_low_bits", "p144_high_bits", "p152_low_bits", "p152_high_bits",
            "p160_low_bits", "p160_high_bits", "p168_bits", "a_base_x_bits",
            "a_base_y_bits", "a_scale_x_bits", "a_scale_y_bits", "a_post_x_bits",
            "a_post_y_bits",
        )
        for name in f32_fields:
            value = getattr(self, name)
            if not f.is_finite_word(value):
                raise ValueError(name + " must be finite unsigned F32 words")

    @classmethod
    def from_ptx_param_bytes(cls, raw: bytes, *, blend_half_bits: int | None) -> "FrameConfig":
        """Decode only the transparent scalar fields from a 184-byte ABI packet.

        Texture object words at +56/+88/+96 are deliberately ignored: hardware
        objects are represented only by :class:`LiveTextures` callbacks.
        """
        if type(raw) is not bytes or len(raw) != 184:
            raise ValueError("exact 184-byte block70 ABI packet required")
        u32_at = lambda off: struct.unpack_from("<I", raw, off)[0]
        i32_at = lambda off: struct.unpack_from("<i", raw, off)[0]
        return cls(
            source_width=u32_at(172), source_height=u32_at(176),
            origin_x=i32_at(40), origin_y=i32_at(44),
            rgb_multiplier_bits=u32_at(48), decode_enabled_u32=u32_at(52),
            blend_half_bits=blend_half_bits,
            a_base_x_bits=u32_at(64), a_base_y_bits=u32_at(68),
            a_scale_x_bits=u32_at(72), a_scale_y_bits=u32_at(76),
            a_post_x_bits=u32_at(80), a_post_y_bits=u32_at(84),
            p112_low_u32=u32_at(112), p112_high_bits=u32_at(116),
            p120_low_bits=u32_at(120), p120_high_bits=u32_at(124),
            p128_low_bits=u32_at(128), p128_high_bits=u32_at(132),
            p136_low_bits=u32_at(136), p136_high_bits=u32_at(140),
            p144_low_bits=u32_at(144), p144_high_bits=u32_at(148),
            p152_low_bits=u32_at(152), p152_high_bits=u32_at(156),
            p160_low_bits=u32_at(160), p160_high_bits=u32_at(164),
            p168_bits=u32_at(168),
        )


@dataclass(frozen=True)
class PixelProbe:
    """Optional per-pixel transparent state; all numerical values are F32 bits."""

    x: int
    y: int
    ptx_path: int
    uv_bits: tuple[int, int]
    a_coordinate_bits: tuple[int, int] | None
    motion_coordinate_bits: tuple[int, int] | None
    history_coordinate_bits: tuple[tuple[int, int], ...]
    blend_bits: int
    history_weight_bits: int | None
    rgba_bits: tuple[int, int, int, int]


Probe = Callable[[PixelProbe], None]


@dataclass(frozen=True)
class _Axis:
    point0: int
    point1: int
    point2: int
    left: int
    center: int
    right: int


def _source_f32_dimensions(config: FrameConfig) -> tuple[int, int]:
    return f.cvt_rn_f32_s32(config.source_width), f.cvt_rn_f32_s32(config.source_height)


def _blend_bits(config: FrameConfig) -> int:
    # PTX: null +104 -> 1; otherwise cvt.f32.f16, abs==inf -> 0, then clamp.
    if config.blend_half_bits is None:
        return ONE
    raw = f.half_to_f32_bits(config.blend_half_bits)
    if f.is_inf(raw):
        return ZERO
    # half NaNs are not a finite transparent primitive domain.  Do not invent
    # max/min-with-NaN behavior for an unmeasured temporal path.
    if f.is_nan(raw):
        raise f.F32DomainError("blend half NaN is outside admitted live-TEX domain")
    return f.clamp01_ftz(raw)


def _a_coordinates(u: int, v: int, c: FrameConfig) -> tuple[int, int]:
    # PTX 13908..13914 / 16038..16044, preserving FMA/multiply boundaries.
    x = f.mul_ftz(c.a_post_x_bits, f.fma_rn_ftz(c.a_scale_x_bits, u, c.a_base_x_bits))
    y = f.mul_ftz(c.a_post_y_bits, f.fma_rn_ftz(c.a_scale_y_bits, v, c.a_base_y_bits))
    return x, y


def _motion_coordinates(u: int, v: int, c: FrameConfig) -> tuple[int, int]:
    # PTX 13977..13984 / 16110..16117.  The C object is parameter +96.
    x = f.mul_ftz(c.p152_high_bits, f.fma_rn_ftz(c.p144_high_bits, u, c.p136_high_bits))
    y = f.mul_ftz(c.p160_low_bits, f.fma_rn_ftz(c.p152_low_bits, v, c.p144_low_bits))
    return x, y


def _axis(position: int, extent: int, reciprocal_extent: int, path: int,
          approx: f.SyntheticRNApprox, path0_clamp_max: int | None = None) -> _Axis:
    """Exact five-tap coordinate/weight sequence in PTX order.

    ``path`` keeps the two source forms explicit.  Path0 uses the precomputed
    +``extent-.5`` registers (r35/r36); path1 recomputes that subtraction in
    lines 16171/16175.  They are not algebraically collapsed here.
    """
    # PTX uses ``+ 0f3F000000`` here: +0.5, not +1.0.
    base = f.add_ftz(f.floor_rmi_ftz(f.add_ftz(position, NEG_HALF)), HALF)
    frac = f.sub_ftz(position, base)
    t = f.min_ftz(f.max_ftz(frac, ZERO), ONE)
    t2 = f.mul_ftz(t, t)
    t3 = f.mul_ftz(t, t2)
    sum_t_t3 = f.add_ftz(t, t3)
    left = f.fma_rn_ftz(sum_t_t3, NEG_HALF, t2)
    cubic_center = f.add_ftz(f.sub_ftz(f.mul_ftz(t3, CUBIC_1P5),
                                       f.mul_ftz(t2, CUBIC_2P5)), ONE)
    right = f.mul_ftz(f.sub_ftz(t3, t2), HALF)
    # r4773/r4777 (or r5803/r5807): retain all subtractions/additions.
    center_numerator = f.sub_ftz(f.sub_ftz(f.sub_ftz(ONE, left), cubic_center), right)
    center_denominator = f.add_ftz(cubic_center, center_numerator)
    if path == 0:
        if path0_clamp_max is None:
            raise ValueError("path0 requires its precomputed r35/r36 clamp maximum")
        clamp_max = path0_clamp_max
    elif path == 1:
        # path1's local r5812/r5816 computation, deliberately not shared.
        clamp_max = f.add_ftz(extent, NEG_HALF)
    else:
        raise ValueError("PTX path must be 0 or 1")
    # PTX r4780 is 0f3F000000 (+0.5): history sample centers are clamped
    # to [0.5, extent-0.5], not to [0, extent-0.5].
    point0 = f.min_ftz(f.max_ftz(f.add_ftz(base, NEG_ONE), HALF), clamp_max)
    point1 = f.min_ftz(f.max_ftz(f.add_ftz(approx.div_approx(center_numerator,
                                                               center_denominator), base), HALF),
                       clamp_max)
    # PTX r4794/r4797 add 0f40000000 (+2.0) to the half-shifted base.
    point2 = f.min_ftz(f.max_ftz(f.add_ftz(base, TWO), HALF), clamp_max)
    # reciprocal_extent is intentionally an argument because PTX prepares r37/r38
    # before the output paths.  It is consumed by caller coordinate conversion.
    if not f.is_finite(reciprocal_extent):
        raise f.F32DomainError("finite reciprocal extent required")
    return _Axis(point0, point1, point2, left, center_denominator, right)


def _history_coordinate_x(pixel: int, reciprocal_width: int, c: FrameConfig) -> int:
    normalized = f.mul_ftz(reciprocal_width, pixel)
    return f.mul_ftz(c.p128_high_bits,
                     f.fma_rn_ftz(c.p120_high_bits, normalized, c.p112_high_bits))


def _history_coordinate_y(pixel: int, reciprocal_height: int, c: FrameConfig) -> int:
    normalized = f.mul_ftz(reciprocal_height, pixel)
    return f.mul_ftz(c.p136_low_bits,
                     f.fma_rn_ftz(c.p128_low_bits, normalized, c.p120_low_bits))


def _history_rgb(textures: LiveTextures, x: _Axis, y: _Axis,
                 reciprocal_width: int, reciprocal_height: int,
                 c: FrameConfig, approx: f.SyntheticRNApprox) -> tuple[tuple[int, int, int],
                                                                          tuple[tuple[int, int], ...]]:
    # PTX 14077..14109 / 16212..16244: exactly five B/history TEX calls.
    # PTX TEX order is cross-first, then the center: (x0,y1), (x1,y0),
    # (x1,y1), (x1,y2), (x2,y1).  Preserve it because callback observations
    # and the corresponding r4843..r4847 weight associations are ordered.
    points = ((x.point0, y.point1), (x.point1, y.point0), (x.point1, y.point1),
              (x.point1, y.point2), (x.point2, y.point1))
    coordinates = tuple((_history_coordinate_x(px, reciprocal_width, c),
                         _history_coordinate_y(py, reciprocal_height, c))
                        for px, py in points)
    samples = tuple(textures.fetch_history_b(u, v) for u, v in coordinates)

    w0 = f.mul_ftz(x.left, y.center)
    w1 = f.mul_ftz(y.left, x.center)
    w2 = f.mul_ftz(x.center, y.center)
    w3 = f.mul_ftz(y.right, x.center)
    w4 = f.mul_ftz(x.right, y.center)
    # r4848..r4851 / r5880..r5883: preserve even the source operand order
    # (relevant for signed-zero boundaries in this finite bit primitive).
    denominator = f.add_ftz(w0, w1)
    denominator = f.add_ftz(w2, denominator)
    denominator = f.add_ftz(w3, denominator)
    denominator = f.add_ftz(w4, denominator)
    reciprocal = approx.rcp_approx(denominator)
    rgb: list[int] = []
    for channel in range(3):
        value = f.mul_ftz(samples[1][channel], w1)
        value = f.fma_rn_ftz(samples[0][channel], w0, value)
        value = f.fma_rn_ftz(samples[2][channel], w2, value)
        value = f.fma_rn_ftz(samples[3][channel], w3, value)
        value = f.fma_rn_ftz(samples[4][channel], w4, value)
        rgb.append(f.mul_ftz(value, reciprocal))
    return (tuple(rgb), coordinates)  # type: ignore[return-value]


def _ptx_path_for_pixel(y: int, origin_y: int) -> int:
    # The first store block (139xx) covers local rows 0..3; the second (160xx)
    # covers local rows 4..7.  Keep this source-derived selection explicit.
    return 0 if ((y - origin_y) % 8) < 4 else 1


def _one_pixel(learned: tuple[int, int, int, int], x: int, y: int, *,
               config: FrameConfig, textures: LiveTextures,
               approx: f.SyntheticRNApprox, blend: int) -> tuple[tuple[int, int, int, int], PixelProbe]:
    source_w, source_h = _source_f32_dimensions(config)
    path = _ptx_path_for_pixel(y, config.origin_y)
    # r35/r36 are prepared before the first 139xx output tail.  The 160xx
    # tail instead computes equivalent local maxima beside its B taps.
    path0_max_w = f.add_ftz(source_w, NEG_HALF) if path == 0 else None
    path0_max_h = f.add_ftz(source_h, NEG_HALF) if path == 0 else None
    # PTX computes u/v before the source bounds predicates (13997..13905 and
    # 16025..16035), so retain that order even when no callback will be reached.
    u = approx.div_approx(f.add_ftz(f.cvt_rn_f32_u32(x), HALF), source_w)
    v = approx.div_approx(f.add_ftz(f.cvt_rn_f32_u32(y), HALF), source_h)
    inside = x < config.source_width and y < config.source_height

    rgb = [f.mul_ftz(config.rgb_multiplier_bits, learned[channel]) for channel in range(3)]
    a_coordinate: tuple[int, int] | None = None
    if textures.a is not None and inside:
        a_coordinate = _a_coordinates(u, v, config)
        a_rgba = textures.fetch_a(*a_coordinate)
        for channel in range(3):
            decoded = f.fma_rn_ftz(a_rgba[channel], A_DECODE_SCALE, A_DECODE_BIAS)
            rgb[channel] = f.fma_rn_ftz(config.rgb_multiplier_bits, learned[channel], decoded)

    alpha = ONE if config.decode_enabled_u32 != 0 else ZERO
    if config.decode_enabled_u32 != 0:
        for channel in range(3):
            rgb[channel] = f.clamp01_ftz(f.fma_rn_ftz(rgb[channel], EIGHT, HALF))

    motion_coordinate: tuple[int, int] | None = None
    history_coordinates: tuple[tuple[int, int], ...] = ()
    history_weight: int | None = None
    # Exact conjunction from 13960..13972 / 16093..16106.
    history_active = (config.decode_enabled_u32 != 0 and blend != ZERO and inside
                      and textures.motion_c is not None and textures.history_b is not None)
    if history_active:
        motion_coordinate = _motion_coordinates(u, v, config)
        motion = textures.fetch_motion_c(*motion_coordinate)
        moved_u = f.fma_rn_ftz(config.p160_high_bits, motion[0], u)
        moved_v = f.fma_rn_ftz(config.p168_bits, motion[1], v)
        if config.p112_low_u32 == 0:
            moved_u, moved_v = u, v
        position_x = f.mul_ftz(moved_u, source_w)
        position_y = f.mul_ftz(moved_v, source_h)
        reciprocal_w = approx.rcp_approx(source_w)
        reciprocal_h = approx.rcp_approx(source_h)
        axis_x = _axis(position_x, source_w, reciprocal_w, path, approx, path0_max_w)
        axis_y = _axis(position_y, source_h, reciprocal_h, path, approx, path0_max_h)
        history_rgb, history_coordinates = _history_rgb(textures, axis_x, axis_y,
                                                          reciprocal_w, reciprocal_h,
                                                          config, approx)
        exponent = f.mul_ftz(learned[3], SIGMOID_NEG_LOG2E)
        sigmoid = approx.rcp_approx(f.add_ftz(approx.ex2_approx(exponent), ONE))
        history_weight = f.clamp01_ftz(f.mul_ftz(sigmoid, blend))
        for channel in range(3):
            rgb[channel] = f.fma_rn_ftz(history_weight,
                                         f.sub_ftz(history_rgb[channel], rgb[channel]),
                                         rgb[channel])

    rgba = (rgb[0], rgb[1], rgb[2], alpha)
    return rgba, PixelProbe(x=x, y=y, ptx_path=path, uv_bits=(u, v),
                            a_coordinate_bits=a_coordinate,
                            motion_coordinate_bits=motion_coordinate,
                            history_coordinate_bits=history_coordinates,
                            blend_bits=blend, history_weight_bits=history_weight,
                            rgba_bits=rgba)


def forward(learned_rgba_f32_bits: np.ndarray, *, config: FrameConfig,
            textures: LiveTextures | None = None,
            approx: f.SyntheticRNApprox | None = None,
            probe: Probe | None = None) -> np.ndarray:
    """Run the final transparent live-TEX postprocess over an H×W×4 F32-bit array.

    This is the production-facing forward.  It contains no PTX text, VM import,
    image array sampler, hardware API, network I/O, or deployment action.
    ``textures=None`` means all optional objects are null and exactly follows
    the null gates, while non-null bindings require callback-supplied samples.
    """
    if not isinstance(learned_rgba_f32_bits, np.ndarray) or learned_rgba_f32_bits.dtype != np.uint32:
        raise TypeError("learned input must be a numpy uint32 HxWx4 F32-bit array")
    if learned_rgba_f32_bits.ndim != 3 or learned_rgba_f32_bits.shape[2] != 4:
        raise ValueError("learned input must have shape HxWx4")
    h, w, _ = learned_rgba_f32_bits.shape
    if not (8 <= h <= 512 and 8 <= w <= 512 and h % 8 == 0 and w % 8 == 0):
        raise ValueError("only pinned host ABI output H/W multiples of 8 in [8,512] are admitted")
    if np.any((learned_rgba_f32_bits & np.uint32(f.EXP)) == np.uint32(f.EXP)):
        raise f.F32DomainError("learned F32 array must be finite")
    if textures is None:
        textures = LiveTextures()
    if not isinstance(textures, LiveTextures):
        raise TypeError("textures must be LiveTextures or None")
    if approx is None:
        raise ValueError("explicit APPROX policy required; no native approximation is assumed")
    if not isinstance(approx, f.SyntheticRNApprox):
        raise TypeError("probe currently admits the explicit SyntheticRNApprox policy only")

    blend = _blend_bits(config)
    out = np.empty_like(learned_rgba_f32_bits)
    for yy in range(h):
        for xx in range(w):
            learned = tuple(int(v) for v in learned_rgba_f32_bits[yy, xx])
            rgba, event = _one_pixel(learned, xx, yy, config=config, textures=textures,
                                      approx=approx, blend=blend)
            out[yy, xx] = rgba
            if probe is not None:
                probe(event)
    return out


def half_array_to_f32_bits(learned_rgba_f16: np.ndarray) -> np.ndarray:
    """Transparent exact widening helper for the existing block70 head array."""
    if not isinstance(learned_rgba_f16, np.ndarray) or learned_rgba_f16.dtype != np.float16:
        raise TypeError("learned half input must be numpy float16")
    if learned_rgba_f16.ndim != 3 or learned_rgba_f16.shape[2] != 4:
        raise ValueError("learned half input must have shape HxWx4")
    if not np.all(np.isfinite(learned_rgba_f16)):
        raise f.F32DomainError("finite learned half array required")
    # IEEE binary16->binary32 widening is exact; the view retains raw bit output.
    return learned_rgba_f16.astype(np.float32).view(np.uint32)


def forward_half(learned_rgba_f16: np.ndarray, *, config: FrameConfig,
                 textures: LiveTextures | None = None,
                 approx: f.SyntheticRNApprox | None = None,
                 probe: Probe | None = None) -> np.ndarray:
    return forward(half_array_to_f32_bits(learned_rgba_f16), config=config,
                   textures=textures, approx=approx, probe=probe)


def synthetic_identity_config(width: int, height: int, *, blend_half_bits: int = 0x3A00) -> FrameConfig:
    """A fully named deterministic probe configuration, not a host-default guess."""
    if type(width) is not int or type(height) is not int:
        raise TypeError("dimensions must be int")
    if not (8 <= width <= 512 and 8 <= height <= 512 and width % 8 == height % 8 == 0):
        raise ValueError("only pinned host ABI H/W multiples of 8 in [8,512] are admitted")
    return FrameConfig(source_width=width, source_height=height,
                       rgb_multiplier_bits=0x3D000000, decode_enabled_u32=1,
                       blend_half_bits=blend_half_bits,
                       # A samples normalized output coordinate directly.
                       a_base_x_bits=ZERO, a_base_y_bits=ZERO,
                       a_scale_x_bits=ONE, a_scale_y_bits=ONE,
                       a_post_x_bits=ONE, a_post_y_bits=ONE,
                       # C/motion samples u,v and applies a small explicit delta.
                       p112_low_u32=1, p112_high_bits=ZERO,
                       p120_low_bits=ZERO, p120_high_bits=ONE,
                       p128_low_bits=ONE, p128_high_bits=ONE,
                       p136_low_bits=ONE, p136_high_bits=ZERO,
                       p144_low_bits=ZERO, p144_high_bits=ONE,
                       p152_low_bits=ONE, p152_high_bits=ONE,
                       p160_low_bits=ONE, p160_high_bits=0x3E000000,  # 1/8
                       p168_bits=0x3E000000)  # 1/8
