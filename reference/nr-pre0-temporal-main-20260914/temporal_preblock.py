"""Transparent CPU recovery of PREBLOCK0 temporal ingress.

This is deliberately a feature/array implementation, not a PTX VM.  It never
imports Replay, Thread, a PTX parser, ctypes, or a native runtime.  Every image
read is an explicit raw-F32 sampler callback and every PTX approximate operation
is an explicit non-native callback.  The only admitted real host family is the
recovered frontend route: nonzero full64 counter, valid T1(previous output)+
T2(MVec) pair, null T3, and optional T4(ControlMask).  A separately labelled
non-frontend selector proof exists solely to exercise the PTX T3 math without
misnaming DLSSNR.Depth as T3.

The core operates on integers/Fractions/raw IEEE bit patterns and produces
logical[x][y][16] uint16 half words.  It does not create a history image, clear
one, mutate a counter, infer sampler behavior, or fabricate a missing resource.
"""
from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
import struct
from typing import Callable

U32 = (1 << 32) - 1
U64 = (1 << 64) - 1
FZERO = 0x00000000
FHALF = 0x3F000000
FNEG_HALF = 0xBF000000
FONE = 0x3F800000
FNEGONE = 0xBF800000
HZERO = 0x0000
HHALF = 0x3800
HONE = 0x3C00
HNEGONE = 0xBC00


class UnsupportedDomain(ValueError):
    """The source operation/resource meaning is not admitted by this recovery."""


def uint(value: int, width: int) -> int:
    if type(value) is not int or not 0 <= value < (1 << width):
        raise ValueError(f"required raw uint{width} bits, not float/bool")
    return value


def u32(value: int) -> int:
    if type(value) is not int:
        raise TypeError("integer required")
    return value & U32


def _decode(bits: int, exponent_bits: int, fraction_bits: int) -> Fraction:
    uint(bits, 1 + exponent_bits + fraction_bits)
    exponent = (bits >> fraction_bits) & ((1 << exponent_bits) - 1)
    fraction = bits & ((1 << fraction_bits) - 1)
    if exponent == (1 << exponent_bits) - 1:
        raise UnsupportedDomain("consumed NaN/Inf has no admitted native semantics")
    bias = (1 << (exponent_bits - 1)) - 1
    mantissa = fraction if exponent == 0 else (1 << fraction_bits) + fraction
    power = (1 - bias if exponent == 0 else exponent - bias) - fraction_bits
    value = Fraction(mantissa << power) if power >= 0 else Fraction(mantissa, 1 << -power)
    return -value if bits >> (exponent_bits + fraction_bits) else value


def fvalue(bits: int) -> Fraction:
    return _decode(bits, 8, 23)


def hvalue(bits: int) -> Fraction:
    return _decode(bits, 5, 10)


def ftz(bits: int) -> int:
    """PTX .ftz applied to a raw F32 operand/result, preserving zero sign."""
    uint(bits, 32)
    return bits if bits & 0x7F800000 else bits & 0x80000000


def _round_ratio(numerator: int, denominator: int) -> int:
    quotient, remainder = divmod(numerator, denominator)
    return quotient + int(2 * remainder > denominator or
                          (2 * remainder == denominator and quotient & 1))


def _encode(value: Fraction, exponent_bits: int, fraction_bits: int, zero_sign: int = 0) -> int:
    """Exact rational -> finite IEEE binary RN-even with gradual subnormals."""
    value = Fraction(value)
    if value == 0:
        return (zero_sign & 1) << (exponent_bits + fraction_bits)
    sign = int(value < 0)
    numerator, denominator = abs(value.numerator), value.denominator
    exponent = numerator.bit_length() - denominator.bit_length()
    if (numerator < (denominator << exponent)) if exponent >= 0 else (
            (numerator << -exponent) < denominator):
        exponent -= 1
    bias = (1 << (exponent_bits - 1)) - 1
    quantum = max(exponent, 1 - bias) - fraction_bits
    mantissa = (_round_ratio(numerator, denominator << quantum) if quantum >= 0
                else _round_ratio(numerator << -quantum, denominator))
    if mantissa >= (1 << (fraction_bits + 1)):
        mantissa >>= 1
        quantum += 1
    sign_bits = sign << (exponent_bits + fraction_bits)
    if mantissa < (1 << fraction_bits):
        return sign_bits | mantissa
    exponent_field = quantum + fraction_bits + bias
    if exponent_field >= (1 << exponent_bits) - 1:
        raise UnsupportedDomain("finite arithmetic/conversion overflow")
    return sign_bits | (exponent_field << fraction_bits) | (mantissa - (1 << fraction_bits))


def f32_rn(value: Fraction | int, zero_sign: int = 0) -> int:
    """F32 conversion without .ftz (used by integer conversions and floor)."""
    return _encode(Fraction(value), 8, 23, zero_sign)


def h32(bits: int) -> int:
    """cvt.rn.f16.f32: no F16 FTZ before or after conversion."""
    return _encode(fvalue(bits), 5, 10, bits >> 31)


def _binary(a: int, b: int, exponent_bits: int, fraction_bits: int, *, multiply: bool) -> int:
    x, y = _decode(a, exponent_bits, fraction_bits), _decode(b, exponent_bits, fraction_bits)
    sign_a, sign_b = a >> (exponent_bits + fraction_bits), b >> (exponent_bits + fraction_bits)
    if multiply:
        return _encode(x * y, exponent_bits, fraction_bits, sign_a ^ sign_b)
    # Exact cancellation is +0 except -0 + -0.
    zero_sign = int(x == 0 and y == 0 and sign_a == sign_b == 1)
    return _encode(x + y, exponent_bits, fraction_bits, zero_sign)


def fadd(a: int, b: int) -> int:
    return ftz(_binary(ftz(a), ftz(b), 8, 23, multiply=False))


def fsub(a: int, b: int) -> int:
    return fadd(a, uint(b, 32) ^ 0x80000000)


def fneg(a: int) -> int:
    """neg.ftz.f32 is a sign-bit flip after F32 FTZ, including signed zero."""
    return ftz(uint(a, 32)) ^ 0x80000000


def fmul(a: int, b: int) -> int:
    return ftz(_binary(ftz(a), ftz(b), 8, 23, multiply=True))


def ffma(a: int, b: int, c: int) -> int:
    """fma.rn.ftz.f32: one exact product-plus-add rounding boundary."""
    a, b, c = ftz(a), ftz(b), ftz(c)
    x, y, z = fvalue(a), fvalue(b), fvalue(c)
    zero_sign = int(x * y == 0 and z == 0 and ((a ^ b) >> 31) == (c >> 31) == 1)
    return ftz(f32_rn(x * y + z, zero_sign))


def hsub(a: int, b: int) -> int:
    return _binary(a, uint(b, 16) ^ 0x8000, 5, 10, multiply=False)


def hmul(a: int, b: int) -> int:
    return _binary(a, b, 5, 10, multiply=True)


def fmin(a: int, b: int) -> int:
    """Finite max/min .ftz semantics including signed-zero selection."""
    a, b = ftz(a), ftz(b)
    x, y = fvalue(a), fvalue(b)
    if x == 0 and y == 0:
        return 0x80000000 if (a | b) & 0x80000000 else 0
    return a if x < y else b


def fmax(a: int, b: int) -> int:
    a, b = ftz(a), ftz(b)
    x, y = fvalue(a), fvalue(b)
    if x == 0 and y == 0:
        return 0x80000000 if (a & b) & 0x80000000 else 0
    return a if x > y else b


def ffloor(a: int) -> int:
    """cvt.rmi.ftz.f32.f32 in the finite bounded coordinate domain."""
    a = ftz(a)
    value = fvalue(a)
    if value == 0:
        return a
    return ftz(f32_rn(value.numerator // value.denominator))


APPROX_ARITY = {
    "div.approx.ftz.f32": 2,
    "rcp.approx.ftz.f32": 1,
    "lg2.approx.ftz.f32": 1,
    "sqrt.approx.ftz.f32": 1,
    "sin.approx.ftz.f32": 1,
    "cos.approx.ftz.f32": 1,
}


@dataclass(frozen=True)
class ApproxPolicy:
    """An explicit non-native evaluator for PTX approximate operations."""
    name: str
    nonnative: bool
    evaluate: Callable[[str, tuple[int, ...]], int]

    def __post_init__(self) -> None:
        if not self.name or self.nonnative is not True or not callable(self.evaluate):
            raise ValueError("named explicit NONNATIVE approximation policy required")

    def call(self, op: str, *operands: int) -> int:
        if op not in APPROX_ARITY or len(operands) != APPROX_ARITY[op]:
            raise ValueError("unsupported approximate operation or arity")
        raw = tuple(ftz(uint(value, 32)) for value in operands)
        for value in raw:
            fvalue(value)
        result = ftz(uint(self.evaluate(op, raw), 32))
        fvalue(result)
        return result


@dataclass(frozen=True)
class SamplerPolicy:
    """Opaque raw sampler boundary; no array, filter, or fallback is invented.

    sample(handle_u64, x_f32_bits, y_f32_bits) -> four raw F32 uint32 words.
    The callback must be deterministic over the caller's read-only snapshot.
    """
    name: str
    native_bit_oracle: bool
    sample: Callable[[int, int, int], tuple[int, int, int, int]]

    def __post_init__(self) -> None:
        if not self.name or type(self.native_bit_oracle) is not bool or not callable(self.sample):
            raise ValueError("named raw-F32 sampler callback required")


@dataclass(frozen=True)
class Transform:
    offset: tuple[int, int]
    extent: tuple[int, int]
    post_scale: tuple[int, int]

    @classmethod
    def from_abi(cls, args264: bytes, offset: int) -> "Transform":
        words = struct.unpack_from("<6I", args264, offset)
        return cls(tuple(words[:2]), tuple(words[2:4]), tuple(words[4:]))

    def apply(self, u: int, v: int) -> tuple[int, int]:
        return tuple(
            fmul(ffma(coordinate, self.extent[index], self.offset[index]), self.post_scale[index])
            for index, coordinate in enumerate((u, v))
        )  # type: ignore[return-value]


@dataclass(frozen=True)
class SampleCall:
    role: str
    handle: int
    x_bits: int
    y_bits: int
    rgba_bits: tuple[int, int, int, int]


@dataclass(frozen=True)
class FeatureTile:
    """One original 8x8 shared fill, represented as transparent logical halves."""
    logical: list[list[list[int]]]  # [x][y][feature]
    source_coordinate_bits: list[list[tuple[int, int]]]
    guide_coordinate_bits: list[list[tuple[int, int] | None]]
    sample_calls: tuple[SampleCall, ...]
    sampler_policy: str
    approx_policy: str
    counter_before: int
    temporal_pair_used: bool
    selector_t3_used: bool

    def shared_bytes(self) -> bytes:
        """Pack exactly the source's 2048-byte shared fill, no unwritten holes."""
        output = bytearray(2048)
        for x in range(8):
            for y in range(8):
                pixel = 8 * y + x
                for channel, half in enumerate(self.logical[x][y]):
                    struct.pack_into("<H", output,
                                     1024 * (channel // 8) + 16 * pixel + 2 * (channel % 8),
                                     uint(half, 16))
        return bytes(output)


@dataclass(frozen=True)
class ValidatedInvocation:
    """Validated binding only; no resource contents or counter mutation live here."""
    handles: tuple[int, int, int, int, int]
    counter_before: int
    counter_after: int
    family: str
    t3_role: str  # "frontend_null" or "selector_x_proven"


@dataclass(frozen=True)
class ActualFrontendState:
    """Inputs to the recovered host/frontend gate, before the 264-byte launch.

    `depth_view` is intentionally not routed into T3: recovered frontend code
    has a real Depth getter but writes R9D=0 into N+0x2a8 for this call path.
    A nonzero depth_view therefore remains caller state, not a substitute for a
    selector resource.
    """
    counter_before: int
    source_view: int
    previous_output_view: int
    motion_view: int
    control_mask_view: int
    depth_view: int
    block0_no_edge_route: bool
    model_history_enabled: bool
    effective_reset: bool
    global_history_enabled: bool
    history_view_created: bool
    motion_view_created: bool

    def __post_init__(self) -> None:
        for value in (self.counter_before,):
            uint(value, 64)
        for value in (self.source_view, self.previous_output_view, self.motion_view,
                      self.control_mask_view, self.depth_view):
            uint(value, 64)
        for value in (self.block0_no_edge_route, self.model_history_enabled,
                      self.effective_reset, self.global_history_enabled,
                      self.history_view_created, self.motion_view_created):
            if type(value) is not bool:
                raise ValueError("host gates must be explicit bool values")
        if self.source_view == 0:
            raise ValueError("actual frontend T0 selected source view is mandatory")

    @property
    def pair_can_bind(self) -> bool:
        return (self.block0_no_edge_route and self.model_history_enabled and
                not self.effective_reset and self.global_history_enabled and
                self.history_view_created and self.motion_view_created and
                self.previous_output_view != 0 and self.motion_view != 0)


def _args_handles_counter(args264: bytes) -> tuple[tuple[int, int, int, int, int], int]:
    if type(args264) is not bytes or len(args264) != 264:
        raise ValueError("one exact 264-byte ABI aggregate required")
    return struct.unpack_from("<5Q", args264, 0), struct.unpack_from("<Q", args264, 200)[0]


def validate_actual_frontend(args264: bytes, state: ActualFrontendState) -> ValidatedInvocation:
    """Check, never repair, the host's actual nonzero-counter launch family.

    The host copies old counter to +200 then increments its state before submit.
    Reset suppresses the T1/T2 pair but is not evidence that the counter resets.
    """
    handles, counter = _args_handles_counter(args264)
    if counter != state.counter_before or counter == 0:
        raise UnsupportedDomain("temporal API requires the matching nonzero full64 counter")
    if not state.block0_no_edge_route:
        raise UnsupportedDomain("block0 ordered gather route is not explicitly established")
    if state.pair_can_bind:
        expected_pair = ((state.previous_output_view, state.motion_view)
                         if counter != 0 else (0, 0))
    else:
        expected_pair = (0, 0)
    expected = (state.source_view, *expected_pair, 0, state.control_mask_view)
    if handles != expected:
        raise UnsupportedDomain(
            "ABI handles disagree with recovered frontend/counter/reset gate; "
            "no supplied resource is cleared or substituted")
    return ValidatedInvocation(handles, counter, (counter + 1) & U64,
                               "actual_frontend_history_pair" if expected_pair[0] else
                               "actual_frontend_reset_or_invalid_pair",
                               "frontend_null")


@dataclass(frozen=True)
class NonFrontendSelectorProof:
    """Explicit non-frontend T3 semantic declaration for source-math testing.

    This is not a claim that the recovered frontend binds DLSSNR.Depth to T3.
    It is deliberately impossible to reach from `ActualFrontendState`.
    """
    counter_before: int
    source_view: int
    previous_output_view: int
    motion_view: int
    selector_view: int
    control_mask_view: int
    evidence_id: str
    selector_semantic: str = "selector_x_only"

    def __post_init__(self) -> None:
        uint(self.counter_before, 64)
        for value in (self.source_view, self.previous_output_view, self.motion_view,
                      self.selector_view, self.control_mask_view):
            uint(value, 64)
            if value == 0:
                raise ValueError("proved non-frontend T0..T4 exercise requires nonnull views")
        if not self.evidence_id or self.selector_semantic != "selector_x_only":
            raise UnsupportedDomain("T3 semantic proof must explicitly identify selector X only")


def validate_nonfrontend_selector(args264: bytes, proof: NonFrontendSelectorProof) -> ValidatedInvocation:
    """Fail closed unless a caller supplied a distinct, explicit T3 proof."""
    handles, counter = _args_handles_counter(args264)
    expected = (proof.source_view, proof.previous_output_view, proof.motion_view,
                proof.selector_view, proof.control_mask_view)
    if counter != proof.counter_before or counter == 0 or handles != expected:
        raise UnsupportedDomain("non-frontend selector ABI/proof mismatch; no unknown T3 is zeroed")
    return ValidatedInvocation(handles, counter, (counter + 1) & U64,
                               "explicit_nonfrontend_selector_test", "selector_x_proven")


def mirror_once(coordinate: int, extent: int) -> int:
    if type(coordinate) is not int or type(extent) is not int or coordinate < 0 or extent < 2:
        raise UnsupportedDomain("nonnegative coordinate and extent>=2 required")
    mirrored = coordinate if coordinate < extent else 2 * extent - coordinate - 2
    if mirrored < 0 or mirrored >= extent:
        raise UnsupportedDomain("more than one upper mirror would be needed")
    return mirrored


def _normalize_rgb(raw_rgba: tuple[int, int, int, int], doubled_gain_half: int) -> tuple[int, int, int]:
    # Raw alpha is deliberately not consumed by the source RGB feature path.
    return tuple(hmul(hsub(h32(component), HHALF), doubled_gain_half)
                 for component in raw_rgba[:3])  # type: ignore[return-value]


def noise_integer_words(x: int, y: int, seed: int) -> tuple[int, tuple[int, int, int, int]]:
    """Source mul.lo/mad.lo integer hash; all four rounds share the same state."""
    x, y, seed = u32(x), u32(y), u32(seed)
    z = u32(u32(x * -1918454973) ^ u32(y * -669632447) ^
            u32(seed * -1640531527) ^ 608135816)
    mixed = u32(((z >> ((z >> 28) + 4)) ^ z) * 277803737)
    state = u32(mixed ^ (mixed >> 22))
    words = []
    for multiplier, addend in ((747796405, -1403630843), (-93469191, 1192405134),
                                (-895109107, 568162667), (-2094846927, 878960812)):
        value = u32(state * multiplier + addend)
        mixed_value = u32(((value >> ((value >> 28) + 4)) ^ value) * 277803737)
        words.append(u32(((mixed_value >> 30) ^ (mixed_value >> 8)) + 1))
    return state, tuple(words)  # type: ignore[return-value]


def noise_half(x: int, y: int, seed: int, approx: ApproxPolicy) -> tuple[int, int, int]:
    _, words = noise_integer_words(x, y, seed)
    uniforms = tuple(fmul(f32_rn(word), 0x33800000) for word in words)
    radii = []
    for index in (0, 2):
        logarithm = approx.call("lg2.approx.ftz.f32", uniforms[index])
        radii.append(approx.call("sqrt.approx.ftz.f32",
                                 fmul(fmul(logarithm, 0x3F317218), 0xC0000000)))
    angle0 = fmul(uniforms[1], 0x40C90FDB)
    angle1 = fmul(uniforms[3], 0x40C90FDB)
    return (h32(fmul(radii[0], approx.call("cos.approx.ftz.f32", angle0))),
            h32(fmul(radii[0], approx.call("sin.approx.ftz.f32", angle0))),
            h32(fmul(radii[1], approx.call("cos.approx.ftz.f32", angle1))))


def _control_half(tone: int, structure: int, skin_override: int, structure_override: int,
                  mode: int, mask_rgba: tuple[int, int, int, int] | None) -> tuple[int, int, int, int]:
    uint(mode, 32)
    for value in (tone, structure, skin_override, structure_override):
        fvalue(value)
    if mode != 0 and mask_rgba is None:
        skin, structural = fvalue(ftz(skin_override)), fvalue(ftz(structure_override))
        enabled = max(skin, structural) >= 0
        return (
            h32(tone),
            h32(FONE if enabled else structure),
            h32((structure if skin < 0 else skin_override) if enabled else FNEGONE),
            h32((structure if structural < 0 else structure_override) if enabled else FNEGONE),
        )
    if mask_rgba is not None:
        # T4 X/W are loaded by tex but unconsumed; only Y/Z participate.
        tone, structure = fmul(mask_rgba[1], tone), fmul(mask_rgba[2], structure)
    sentinel = HNEGONE if mode != 0 else HZERO
    return h32(tone), h32(structure), sentinel, sentinel


@dataclass(frozen=True)
class _CubicAxis:
    left: int
    middle: int
    right: int
    w0: int
    wmid: int
    w3: int


def _clamp_source_coordinate(value: int, extent_f32: int) -> int:
    return fmin(fmax(value, FHALF), fadd(extent_f32, FNEG_HALF))


def _cubic_axis(coordinate: int, extent_f32: int, approx: ApproxPolicy) -> _CubicAxis:
    """Exact PTX source ordering for one history-reconstruction axis."""
    base = fadd(ffloor(fadd(coordinate, FNEG_HALF)), FHALF)
    fraction = fmin(fmax(fsub(coordinate, base), FZERO), FONE)
    squared = fmul(fraction, fraction)
    cubed = fmul(fraction, squared)
    w0 = ffma(fadd(fraction, cubed), FNEG_HALF, squared)
    w1 = fadd(fsub(fmul(cubed, 0x3FC00000), fmul(squared, 0x40200000)), FONE)
    w3 = fmul(fsub(cubed, squared), FHALF)
    w2 = fsub(fsub(fsub(FONE, w0), w1), w3)
    middle_weight = fadd(w1, w2)
    if fvalue(middle_weight) == 0:
        raise UnsupportedDomain("history cubic middle denominator is zero")
    left = _clamp_source_coordinate(fadd(base, FNEGONE), extent_f32)
    middle = _clamp_source_coordinate(
        fadd(approx.call("div.approx.ftz.f32", w2, middle_weight), base), extent_f32)
    right = _clamp_source_coordinate(fadd(base, 0x40000000), extent_f32)
    return _CubicAxis(left, middle, right, w0, middle_weight, w3)


def _selector_offset(u: int, v: int, *, t3_handle: int, t3: Transform, t2: Transform,
                     depth_inverted: int, sample: Callable[[str, int, tuple[int, int]], tuple[int, int, int, int]],
                     approx: ApproxPolicy) -> tuple[int, int]:
    """PTX T3 centre/TL/TR/BL/BR strict selector, with no Depth aliasing.

    +168==0 chooses strict minimum X; nonzero chooses strict maximum X.  Equal
    and nonfinite selector X values retain the previous candidate.  Nonfinite X
    is outside the admitted finite consumed domain and therefore fails closed.
    """
    uint(depth_inverted, 32)
    reciprocal_x = approx.call("rcp.approx.ftz.f32", t3.extent[0])
    reciprocal_y = approx.call("rcp.approx.ftz.f32", t3.extent[1])
    center = sample("T3.selector_x", t3_handle, t3.apply(u, v))
    chosen_x = fvalue(center[0])
    chosen_offset = (FZERO, FZERO)
    candidates = (
        (fsub(u, reciprocal_x), fsub(v, reciprocal_y), fneg(reciprocal_x), fneg(reciprocal_y)),
        (fadd(u, reciprocal_x), fsub(v, reciprocal_y), reciprocal_x, fneg(reciprocal_y)),
        (fsub(u, reciprocal_x), fadd(v, reciprocal_y), fneg(reciprocal_x), reciprocal_y),
        (fadd(u, reciprocal_x), fadd(v, reciprocal_y), reciprocal_x, reciprocal_y),
    )
    for candidate_u, candidate_v, offset_x, offset_y in candidates:
        candidate = sample("T3.selector_x", t3_handle, t3.apply(candidate_u, candidate_v))
        candidate_x = fvalue(candidate[0])
        better = candidate_x < chosen_x if depth_inverted == 0 else candidate_x > chosen_x
        if better:  # strict: source ties retain the earlier candidate.
            chosen_x, chosen_offset = candidate_x, (offset_x, offset_y)
    scale_x = approx.call("div.approx.ftz.f32", t3.extent[0], t2.extent[0])
    scale_y = approx.call("div.approx.ftz.f32", t3.extent[1], t2.extent[1])
    return fmul(chosen_offset[0], scale_x), fmul(chosen_offset[1], scale_y)


def _reconstruct_history(u: int, v: int, *, t1_handle: int, t2_handle: int,
                         t1: Transform, t2: Transform, t3_handle: int, t3: Transform | None,
                         depth_inverted: int, motion_scale: tuple[int, int], width_f32: int,
                         height_f32: int, sample: Callable[[str, int, tuple[int, int]], tuple[int, int, int, int]],
                         approx: ApproxPolicy) -> tuple[tuple[int, int, int], tuple[int, int]]:
    """T2 motion reprojection then exact five-tap T1 cross-cubic reconstruction."""
    if t3_handle:
        if t3 is None:
            raise AssertionError("validated T3 needs transform")
        offset_x, offset_y = _selector_offset(
            u, v, t3_handle=t3_handle, t3=t3, t2=t2,
            depth_inverted=depth_inverted, sample=sample, approx=approx)
    else:
        offset_x, offset_y = FZERO, FZERO
    motion_coordinate = t2.apply(fadd(u, offset_x), fadd(v, offset_y))
    motion = sample("T2.motion", t2_handle, motion_coordinate)
    # T2.Z/W are unconsumed by PTX.  Reprojection uses FMA then multiplication.
    reproject_x = fmul(ffma(motion[0], motion_scale[0], u), width_f32)
    reproject_y = fmul(ffma(motion[1], motion_scale[1], v), height_f32)
    axis_x = _cubic_axis(reproject_x, width_f32, approx)
    axis_y = _cubic_axis(reproject_y, height_f32, approx)
    inverse_width = approx.call("rcp.approx.ftz.f32", width_f32)
    inverse_height = approx.call("rcp.approx.ftz.f32", height_f32)

    def t1_coordinate(x: int, y: int) -> tuple[int, int]:
        return t1.apply(fmul(inverse_width, x), fmul(inverse_height, y))

    # Source order: LM, MT, MM, MB, RM.  Raw alpha remains unconsumed.
    lm = sample("T1.previous_output", t1_handle, t1_coordinate(axis_x.left, axis_y.middle))
    mt = sample("T1.previous_output", t1_handle, t1_coordinate(axis_x.middle, axis_y.left))
    mm = sample("T1.previous_output", t1_handle, t1_coordinate(axis_x.middle, axis_y.middle))
    mb = sample("T1.previous_output", t1_handle, t1_coordinate(axis_x.middle, axis_y.right))
    rm = sample("T1.previous_output", t1_handle, t1_coordinate(axis_x.right, axis_y.middle))
    a = fmul(axis_x.w0, axis_y.wmid)
    b = fmul(axis_y.w0, axis_x.wmid)
    c = fmul(axis_x.wmid, axis_y.wmid)
    d = fmul(axis_y.w3, axis_x.wmid)
    e = fmul(axis_x.w3, axis_y.wmid)
    denominator = fadd(e, fadd(d, fadd(c, fadd(b, a))))
    if fvalue(denominator) == 0:
        raise UnsupportedDomain("history five-tap denominator is zero")
    inverse = approx.call("rcp.approx.ftz.f32", denominator)
    reconstructed = []
    for channel in range(3):
        value = fmul(b, mt[channel])
        value = ffma(a, lm[channel], value)
        value = ffma(c, mm[channel], value)
        value = ffma(d, mb[channel], value)
        value = ffma(e, rm[channel], value)
        reconstructed.append(fmul(inverse, value))
    return tuple(reconstructed), motion_coordinate  # type: ignore[return-value]


def preprocess_temporal(args264: bytes, *, invocation: ValidatedInvocation, sampler: SamplerPolicy,
                        approx: ApproxPolicy, cta: tuple[int, int] = (0, 0)) -> FeatureTile:
    """Build one transparent temporal 16-feature tile from a validated ABI launch."""
    if not isinstance(invocation, ValidatedInvocation):
        raise ValueError("validated host/resource state is required")
    if not isinstance(sampler, SamplerPolicy) or not isinstance(approx, ApproxPolicy):
        raise ValueError("explicit sampler and nonnative approximation callbacks are required")
    if len(cta) != 2 or any(type(value) is not int or value < 0 for value in cta):
        raise ValueError("CTA must be two nonnegative integers")
    handles, counter = _args_handles_counter(args264)
    if handles != invocation.handles or counter != invocation.counter_before:
        raise UnsupportedDomain("validated launch no longer matches immutable ABI bytes")
    if counter == 0:
        raise UnsupportedDomain("this recovery accepts only counter>0")
    if handles[0] == 0:
        raise ValueError("T0 source is mandatory")
    if bool(handles[1]) != bool(handles[2]):
        raise UnsupportedDomain("one nonnull T1/T2 member is not a recovered host family")
    if handles[3] and invocation.t3_role != "selector_x_proven":
        raise UnsupportedDomain("nonnull T3 has no recovered frontend resource semantic")
    if not handles[3] and invocation.t3_role == "selector_x_proven":
        raise UnsupportedDomain("selector proof requires its nonnull T3 ABI handle")
    height, width = struct.unpack_from("<ii", args264, 208)
    if not (2 <= height <= (1 << 30) - 1 and 2 <= width <= (1 << 30) - 1):
        raise UnsupportedDomain("bounded positive source H/W required")
    xs = [8 * cta[0] + x for x in range(8)]
    ys = [8 * cta[1] + y for y in range(8)]
    mirrored_x = [mirror_once(x, width) for x in xs]
    mirrored_y = [mirror_once(y, height) for y in ys]
    source_transform = Transform.from_abi(args264, 136)
    t1 = Transform.from_abi(args264, 40) if handles[1] else None
    t2 = Transform.from_abi(args264, 64) if handles[2] else None
    t3 = Transform.from_abi(args264, 88) if handles[3] else None
    mask_transform = Transform.from_abi(args264, 112) if handles[4] else None
    depth_inverted, tone, structure, style, skin_override, structure_override, mode, gain = struct.unpack_from(
        "<8I", args264, 168)
    motion_scale = struct.unpack_from("<2I", args264, 160)
    scale_half = h32(fadd(gain, gain))
    style_half = h32(style)
    width_f32, height_f32 = f32_rn(width), f32_rn(height)
    us = [approx.call("div.approx.ftz.f32", fadd(f32_rn(value), FHALF), width_f32)
          for value in mirrored_x]
    vs = [approx.call("div.approx.ftz.f32", fadd(f32_rn(value), FHALF), height_f32)
          for value in mirrored_y]
    calls: list[SampleCall] = []

    def sample(role: str, handle: int, coordinate: tuple[int, int]) -> tuple[int, int, int, int]:
        if handle == 0:
            raise UnsupportedDomain(f"attempted sample of null {role}")
        result = sampler.sample(handle, *coordinate)
        if not isinstance(result, (tuple, list)) or len(result) != 4:
            raise ValueError("sampler must return exactly four raw F32 uint32 words")
        rgba = tuple(uint(value, 32) for value in result)
        calls.append(SampleCall(role, handle, coordinate[0], coordinate[1], rgba))
        return rgba  # type: ignore[return-value]

    logical: list[list[list[int]]] = [[[] for _ in range(8)] for _ in range(8)]
    source_coordinates: list[list[tuple[int, int]]] = [[(0, 0) for _ in range(8)] for _ in range(8)]
    guide_coordinates: list[list[tuple[int, int] | None]] = [[None for _ in range(8)] for _ in range(8)]
    pair_used = bool(handles[1])
    for y in range(8):
        for x in range(8):
            u, v = us[x], vs[y]
            source_coordinate = source_transform.apply(u, v)
            source_coordinates[x][y] = source_coordinate
            source = _normalize_rgb(sample("T0.selected_source", handles[0], source_coordinate), scale_half)
            if pair_used:
                if t1 is None or t2 is None:
                    raise AssertionError("paired handles need paired transforms")
                reconstructed, motion_coordinate = _reconstruct_history(
                    u, v, t1_handle=handles[1], t2_handle=handles[2], t1=t1, t2=t2,
                    t3_handle=handles[3], t3=t3, depth_inverted=depth_inverted,
                    motion_scale=motion_scale, width_f32=width_f32, height_f32=height_f32,
                    sample=sample, approx=approx)
                guide = _normalize_rgb((reconstructed[0], reconstructed[1], reconstructed[2], FZERO), scale_half)
                guide_coordinates[x][y] = motion_coordinate
            else:
                guide = source  # PTX mov.b16 copy when either T1/T2 is null.
            mask = sample("T4.control_mask", handles[4], mask_transform.apply(u, v)) if handles[4] else None
            controls = _control_half(tone, structure, skin_override, structure_override, mode, mask)
            logical[x][y] = [*noise_half(xs[x], ys[y], counter & U32, approx), HONE,
                             *source, *guide, style_half, *controls, HZERO]
    return FeatureTile(logical, source_coordinates, guide_coordinates, tuple(calls), sampler.name,
                       approx.name, counter, pair_used, bool(handles[3]))
