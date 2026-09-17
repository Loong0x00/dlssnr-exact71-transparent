"""Independent high-level PRE0 first-counter-zero FEATURE-ONLY candidate.

No PTX parser, register machine, Replay/Thread import, weight read, activation,
MMA, NN prediction, resource synthesis, or native execution. Coordinates and
sampler responses are uint32 F32 encodings; output is logical[x][y][16] half
encodings for ONE 8x8 tile. See INPUT_CONTRACT.json and REPORT.md.

Finite consumed-value domain, including signed zero/subnormals. Nonfinite
consumed inputs/results and overflow FAIL CLOSED, never nan_to_num. All plain
F32/F16 arithmetic below is exact rational -> binary RN-even, not host libm.
Only an explicitly passed NONNATIVE policy evaluates approximate operations.
Dead prologue computations are eliminated; this is not an instruction trace.
"""
from dataclasses import dataclass
from fractions import Fraction
import struct
from typing import Callable

U32 = (1 << 32) - 1
U64 = (1 << 64) - 1
FZERO, FHALF, FONE = 0, 0x3f000000, 0x3f800000
HZERO, HHALF, HONE, HNEGONE = 0, 0x3800, 0x3c00, 0xbc00


class UnsupportedDomain(ValueError):
    pass


def uint(bits, width):
    if type(bits) is not int or not 0 <= bits < (1 << width):
        raise ValueError(f'required raw uint{width} bits, not a float/bool')
    return bits


def u32(x):
    """Original mul.lo/mad.lo/add/xor register result is modulo 2**32."""
    if type(x) is not int:
        raise TypeError('integer required')
    return x & U32


def ftz(bits):
    uint(bits, 32)
    return bits if bits & 0x7f800000 else bits & 0x80000000


def _decode(bits, eb, fb):
    uint(bits, 1 + eb + fb)
    exponent = (bits >> fb) & ((1 << eb) - 1)
    fraction = bits & ((1 << fb) - 1)
    if exponent == (1 << eb) - 1:
        raise UnsupportedDomain('consumed NaN/Inf: native exceptional semantics not admitted')
    bias = (1 << (eb - 1)) - 1
    mantissa = fraction if exponent == 0 else (1 << fb) + fraction
    power = (1 - bias if exponent == 0 else exponent - bias) - fb
    q = Fraction(mantissa << power) if power >= 0 else Fraction(mantissa, 1 << -power)
    return -q if bits >> (eb + fb) else q


def fvalue(bits):
    return _decode(bits, 8, 23)


def hvalue(bits):
    return _decode(bits, 5, 10)


def _rn_ratio(n, d):
    q, r = divmod(n, d)
    return q + int(2*r > d or (2*r == d and q & 1))


def _encode(q, eb, fb, zero_sign=0):
    """Exact rational -> IEEE RN-even including gradual subnormals/signed zero.

    Overflow is rejected instead of silently creating a nonfinite activation.
    """
    q = Fraction(q)
    if q == 0:
        return (zero_sign & 1) << (eb + fb)
    sign = int(q < 0)
    n, d = abs(q.numerator), q.denominator
    e = n.bit_length() - d.bit_length()
    if (n < (d << e)) if e >= 0 else ((n << -e) < d):
        e -= 1
    bias = (1 << (eb - 1)) - 1
    quantum = max(e, 1 - bias) - fb
    m = _rn_ratio(n, d << quantum) if quantum >= 0 else _rn_ratio(n << -quantum, d)
    if m >= (1 << (fb + 1)):
        m >>= 1
        quantum += 1
    signbits = sign << (eb + fb)
    if m < (1 << fb):
        return signbits | m
    expfield = quantum + fb + bias
    if expfield >= (1 << eb) - 1:
        raise UnsupportedDomain('finite arithmetic/conversion overflow')
    return signbits | (expfield << fb) | (m - (1 << fb))


def f32_rn(q, zero_sign=0):
    """No FTZ here: used for conversions; arithmetic wrappers apply FTZ."""
    return _encode(q, 8, 23, zero_sign)


def h32(bits):
    """cvt.rn.f16.f32: no input/output FTZ modifier."""
    return _encode(fvalue(bits), 5, 10, bits >> 31)


def _binary(a, b, eb, fb, multiply=False):
    x, y = _decode(a, eb, fb), _decode(b, eb, fb)
    sa, sb = a >> (eb + fb), b >> (eb + fb)
    if multiply:
        return _encode(x*y, eb, fb, sa ^ sb)
    # RN exact cancellation is +0, except -0 + -0 = -0.
    zs = int(x == 0 and y == 0 and sa == sb == 1)
    return _encode(x+y, eb, fb, zs)


def fadd(a, b):
    return ftz(_binary(ftz(a), ftz(b), 8, 23))


def fsub(a, b):
    return fadd(a, uint(b, 32) ^ 0x80000000)


def fmul(a, b):
    return ftz(_binary(ftz(a), ftz(b), 8, 23, True))


def ffma(a, b, c):
    """fma.rn.ftz.f32: exact product+add, ONE RN F32, then FTZ."""
    a, b, c = ftz(a), ftz(b), ftz(c)
    x, y, z = fvalue(a), fvalue(b), fvalue(c)
    zs = int(x*y == 0 and z == 0 and ((a ^ b) >> 31) == (c >> 31) == 1)
    return ftz(f32_rn(x*y + z, zs))


def hsub(a, b):
    return _binary(a, uint(b, 16) ^ 0x8000, 5, 10)


def hmul(a, b):
    return _binary(a, b, 5, 10, True)


APPROX_OPS = {'div.approx.ftz.f32': 2, 'rcp.approx.ftz.f32': 1,
              'lg2.approx.ftz.f32': 1, 'sqrt.approx.ftz.f32': 1,
              'sin.approx.ftz.f32': 1, 'cos.approx.ftz.f32': 1}


@dataclass(frozen=True)
class ApproxPolicy:
    name: str
    nonnative: bool
    evaluate: Callable[[str, tuple], int]

    def __post_init__(self):
        if not self.name or self.nonnative is not True or not callable(self.evaluate):
            raise ValueError('explicit named NONNATIVE approximation policy required')

    def call(self, op, *operands):
        if op not in APPROX_OPS or len(operands) != APPROX_OPS[op]:
            raise ValueError('approximate operation/arity')
        operands = tuple(ftz(x) for x in operands)
        for x in operands:
            fvalue(x)
        result = ftz(uint(self.evaluate(op, operands), 32))
        fvalue(result)
        return result


@dataclass(frozen=True)
class SamplerPolicy:
    """Explicit opaque policy, NOT a guessed image array/nearest fallback.

    sample(handle64, coordinate_x_f32_bits, coordinate_y_f32_bits) -> four
    uint32 F32 encodings. A declaration of native_bit_oracle is caller evidence,
    not validation by this code. Unknown formats/flags must stay here.
    """
    name: str
    native_bit_oracle: bool
    sample: Callable[[int, int, int], tuple]

    def __post_init__(self):
        if not self.name or type(self.native_bit_oracle) is not bool or not callable(self.sample):
            raise ValueError('named raw-F32 sampler policy required')


@dataclass(frozen=True)
class Transform:
    offset: tuple
    extent: tuple
    post_scale: tuple

    @classmethod
    def from_abi(cls, args264, start):
        words = struct.unpack_from('<6I', args264, start)
        return cls(words[:2], words[2:4], words[4:])

    def apply(self, u, v):
        # Each axis: F32mul(F32fma(normalized, extent, offset), post_scale).
        return tuple(fmul(ffma(c, self.extent[i], self.offset[i]), self.post_scale[i])
                     for i, c in enumerate((u, v)))


def normalize_rgb(raw_rgba, doubled_gain_half):
    """Separate H(texture), Hsub(...,.5), Hmul(...,H(F32add(g,g))).

    If a temporal lift is later authorized, reconstructed guide RGB MUST pass
    through this function separately, not through flattened F32 normalization.
    On first-counter zero, guide is a bitwise COPY of already-normalized source.
    """
    if len(raw_rgba) != 4:
        raise ValueError('four raw F32 sampler channels required')
    for x in raw_rgba:
        uint(x, 32)
    return tuple(hmul(hsub(h32(x), HHALF), doubled_gain_half) for x in raw_rgba[:3])


def noise_integer_words(x, y, seed):
    """PTX lines 203..260; four rounds all derive from SAME hashed state.

    Coordinates are unmirrored. Every low product/mad wraps to uint32 BEFORE
    shifts. Logical shifts only; shift amount is in [4,19], never signed.
    """
    x, y, seed = u32(x), u32(y), u32(seed)
    z = u32(u32(x * -1918454973) ^ u32(y * -669632447)
            ^ u32(seed * -1640531527) ^ 608135816)
    r = u32(((z >> ((z >> 28) + 4)) ^ z) * 277803737)
    state = u32(r ^ (r >> 22))
    words = []
    for a, b in ((747796405, -1403630843), (-93469191, 1192405134),
                 (-895109107, 568162667), (-2094846927, 878960812)):
        t = u32(state * a + b)
        q = u32(((t >> ((t >> 28) + 4)) ^ t) * 277803737)
        words.append(u32(((q >> 30) ^ (q >> 8)) + 1))
    return state, tuple(words)


def noise_half(x, y, seed, approx):
    _, words = noise_integer_words(x, y, seed)
    # cvt.rn.f32.u32 then mul.ftz.f32 by literal 2**-24.
    uniforms = tuple(fmul(f32_rn(i), 0x33800000) for i in words)
    radii = []
    for i in (0, 2):
        logarithm = approx.call('lg2.approx.ftz.f32', uniforms[i])
        ln = fmul(logarithm, 0x3f317218)   # literal ln(2), not host constant
        square = fmul(ln, 0xc0000000)     # separate RN/FTZ multiply by -2
        radii.append(approx.call('sqrt.approx.ftz.f32', square))
    angles = (fmul(uniforms[1], 0x40c90fdb), fmul(uniforms[3], 0x40c90fdb))
    sine = approx.call('sin.approx.ftz.f32', angles[0])
    cosine0 = approx.call('cos.approx.ftz.f32', angles[0])
    cosine1 = approx.call('cos.approx.ftz.f32', angles[1])
    return (h32(fmul(radii[0], cosine0)), h32(fmul(radii[0], sine)),
            h32(fmul(radii[1], cosine1)))


def control_half(tone, structure, skin_override, structure_override, mode, mask_rgba):
    """Features 11..14. Finite FTZ comparisons; no UseAutoMask alias for mode.

    No-mask normal branch converts B,C DIRECTLY (NOT times a synthetic 1).
    This preserves cvt's lack of FTZ and the native optional-resource branch.
    """
    uint(mode, 32)
    for v in (tone, structure, skin_override, structure_override):
        fvalue(v)
    if mode != 0 and mask_rgba is None:
        a, b = fvalue(ftz(skin_override)), fvalue(ftz(structure_override))
        enabled = max(a, b) >= 0
        return (h32(tone), h32(FONE if enabled else structure),
                h32((structure if a < 0 else skin_override) if enabled else 0xbf800000),
                h32((structure if b < 0 else structure_override) if enabled else 0xbf800000))
    if mask_rgba is not None:
        if len(mask_rgba) != 4:
            raise ValueError('four mask channels required')
        for x in mask_rgba:
            uint(x, 32)
        tone, structure = fmul(mask_rgba[1], tone), fmul(mask_rgba[2], structure)
    sentinel = HNEGONE if mode != 0 else HZERO
    return h32(tone), h32(structure), sentinel, sentinel


def launcher_texture_gate(counter64, input_handles, control_mask_handle):
    """Pure host gate, not a resource mutation/temporal implementation.

    Inputs already underwent frontend/global/view-success/routing gates.
    Return (T0..T4, noise_low32, next_host_counter64). The first-counter
    candidate below REJECTS inconsistent ABI rather than silently calling this.
    """
    uint(counter64, 64)
    uint(control_mask_handle, 64)
    if not input_handles:
        raise ValueError('mandatory I[0]')
    for handle in input_handles:
        uint(handle, 64)
    temporal = tuple(input_handles[i] if counter64 != 0 and len(input_handles) > i else 0
                     for i in (1, 2, 3))
    return ((input_handles[0], *temporal, control_mask_handle), counter64 & U32,
            (counter64 + 1) & U64)


def mirror_once(coordinate, extent):
    if type(coordinate) is not int or type(extent) is not int or coordinate < 0 or extent < 2:
        raise UnsupportedDomain('nonnegative coordinate and extent>=2 required')
    mirrored = coordinate if coordinate < extent else 2*extent - coordinate - 2
    if mirrored < 0 or mirrored >= extent:
        raise UnsupportedDomain('tile exceeds ONE upper mirror; no clamp/repeated mirror')
    return mirrored


@dataclass
class FeatureTile:
    logical: list  # logical[x][y][feature], x,y=0..7, each uint16
    source_coordinate_bits: list  # [x][y] transformed source (u,v)
    sample_calls: list  # (handle64, xbits, ybits, RGBA raw bits)
    approx_policy: str
    sampler_policy: str

    def shared_bytes(self):
        """Original fill bytes only; no shared/global loads or first MMA."""
        out = bytearray(2048)
        for x in range(8):
            for y in range(8):
                p = 8*y + x
                for c, h in enumerate(self.logical[x][y]):
                    struct.pack_into('<H', out, 1024*(c//8)+16*p+2*(c%8), uint(h, 16))
        return bytes(out)


def preprocess_firstcounter0(args264, *, sampler, approx, cta=(0, 0)):
    """Transparent one-tile logical equations, NOT a launch or NN evaluator.

    args+200..207 MUST be full64 zero, T1..T3 MUST be zero as host first gate
    dictates. T4 is NOT suppressed: if nonzero its genuine sampler is required.
    Width/height domain avoids original s32 shift/address overflows; odd sizes
    are supported only when all 64 pixels fit ONE upper-side mirror.
    """
    if type(args264) is not bytes or len(args264) != 264:
        raise ValueError('one exact 264-byte ABI aggregate required')
    if not isinstance(sampler, SamplerPolicy):
        raise ValueError('required source sampler raw-F32 policy is missing')
    if not isinstance(approx, ApproxPolicy):
        raise ValueError('explicit NONNATIVE approximate policy is missing')
    if len(cta) != 2 or any(type(c) is not int or c < 0 for c in cta):
        raise ValueError('CTA must be two nonnegative integers')
    handles = struct.unpack_from('<5Q', args264, 0)
    counter, = struct.unpack_from('<Q', args264, 200)
    if counter != 0:
        raise UnsupportedDomain('first-counter-zero ONLY: full64, not low32 noise test')
    if any(handles[1:4]):
        raise UnsupportedDomain('counter-zero ABI has genuine temporal handles; will NOT zero them')
    if handles[0] == 0:
        raise ValueError('mandatory T0 source texture handle is null')
    height, width = struct.unpack_from('<ii', args264, 208)
    if not 2 <= min(height, width) or max(height, width) > (1 << 30) - 1:
        raise UnsupportedDomain('positive bounded s32 dimensions required')
    xs, ys = [8*cta[0]+x for x in range(8)], [8*cta[1]+y for y in range(8)]
    xr = [mirror_once(x, width) for x in xs]
    yr = [mirror_once(y, height) for y in ys]
    source_transform = Transform.from_abi(args264, 136)
    mask_transform = Transform.from_abi(args264, 112) if handles[4] else None
    tone, structure, style, a, b, mode, gain = struct.unpack_from('<7I', args264, 172)
    scale_h = h32(fadd(gain, gain))
    style_h = h32(style)
    width_f, height_f = f32_rn(width), f32_rn(height)
    us = [approx.call('div.approx.ftz.f32', fadd(f32_rn(x), FHALF), width_f) for x in xr]
    vs = [approx.call('div.approx.ftz.f32', fadd(f32_rn(y), FHALF), height_f) for y in yr]
    logical = [[None for _ in range(8)] for _ in range(8)]
    coordinates = [[None for _ in range(8)] for _ in range(8)]
    calls = []

    def sample(handle, xy):
        result = sampler.sample(handle, *xy)
        if not isinstance(result, (tuple, list)) or len(result) != 4:
            raise ValueError('sampler MUST return four raw F32 uint32 words')
        result = tuple(uint(x, 32) for x in result)
        calls.append((handle, *xy, result))
        return result

    for y in range(8):
        for x in range(8):
            noise = noise_half(xs[x], ys[y], counter & U32, approx)
            xy = source_transform.apply(us[x], vs[y])
            coordinates[x][y] = xy
            source = normalize_rgb(sample(handles[0], xy), scale_h)
            guide = source  # Original mov.b16 RGB copy, not a second normalization.
            mask = sample(handles[4], mask_transform.apply(us[x], vs[y])) if handles[4] else None
            controls = control_half(tone, structure, a, b, mode, mask)
            logical[x][y] = [*noise, HONE, *source, *guide, style_h, *controls, HZERO]
    return FeatureTile(logical, coordinates, calls, approx.name, sampler.name)
