"""Finite F32-bit primitives used by the transparent block70 live-TEX model.

Every arithmetic operand/result is a uint32 IEEE-754 bit pattern.  ``.ftz`` is
applied at every PTX-visible input/output.  This is deliberately a small,
fail-closed domain: NaN/Inf, zero divisors, and overflowing synthetic
approximation results are rejected rather than assigned a made-up GPU value,
except that raw F16 infinity is preserved long enough for the separately proved
+104 blend-infinity branch to set its specified zero.

This file is a CPU numerical primitive, not a PTX interpreter and not a claim
about the device's APPROX instructions.  The synthetic APPROX policy below is
only used to freeze/replay the CPU probe with the same explicitly named policy.
"""
from __future__ import annotations

import math
import struct
from fractions import Fraction
from dataclasses import dataclass

U32_MASK = 0xFFFFFFFF
SIGN = 0x80000000
EXP = 0x7F800000
MANT = 0x007FFFFF


class F32DomainError(ValueError):
    """A requested operation has no admitted finite/defined F32 result."""


def u32(value: int) -> int:
    if type(value) is not int:
        raise TypeError("F32 bits must be int")
    return value & U32_MASK


def is_nan(bits: int) -> bool:
    b = u32(bits)
    return (b & EXP) == EXP and (b & MANT) != 0


def is_inf(bits: int) -> bool:
    b = u32(bits)
    return (b & ~SIGN) == EXP


def is_finite(bits: int) -> bool:
    return (u32(bits) & EXP) != EXP


def is_finite_word(bits: int) -> bool:
    """True only for an unsigned raw finite F32 storage word."""
    return type(bits) is int and 0 <= bits <= U32_MASK and is_finite(bits)


def bits_to_float(bits: int) -> float:
    b = u32(bits)
    if not is_finite(b):
        raise F32DomainError("finite F32 bits required")
    return struct.unpack("<f", struct.pack("<I", b))[0]


def float_to_bits(value: float) -> int:
    try:
        b = struct.unpack("<I", struct.pack("<f", float(value)))[0]
    except (OverflowError, struct.error) as exc:
        raise F32DomainError("F32 overflow") from exc
    if not is_finite(b):
        raise F32DomainError("finite F32 result required")
    return b


def ftz(bits: int) -> int:
    """PTX ``.ftz`` for a finite bit pattern, preserving signed zero."""
    b = u32(bits)
    if not is_finite(b):
        raise F32DomainError("finite F32 bits required")
    if (b & EXP) == 0:
        return b & SIGN
    return b


def half_to_f32_bits(bits: int) -> int:
    if type(bits) is not int or not 0 <= bits <= 0xFFFF:
        raise TypeError("F16 bits must be uint16")
    # Exact binary16->binary32 widening.  Preserve non-finite encodings here so
    # the one caller with a proved PTX infinity branch (+104 blend) can handle
    # infinity explicitly; ordinary finite arithmetic still rejects them.
    value = struct.unpack("<e", struct.pack("<H", bits))[0]
    return struct.unpack("<I", struct.pack("<f", value))[0]


def cvt_rn_f32_s32(bits: int) -> int:
    n = u32(bits)
    if n & SIGN:
        n -= 1 << 32
    return float_to_bits(float(n))


def cvt_rn_f32_u32(bits: int) -> int:
    return float_to_bits(float(u32(bits)))


def _round_nearest_required() -> None:
    # The integer/Fraction encoder below implements RN-even directly and is
    # independent of the host floating-point environment.
    return None


def _fraction(bits: int) -> Fraction:
    b=ftz(bits);sign=-1 if b&SIGN else 1;e=(b>>23)&255;m=b&MANT
    if e==0:return Fraction(0)
    n=(1<<23)+m;power=e-127-23;q=Fraction(n<<power) if power>=0 else Fraction(n,1<<-power)
    return sign*q


def _rn_ratio(n: int,d: int) -> int:
    q,r=divmod(n,d);return q+int(2*r>d or (2*r==d and q&1))


def _encode_fraction(value: Fraction,zero_sign: int=0) -> int:
    q=Fraction(value)
    if q==0:return (zero_sign&1)<<31
    sign=int(q<0);n,d=abs(q.numerator),q.denominator;e=n.bit_length()-d.bit_length()
    if (n<(d<<e)) if e>=0 else ((n<<-e)<d):e-=1
    quantum=max(e,-126)-23;m=_rn_ratio(n,d<<quantum) if quantum>=0 else _rn_ratio(n<<-quantum,d)
    if m>=1<<24:m>>=1;quantum+=1
    if m<1<<23:return (sign<<31)|m
    ef=quantum+23+127
    if ef>=255:raise F32DomainError('F32 overflow')
    return (sign<<31)|(ef<<23)|(m-(1<<23))


def add_ftz(a: int, b: int) -> int:
    a,b=ftz(a),ftz(b);x,y=_fraction(a),_fraction(b);zs=int(x==0 and y==0 and a&SIGN and b&SIGN);return ftz(_encode_fraction(x+y,zs))


def sub_ftz(a: int, b: int) -> int:
    return add_ftz(a,u32(b)^SIGN)


def mul_ftz(a: int, b: int) -> int:
    a,b=ftz(a),ftz(b);return ftz(_encode_fraction(_fraction(a)*_fraction(b),((a^b)>>31)&1))


def fma_rn_ftz(a: int, b: int, c: int) -> int:
    a,b,c=ftz(a),ftz(b),ftz(c);x,y,z=_fraction(a),_fraction(b),_fraction(c);zs=int(x*y==0 and z==0 and ((a^b)>>31)==(c>>31)==1);return ftz(_encode_fraction(x*y+z,zs))


def min_ftz(a: int, b: int) -> int:
    aa, bb = ftz(a), ftz(b)
    av, bv = bits_to_float(aa), bits_to_float(bb)
    if av == bv == 0.0:
        return (aa | bb) & SIGN
    return ftz(float_to_bits(min(av, bv)))


def max_ftz(a: int, b: int) -> int:
    aa, bb = ftz(a), ftz(b)
    av, bv = bits_to_float(aa), bits_to_float(bb)
    if av == bv == 0.0:
        return (aa & bb) & SIGN
    return ftz(float_to_bits(max(av, bv)))


def abs_ftz(a: int) -> int:
    return ftz(ftz(a) & ~SIGN)


def gt_ftz(a: int, b: int) -> bool:
    return bits_to_float(ftz(a)) > bits_to_float(ftz(b))


def eq_ftz(a: int, b: int) -> bool:
    return bits_to_float(ftz(a)) == bits_to_float(ftz(b))


def floor_rmi_ftz(a: int) -> int:
    v = bits_to_float(ftz(a))
    return ftz(float_to_bits(float(math.floor(v))))


def clamp01_ftz(a: int) -> int:
    return min_ftz(max_ftz(a, 0x00000000), 0x3F800000)


def apply(op: str, operands: list[int] | tuple[int, ...]) -> int | bool:
    """Compatibility dispatcher for the oracle-only copied CPU VM."""
    b = tuple(int(x) for x in operands)
    if op.startswith("cvt.rn.f32."):
        if op.endswith("s32"):
            return cvt_rn_f32_s32(b[0])
        if op.endswith("u32"):
            return cvt_rn_f32_u32(b[0])
    if op == "selp.f32":
        return u32(b[0]) if b[2] else u32(b[1])
    if op == "add.ftz.f32":
        return add_ftz(*b)
    if op == "sub.ftz.f32":
        return sub_ftz(*b)
    if op == "mul.ftz.f32":
        return mul_ftz(*b)
    if op == "fma.rn.ftz.f32":
        return fma_rn_ftz(*b)
    if op == "min.ftz.f32":
        return min_ftz(*b)
    if op == "max.ftz.f32":
        return max_ftz(*b)
    if op == "abs.ftz.f32":
        return abs_ftz(*b)
    if op == "setp.equ.ftz.f32":
        return eq_ftz(*b)
    if op == "setp.gt.ftz.f32":
        return gt_ftz(*b)
    if op == "cvt.rmi.ftz.f32.f32":
        return floor_rmi_ftz(*b)
    raise F32DomainError("unsupported F32 primitive " + op)


@dataclass(frozen=True)
class SyntheticRNApprox:
    """Explicit deterministic CPU policy for PTX ``*.approx`` probe operands.

    It is intentionally *not* called a native approximation implementation.
    Bounds are enforced before the host calculation, and both the transparent
    forward and the original-PTX CPU oracle receive this exact same instance.
    """

    label: str = "synthetic-RN-f32-probe-not-native"

    @staticmethod
    def _finite_nonzero(bits: int, name: str) -> float:
        value = bits_to_float(ftz(bits))
        if value == 0.0:
            raise F32DomainError(name + " requires nonzero finite input")
        return value

    def div_approx(self, numerator: int, denominator: int) -> int:
        _round_nearest_required()
        n = bits_to_float(ftz(numerator))
        d = self._finite_nonzero(denominator, "div.approx.ftz.f32")
        return ftz(float_to_bits(n / d))

    def rcp_approx(self, value: int) -> int:
        _round_nearest_required()
        v = self._finite_nonzero(value, "rcp.approx.ftz.f32")
        return ftz(float_to_bits(1.0 / v))

    def ex2_approx(self, value: int) -> int:
        _round_nearest_required()
        x = bits_to_float(ftz(value))
        # A finite F32 result is the intentional boundary.  No inferred
        # saturation/NaN behavior is supplied beyond this admitted probe domain.
        try:
            return ftz(float_to_bits(math.exp2(x)))
        except OverflowError as exc:
            raise F32DomainError("ex2.approx.ftz.f32 overflow") from exc
