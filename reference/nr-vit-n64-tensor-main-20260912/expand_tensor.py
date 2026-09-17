"""CPU/static N64 candidate64x4096 reference for ``cc_vit_1d_ffn_expand_fp8``.

This is a learned expand *operator*, not a native/GPU equivalence claim.  It
unpacks the source-packed tensors, applies 32 float64 K=32 GEMMs with a
binary16 RN boundary after each GEMM, runs the source-pinned f16 tail, then
re-packs both logical and physical output tensors.  The CTA-x map is derived
from the literal launch/address program rather than treating CTA-0 as a dense
forward by assumption.
"""
from __future__ import annotations

import hashlib
import re
import struct
from pathlib import Path
from typing import Callable, Mapping

import numpy as np

HERE = Path(__file__).resolve().parent
SOURCE = HERE.parent / 'nr-vit-expand-bounds-main-20260912' / 'original.ptx'
SOURCE_SHA256 = '1e82f9620951828e94ab2081d2912a28d05dd4b39a35e0ff2bce97f76e027aa5'
M, K, N, CTAS = 64, 1024, 4096, 32
INPUT_BYTES, WEIGHT_BYTES, OUTPUT_BYTES = M * K, K * N, M * N
# The literal ABI supplies a 4,194,320-byte slab.  The final 16 bytes are
# outside the proven B map; retain the exact ABI size instead of trimming it.
SLAB_BYTES = 4194320


class UnsupportedNonFinite(ValueError):
    """The finite-only source policy cannot represent this operand/carry."""


def _source_text() -> str:
    text = SOURCE.read_text()
    if hashlib.sha256(text.encode()).hexdigest() != SOURCE_SHA256:
        raise ValueError('pinned original.ptx changed')
    return text


def input_offset(m: int, k: int) -> int:
    """Source-derived physical byte location of logical A[m,k]."""
    if not (0 <= m < M and 0 <= k < K):
        raise ValueError('logical A coordinate')
    lane = 4 * (m % 8) + (k % 16) // 4
    return (16384 * (m // 16) + 512 * (k // 32) + 16 * lane
            + 8 * ((k % 32) // 16) + 4 * ((m % 16) // 8) + k % 4)


def weight_offset(k: int, n: int) -> int:
    """Source-derived physical byte location of logical B[k,n]."""
    if not (0 <= k < K and 0 <= n < N):
        raise ValueError('logical B coordinate')
    return ((k // 32) * 131072 + (n // 16) * 512
            + 16 * (4 * (n % 8) + (k % 16) // 4)
            + 8 * ((n % 16) // 8) + 4 * ((k % 32) // 16) + k % 4)


def _offset_array(fn: Callable[[int, int], int], rows: int, columns: int) -> np.ndarray:
    # Built once, explicitly from the inspectable scalar maps above.
    return np.fromiter((fn(i, j) for i in range(rows) for j in range(columns)),
                       dtype=np.intp, count=rows * columns).reshape(rows, columns)


INPUT_MAP = _offset_array(input_offset, M, K)
WEIGHT_MAP = _offset_array(weight_offset, K, N)


def _e4m3_value(code: int) -> float:
    if code in (0x7f, 0xff):
        raise UnsupportedNonFinite('E4M3FN NaN operand')
    sign = -1.0 if code & 0x80 else 1.0
    exponent, fraction = (code >> 3) & 15, code & 7
    return sign * ((fraction / 512.0) if exponent == 0
                   else (8 + fraction) * 2.0 ** (exponent - 10))


# This validated finite-code LUT intentionally has no entry for either NaN
# code.  Decode checks those codes before indexing it.
E4M3_LUT = np.array([_e4m3_value(i) if i not in (0x7f, 0xff) else np.nan
                      for i in range(256)], dtype=np.float64)
E4_POSITIVE = E4M3_LUT[:0x7f].copy()


def decode_e4m3(packed: bytes, offsets: np.ndarray) -> np.ndarray:
    if type(packed) is not bytes:
        raise ValueError('packed tensor must be immutable bytes')
    raw = np.frombuffer(packed, dtype=np.uint8)
    codes = raw[offsets]
    if np.any((codes == 0x7f) | (codes == 0xff)):
        raise UnsupportedNonFinite('E4M3FN NaN input/weight operand refused')
    return E4M3_LUT[codes]


def _half_from_bits(bits: int) -> np.float16:
    return np.array([bits], dtype=np.uint16).view(np.float16)[0]


# Literal f32->f16 constants: source_tail.py independently pins these exact
# values.  Do not substitute activation-name coefficients.
CUBIC_C, CUBIC_B, CUBIC_A = 0xab28, 0x3728, 0x3b28


def _validate_tail_source() -> None:
    """Pin the tail literals/instructions before using the finite algorithm."""
    source = _source_text()
    expected_f32 = {559: 0xc0800000, 560: 0x40800000, 561: 0x3f650000,
                    562: 0x3ee50000, 563: 0xbd650000}
    for register, expected in expected_f32.items():
        values = re.findall(r'mov\.b32 %r' + str(register) + r', (-?\d+);', source)
        if len(values) != 1 or (int(values[0]) & 0xffffffff) != expected:
            raise ValueError(f'changed source tail literal r{register}')
    converted = tuple(struct.unpack('<H', struct.pack('<e', struct.unpack('<f', struct.pack('<I', expected_f32[r]))[0]))[0]
                      for r in (563, 562, 561))
    if converted != (CUBIC_C, CUBIC_B, CUBIC_A):
        raise ValueError('source f32->f16 cubic constants changed')
    for instruction in ('min.f16x2 %r565,%r1129,%r564;',
                        'max.f16x2 %r567,%r565,%r566;',
                        'fma.rn.f16x2 %r571,%r568,%r569,%r570;',
                        'fma.rn.f16x2 %r573,%r567,%r571,%r572;',
                        'mul.f16x2 %r952,%r1129,%r573;',
                        'cvt.rn.satfinite.e4m3x2.f16x2 %rs52, %r952;'):
        if instruction not in source:
            raise ValueError('changed source tail instruction')




def _rn_half(values: np.ndarray, stage: str) -> np.ndarray:
    """RN-even binary16 conversion plus the finite-carry-domain guard."""
    values = np.asarray(values, dtype=np.float64)
    if not np.all(np.isfinite(values)):
        raise UnsupportedNonFinite(f'non-finite f64 before {stage}')
    with np.errstate(over='ignore', invalid='ignore'):
        rounded = values.astype(np.float16)
    if not np.all(np.isfinite(rounded)):
        raise UnsupportedNonFinite(f'binary16 overflow/non-finite at {stage}')
    return rounded


def _readonly_snapshot(array: np.ndarray) -> np.ndarray:
    """An immutable detached view; hooks cannot alter the operator's tensors."""
    frozen = np.frombuffer(np.ascontiguousarray(array).tobytes(), dtype=array.dtype)
    return frozen.reshape(array.shape)


def _call_hook(hooks: Mapping[str, Callable[[np.ndarray], None]] | None,
               name: str, array: np.ndarray) -> None:
    if hooks is None:
        return
    if not isinstance(hooks, Mapping):
        raise ValueError('hooks must be a mapping of tensor name to callback')
    callback = hooks.get(name)
    if callback is not None:
        if not callable(callback):
            raise ValueError(f'hook {name!r} is not callable')
        callback(_readonly_snapshot(array))


def _activate(preactivation: np.ndarray) -> np.ndarray:
    """Literal finite f16 cubic tail, with RN after each f16 instruction."""
    if preactivation.dtype != np.float16 or not np.all(np.isfinite(preactivation)):
        raise UnsupportedNonFinite('finite binary16 preactivation required')
    x = preactivation.astype(np.float64)
    clamped = np.where(x < -4.0, -4.0, np.where(x > 4.0, 4.0, x))
    c, b, a = (float(_half_from_bits(v)) for v in (CUBIC_C, CUBIC_B, CUBIC_A))
    p = _rn_half(c * np.abs(clamped) + b, 'tail fma 1')
    p = _rn_half(clamped * p.astype(np.float64) + a, 'tail fma 2')
    return _rn_half(x * p.astype(np.float64), 'tail mul')


def fp8_rn_satfinite(halves: np.ndarray) -> np.ndarray:
    """Validated finite E4M3FN RN-even/saturating pack; NaNs are refused."""
    if halves.dtype != np.float16 or not np.all(np.isfinite(halves)):
        raise UnsupportedNonFinite('finite binary16 activation required for E4M3 pack')
    bits = halves.view(np.uint16)
    sign = ((bits >> 8) & 0x80).astype(np.uint8)
    value = np.abs(halves.astype(np.float64))
    # searchsorted over the finite, monotone positive E4M3FN table.
    upper = np.searchsorted(E4_POSITIVE, value, side='left')
    result = np.empty(value.shape, dtype=np.uint8)
    result[upper == 0] = 0
    result[upper == len(E4_POSITIVE)] = 0x7e
    middle = (upper > 0) & (upper < len(E4_POSITIVE))
    hi = upper[middle]
    lo = hi - 1
    low_distance = value[middle] - E4_POSITIVE[lo]
    high_distance = E4_POSITIVE[hi] - value[middle]
    choose_low = ((low_distance < high_distance) |
                  ((low_distance == high_distance) & ((lo & 1) == 0)))
    result[middle] = np.where(choose_low, lo, hi).astype(np.uint8)
    return result | sign


def d_word_slots(lane: int, word: int) -> tuple[tuple[int, int], tuple[int, int]]:
    group, thread = divmod(lane, 4)
    return ((group + 8 * word, 2 * thread), (group + 8 * word, 2 * thread + 1))


def _source_schedule() -> tuple[dict[tuple[int, int], tuple[int, int]], dict[int, int]]:
    source = _source_text()
    calls = re.findall(r'mma\.sync\.aligned\.m16n8k32\.row\.col\.f16\.e4m3\.e4m3\.f16\s*\{%r(\d+), %r(\d+)\},\s*\{%r(\d+),', source)
    if len(calls) != 64:
        raise ValueError('changed MMA schedule')
    a_to_mt = {363: 0, 367: 0, 387: 1, 391: 1, 411: 2, 415: 2, 435: 3, 439: 3}
    first_b = re.findall(r'mma\.sync\.aligned\.m16n8k32\.row\.col\.f16\.e4m3\.e4m3\.f16\s*\{%r\d+, %r\d+\},\s*\{%r\d+, %r\d+, %r\d+, %r\d+\},\s*\{%r(\d+),', source)
    b_to_nt = {1164: 0, 1162: 1, 1160: 2, 1158: 3, 1156: 4, 1154: 5, 1152: 6, 1150: 7,
               1148: 0, 1146: 1, 1144: 2, 1142: 3, 1140: 4, 1165: 5, 1167: 6, 1169: 7}
    tiles = {}
    for (d0, d1, a0), b0 in zip(calls, first_b):
        if int(a0) not in a_to_mt or int(b0) not in b_to_nt:
            raise ValueError('changed A/B tile schedule')
        tiles[(int(d0), int(d1))] = (a_to_mt[int(a0)], b_to_nt[int(b0)])
    mul = dict((int(dst), int(src)) for dst, src in re.findall(
        r'\{mul\.f16x2 %r(9(?:5[2-9]|[6-9]\d)|10(?:0\d|1[0-5])),%r(\d+),%r\d+;\s*\}', source))
    converted = [(int(q), int(reg)) for q, reg in re.findall(
        r'cvt\.rn\.satfinite\.e4m3x2\.f16x2 %rs(\d+), %r(\d+);', source) if 52 <= int(q) <= 115]
    tail = {q: mul[reg] for q, reg in converted}
    if len(tail) != 64 or not set(tail.values()).issubset({r for pair in tiles for r in pair}):
        raise ValueError('changed tail schedule')
    return tiles, tail


TILES = {(371, 372): (0, 0), (373, 374): (0, 1), (1129, 1128): (0, 0), (1127, 1126): (0, 1), (375, 376): (0, 2), (377, 378): (0, 3), (1125, 1124): (0, 2), (1123, 1122): (0, 3), (379, 380): (0, 4), (381, 382): (0, 5), (1121, 1120): (0, 4), (1119, 1118): (0, 5), (383, 384): (0, 6), (385, 386): (0, 7), (1117, 1116): (0, 6), (1115, 1114): (0, 7), (395, 396): (1, 0), (397, 398): (1, 1), (1113, 1112): (1, 0), (1111, 1110): (1, 1), (399, 400): (1, 2), (401, 402): (1, 3), (1109, 1108): (1, 2), (1107, 1106): (1, 3), (403, 404): (1, 4), (405, 406): (1, 5), (1105, 1104): (1, 4), (1103, 1102): (1, 5), (407, 408): (1, 6), (409, 410): (1, 7), (1101, 1100): (1, 6), (1099, 1098): (1, 7), (419, 420): (2, 0), (421, 422): (2, 1), (1097, 1096): (2, 0), (1095, 1094): (2, 1), (423, 424): (2, 2), (425, 426): (2, 3), (1093, 1092): (2, 2), (1091, 1090): (2, 3), (427, 428): (2, 4), (429, 430): (2, 5), (1089, 1088): (2, 4), (1087, 1086): (2, 5), (431, 432): (2, 6), (433, 434): (2, 7), (1085, 1084): (2, 6), (1083, 1082): (2, 7), (443, 444): (3, 0), (445, 446): (3, 1), (1081, 1080): (3, 0), (1079, 1078): (3, 1), (447, 448): (3, 2), (449, 450): (3, 3), (1077, 1076): (3, 2), (1075, 1074): (3, 3), (451, 452): (3, 4), (453, 454): (3, 5), (1130, 1131): (3, 4), (1132, 1133): (3, 5), (455, 456): (3, 6), (457, 458): (3, 7), (1134, 1135): (3, 6), (1136, 1137): (3, 7)}
TAIL = {52: 1129, 53: 1127, 54: 1128, 55: 1126, 56: 1125, 57: 1123, 58: 1124, 59: 1122, 60: 1121, 61: 1119, 62: 1120, 63: 1118, 64: 1117, 65: 1115, 66: 1116, 67: 1114, 68: 1113, 69: 1111, 70: 1112, 71: 1110, 72: 1109, 73: 1107, 74: 1108, 75: 1106, 76: 1105, 77: 1103, 78: 1104, 79: 1102, 80: 1101, 81: 1099, 82: 1100, 83: 1098, 84: 1097, 85: 1095, 86: 1096, 87: 1094, 88: 1093, 89: 1091, 90: 1092, 91: 1090, 92: 1089, 93: 1087, 94: 1088, 95: 1086, 96: 1085, 97: 1083, 98: 1084, 99: 1082, 100: 1081, 101: 1079, 102: 1080, 103: 1078, 104: 1077, 105: 1075, 106: 1076, 107: 1074, 108: 1130, 109: 1132, 110: 1131, 111: 1133, 112: 1134, 113: 1136, 114: 1135, 115: 1137}
STORE_BASE = (0, 512, 65536, 66048, 131072, 131584, 196608, 197120)


def output_byte_offsets(q: int, lane: int) -> tuple[int, int]:
    if not (52 <= q <= 115 and 0 <= lane < 32):
        raise ValueError('tail slot')
    group = q - 52
    return tuple(STORE_BASE[group // 8] + lane * 16 + 2 * (group % 8) + half for half in (0, 1))


def cta_slots(cx: int):
    """Yield (physical byte, logical-m, logical-n) for one source CTA.

    The proven CTA relation is r58=1, r2=cx, r3=0, and
    r6=(128*cx)|(64*odd_warp).  Thus CTA-x shifts logical B/output n by
    128*cx and the global output byte base by 2048*cx.
    """
    if type(cx) is not int or not 0 <= cx < CTAS:
        raise ValueError('CTA x must be in [0,31]')
    for warp in range(4):
        if 64 * (warp // 2) >= M:
            continue  # original N64 output predicate, not input padding
        base = 262144 * (warp // 2) + 1024 * (warp % 2) + 2048 * cx
        for q, register in sorted(TAIL.items()):
            pair = next(pair for pair in TILES if register in pair)
            mt, nt = TILES[pair]
            word = pair.index(register)
            for lane in range(32):
                for half, (row, column) in enumerate(d_word_slots(lane, word)):
                    yield (base + output_byte_offsets(q, lane)[half],
                           64 * (warp // 2) + mt * 16 + row,
                           128 * cx + 64 * (warp % 2) + nt * 8 + column)


def _physical_map() -> np.ndarray:
    mapping = np.empty((M, N), dtype=np.intp)
    seen = set()
    for cx in range(CTAS):
        slots = list(cta_slots(cx))
        if len(slots) != 8192:
            raise AssertionError('CTA output byte count')
        for offset, m, n in slots:
            if offset in seen:
                raise AssertionError('CTA output overlap')
            seen.add(offset)
            mapping[m, n] = offset
    if seen != set(range(OUTPUT_BYTES)):
        raise AssertionError('full source output grid has holes')
    return mapping


OUTPUT_MAP = _physical_map()


def metadata() -> dict:
    return dict(operator='cc_vit_1d_ffn_expand_fp8', source=str(SOURCE),
                source_sha256=SOURCE_SHA256, shape=dict(M=M, K=K, N=N),
                launch=dict(CTA=[32, 4, 1], grid=[32, 1, 1]),
                rounding='float64 dot(K=32), RN-f16 after every K=32 block',
                nonfinite_policy='E4M3 NaN and non-finite f16 carry/tail are refused',
                logical_n_shift='128*ctaid.x', global_output_byte_shift='2048*ctaid.x')


def forward(input_fp8: bytes, weight_slab: bytes,
            hooks: Mapping[str, Callable[[np.ndarray], None]] | None = None) -> dict[str, np.ndarray]:
    """Run all 32 proven CTA groups and return inspectable logical tensors.

    Returned keys are ``logical_input`` (float64), ``logical_weight``
    (float64), ``preactivation`` (float16), ``output`` (logical E4M3 bytes),
    and ``output_physical`` (the source's 262144-byte packed layout). Hooks
    may observe any of the first four tensor names; each receives a detached,
    read-only snapshot and cannot modify activation computation.
    """
    if type(input_fp8) is not bytes or len(input_fp8) != INPUT_BYTES:
        raise ValueError('65536-byte packed A required')
    if type(weight_slab) is not bytes or len(weight_slab) != SLAB_BYTES:
        raise ValueError('4194320-byte raw B slab required')
    a = decode_e4m3(input_fp8, INPUT_MAP)
    b = decode_e4m3(weight_slab, WEIGHT_MAP)
    _call_hook(hooks, 'logical_input', a)
    _call_hook(hooks, 'logical_weight', b)
    carry = np.zeros((M, N), dtype=np.float16)
    # All finite E4M3 products are dyadic.  A K=32 sum plus finite f16 carry
    # spans <53 binary positions (down through 2^-24 and below 2^23), so the
    # float64 GEMM/add is exact before this explicit RN-half boundary.
    for k0 in range(0, K, 32):
        dot32 = a[:, k0:k0 + 32] @ b[k0:k0 + 32, :]
        carry = _rn_half(dot32 + carry.astype(np.float64), f'K32 block {k0 // 32}')
    _call_hook(hooks, 'preactivation', carry)
    output = fp8_rn_satfinite(_activate(carry))
    _call_hook(hooks, 'output', output)
    physical = np.empty(OUTPUT_BYTES, dtype=np.uint8)
    physical[OUTPUT_MAP] = output
    return dict(logical_input=a, logical_weight=b, preactivation=carry,
                output=output, output_physical=physical)
