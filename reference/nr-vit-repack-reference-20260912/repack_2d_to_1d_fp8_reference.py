"""CPU/static, byte-exact reference for module05's FP8 2-D -> 1-D repack.

This is a literal b32 load/store permutation, not an FP8 conversion or a
model-forward implementation.  It models only the valid non-negative i32
geometry envelope used by the static host launch arithmetic.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import hashlib
import struct

KERNEL_NAME = "cc_vit_1d_repack_2d_to_1d_fp8"
PARAM_BYTES = 0x18
CTA = (256, 1, 1)
WORD_BYTES = 4
WORDS_PER_LOGICAL_INDEX = 256
WORDS_PER_4X4_TILE = 4096
WORDS_PER_GRID_X = 8192
I32_MAX = (1 << 31) - 1

# These are the host record's byte positions, not guessed tensor names.
ABI_FIELDS = {
    0x00: "input I[0] b32 source pointer",
    0x08: "destination O[1] b32 pointer",
    0x10: "arg5 (i32 geometry factor)",
    0x14: "arg6 (i32 geometry factor)",
}
MAIN_EXPAND_HANDOFF = {"repack_output": "O[1]", "main_expand_param_offset": 0x00}


class RepackContractError(ValueError):
    """The requested CPU launch is outside source-established contracts."""


@dataclass(frozen=True)
class RepackRecord24:
    """Exact host-side 0x18-byte record; pointer values are symbolic on CPU."""

    input_i0: int
    destination_o1: int
    arg5: int
    arg6: int

    def __post_init__(self) -> None:
        for name, pointer in (("input_i0", self.input_i0), ("destination_o1", self.destination_o1)):
            if type(pointer) is not int or not 0 <= pointer < (1 << 64):
                raise RepackContractError(f"{name} must be a u64 pointer value")
        # Reuse the CPU geometry envelope, without treating the pointer values as host pointers.
        RepackGeometry(self.arg5, self.arg6)

    def to_bytes(self) -> bytes:
        return struct.pack("<QQii", self.input_i0, self.destination_o1, self.arg5, self.arg6)

    @classmethod
    def from_bytes(cls, record: bytes | bytearray | memoryview) -> "RepackRecord24":
        try:
            raw = bytes(memoryview(record).cast("B"))
        except TypeError as exc:
            raise RepackContractError("record must be byte-addressable") from exc
        if len(raw) != PARAM_BYTES:
            raise RepackContractError("repack record must be exactly 0x18 bytes")
        return cls(*struct.unpack("<QQii", raw))


def _ceil_div(value: int, divisor: int) -> int:
    return (value + divisor - 1) // divisor


def _valid_factor(value: int, name: str) -> None:
    if type(value) is not int or not 0 <= value <= I32_MAX:
        raise RepackContractError(f"{name} must be a non-negative signed i32")


@dataclass(frozen=True)
class RepackGeometry:
    """The two positional factors at record +0x10/+0x14; they are not H/W."""

    arg5: int
    arg6: int

    def __post_init__(self) -> None:
        _valid_factor(self.arg5, "arg5")
        _valid_factor(self.arg6, "arg6")
        if self.arg5 * self.arg6 > I32_MAX:
            raise RepackContractError("arg5*arg6 must fit the PTX signed i32 product")
        # r71 is the signed-b32 work bound formed from ceil(N/32)*8192.
        if self.n and _ceil_div(self.n, 32) * WORDS_PER_GRID_X > I32_MAX:
            raise RepackContractError("geometry overflows the PTX signed-b32 repack work bound")

    @property
    def n(self) -> int:
        return self.arg5 * self.arg6

    @property
    def grid_x(self) -> int:
        # Static host FP8 branch: ceil(N/32), with N=arg5*arg6.
        return _ceil_div(self.n, 32) if self.n else 0

    @property
    def padded_logical_indices(self) -> int:
        return self.grid_x * 32

    @property
    def source_tile_column_stride(self) -> int:
        """Literal r8 = ceil(arg6/4), used even by either special tile-zero path."""
        return _ceil_div(self.arg6, 4) if self.arg6 else 0

    @staticmethod
    def _tile_coordinate(coordinate: int, factor: int) -> int:
        # `selp` forces this coordinate to zero exactly when factor&-4 == 4.
        return 0 if (factor & -4) == 4 else coordinate // 4

    @property
    def source_max_tile_index(self) -> int:
        """Largest literal r49 tile base reached by active logical indices."""
        if not self.n:
            return -1
        return (self._tile_coordinate(self.arg5 - 1, self.arg5) * self.source_tile_column_stride
                + self._tile_coordinate(self.arg6 - 1, self.arg6))

    @property
    def source_tiled_address_span_bytes(self) -> int:
        """Tiled b32 address span implied by literal tile bases, not allocation proof."""
        return (self.source_max_tile_index + 1) * WORDS_PER_4X4_TILE * WORD_BYTES

    @property
    def minimum_source_read_bytes(self) -> int:
        """Smallest byte extent covering every actual active `ld.global.b32`."""
        if not self.n:
            return 0
        # A factor 4..7 collapses all of its four-wide coordinates to tile zero;
        # otherwise the final tile has only its final partial rectangle.
        row_count = min(self.arg5, 4) if self.arg5 < 8 else (self.arg5 - 1) % 4 + 1
        column_count = min(self.arg6, 4) if self.arg6 < 8 else (self.arg6 - 1) % 4 + 1
        inner_max = max(3982 + (t // 8) + 16 * (t % 8)
                        for row in range(row_count) for column in range(column_count)
                        for t in (row * 4 + column,))
        return (self.source_max_tile_index * WORDS_PER_4X4_TILE + inner_max + 1) * WORD_BYTES

    @property
    def output_write_bytes(self) -> int:
        # Every word in this span is written, including the N..padded-N zero fill.
        return self.grid_x * WORDS_PER_GRID_X * WORD_BYTES


@dataclass(frozen=True)
class LaneAccess:
    """One `$L__BB35_4` -> `$L__BB35_6` iteration for a concrete CTA/lane."""

    ctaid_x: int
    tid_x: int
    iteration: int
    work_index_r72: int
    logical_index_r11: int
    lane_index_r9: int
    destination_word: int
    source_word: int | None

    @property
    def is_padding_write(self) -> bool:
        return self.source_word is None


def source_word_index(geometry: RepackGeometry, logical_index: int, lane_index: int) -> int:
    """Literal active-path b32 address index produced before `ld.global.b32`.

    This is the source's 4x4 tiled physical map.  A word is four raw FP8
    bytes; no dimension is called a channel because the PTX does not do so.
    """
    if not 0 <= logical_index < geometry.n:
        raise RepackContractError("active logical index outside N")
    if not 0 <= lane_index < WORDS_PER_LOGICAL_INDEX:
        raise RepackContractError("lane index must be in [0,255]")
    row, column = divmod(logical_index, geometry.arg6)
    tile = (geometry._tile_coordinate(row, geometry.arg5) * geometry.source_tile_column_stride
            + geometry._tile_coordinate(column, geometry.arg6))
    t = (row % 4) * 4 + (column % 4)
    q = lane_index
    # Normalized from the b16 signed/unsigned shift sequence in $L__BB35_4.
    inner = ((q // 8) * 128 + ((q % 8) // 4) * 2 + (t // 8)
             + (t % 8) * 16 + (q % 4) * 4)
    return tile * WORDS_PER_4X4_TILE + inner


def destination_word_index(work_index_r72: int) -> int:
    """Literal `$L__BB35_6` b32 destination index for non-negative r72."""
    if type(work_index_r72) is not int or work_index_r72 < 0:
        raise RepackContractError("work index must be non-negative")
    logical_index, q = divmod(work_index_r72, WORDS_PER_LOGICAL_INDEX)
    row16 = logical_index % 16
    return ((work_index_r72 // 4096) * 4096 + (q // 8) * 128 + (row16 // 8)
            + ((q % 8) // 4) * 2 + 4 * (((row16 % 8) * 16 + 4 * (q % 4)) // 4))


def lane_accesses(geometry: RepackGeometry, ctaid_x: int, tid_x: int) -> tuple[LaneAccess, ...]:
    """Exact host-grid lane work sequence, including each padded zero store."""
    if type(ctaid_x) is not int or not 0 <= ctaid_x < geometry.grid_x:
        raise RepackContractError("ctaid.x outside host FP8 grid")
    if type(tid_x) is not int or not 0 <= tid_x < CTA[0]:
        raise RepackContractError("tid.x outside CTA(256,1,1)")
    stride = geometry.grid_x * CTA[0]
    limit = geometry.grid_x * WORDS_PER_GRID_X
    work = ctaid_x * CTA[0] + tid_x
    accesses = []
    iteration = 0
    while work < limit:
        logical, q = divmod(work, WORDS_PER_LOGICAL_INDEX)
        source = source_word_index(geometry, logical, q) if logical < geometry.n else None
        accesses.append(LaneAccess(ctaid_x, tid_x, iteration, work, logical, q,
                                   destination_word_index(work), source))
        work += stride
        iteration += 1
    return tuple(accesses)


def run_repack(source: bytes | bytearray | memoryview, arg5: int, arg6: int,
               destination: bytearray | None = None) -> bytearray:
    """Run the source-derived CPU b32 load/store schedule.

    `source` is I[0].  `destination` is O[1], and is modified in place when
    supplied.  At least the actual read footprint and complete write footprint
    are required.  Bytes beyond those footprints are untouched.
    """
    geometry = RepackGeometry(arg5, arg6)
    try:
        source_view = memoryview(source).cast("B")
    except TypeError as exc:
        raise RepackContractError("I[0] must be a byte-addressable buffer") from exc
    if len(source_view) < geometry.minimum_source_read_bytes:
        raise RepackContractError("I[0] does not cover all literal b32 reads")
    if destination is None:
        destination = bytearray(geometry.output_write_bytes)
    if type(destination) is not bytearray:
        raise RepackContractError("O[1] must be a mutable bytearray")
    if len(destination) < geometry.output_write_bytes:
        raise RepackContractError("O[1] does not cover all literal b32 writes")

    # This is the source's CTA/lane/loop traversal.  All stores are raw 4-byte copies.
    for ctaid_x in range(geometry.grid_x):
        for tid_x in range(CTA[0]):
            for access in lane_accesses(geometry, ctaid_x, tid_x):
                out_at = access.destination_word * WORD_BYTES
                if access.source_word is None:  # `%r73` remains the literal zero.
                    destination[out_at:out_at + WORD_BYTES] = b"\0\0\0\0"
                else:
                    in_at = access.source_word * WORD_BYTES
                    destination[out_at:out_at + WORD_BYTES] = source_view[in_at:in_at + WORD_BYTES]
    return destination


def literal_entry_path() -> Path:
    return Path(__file__).resolve().parent.parent / "nr-ptx-bundle-20260911/modules/module-05-dll-00dd4bc0.ptx"


def literal_entry_text() -> str:
    source = literal_entry_path().read_text()
    start = source.index(f".visible .entry {KERNEL_NAME}(")
    end = source.index(".visible .entry cc_vit_1d_repack_1d_to_2d(", start)
    return source[start:end]


def assert_literal_provenance() -> None:
    """Check the pinned PTX header and the instructions on which this map rests."""
    source = literal_entry_path().read_bytes()
    if hashlib.sha256(source).hexdigest() != "641ca106cd3e211587f873de41ba4350608413687b1bb99c33fa7529f82b510c":
        raise RepackContractError("module05 PTX hash differs from pinned source")
    text = source.decode()
    if ".version 9.4" not in text:
        raise RepackContractError("pinned PTX header is not version 9.4")
    entry = literal_entry_text()
    required = (
        f".param .align 8 .b8 {KERNEL_NAME}_param_0[24]",
        f"ld.param.b64 %rd1, [{KERNEL_NAME}_param_0];",
        f"ld.param.b64 %rd2, [{KERNEL_NAME}_param_0+8];",
        f"ld.param.v2.b32 {{%r1, %r2}}, [{KERNEL_NAME}_param_0+16];",
        "mov.u32 %r4, %ntid.x;",
        "mov.u32 %r19, %tid.x;",
        "mul.lo.s32 %r7, %r20, %r4;",
        "div.s32 %r28, %r11, %r2;",
        "mad.lo.s32 %r49, %r44, %r8, %r45;",
        "shl.b32 %r50, %r49, 12;",
        "mul.wide.s16 %r51, %rs36, 128;",
        "and.b32 %r65, %r57, -4096;",
        "mul.wide.s16 %r66, %rs72, 128;",
        "ld.global.b32 %r73, [%rd6];",
        "st.global.b32 [%rd8], %r73;",
    )
    if any(line not in entry for line in required):
        raise RepackContractError("literal repack entry no longer matches this reference")
