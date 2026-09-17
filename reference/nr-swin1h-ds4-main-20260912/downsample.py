"""Transparent DS4 *suffix* candidate, not a complete NN or native predictor.

Input F is 64 rows (8x8 RASTER), 32 unquantized final half neurons/row.
Channel c means final MMA N-neuron, not a packed byte rank. If Main stores
four 4x4 tiles, use tile_rows_to_raster() before calling this helper.

All arithmetic is SUPPLIED: half add/mul RN, half constants, FP8 RN-satfinite,
and a real one-K32 m16n8 MMA policy. There is no default MMA, identity fallback,
fitting, imported VM, source execution, file access, or numpy dot hidden here.

main_pi maps physical K / serialized channel rank -> N-neuron canonical.
inverse_main_pi is the prior report's `pi`, NOT main_pi.
The output exposes both raw MMA-N and serialized-rank channel bases.

Only learned four-plane payload stores are packed here. The original eight-
plane padding remains a SEPARATE static address contract below; no merged
allocation, padding repair, buffer mutation, or cross-CTA store order claimed.
"""
from dataclasses import dataclass
from typing import Any, Callable, Sequence

Half = Any  # policy-owned scalar; can be uint16 bits or a symbolic token


@dataclass(frozen=True)
class Policies:
    half_from_bits: Callable[[int], Half]
    half_add_rn: Callable[[Half, Half], Half]
    half_mul_rn: Callable[[Half, Half], Half]
    fp8_e4m3_rn_satfinite: Callable[[Half], int]
    # A[16][32] FP8 bytes, B[8][32] FP8 bytes in N,K order,
    # C[16][8] half +0 -> D[16][8] half. One call == one m16n8k32 instruction.
    mma_m16n8k32: Callable[[Sequence, Sequence, Sequence], Sequence]
    half_policy_name: str
    quantization_policy_name: str
    mma_policy_name: str

    def __post_init__(self):
        for name in ('half_from_bits','half_add_rn','half_mul_rn','fp8_e4m3_rn_satfinite','mma_m16n8k32'):
            if not callable(getattr(self,name)): raise TypeError(name+' must be explicitly supplied')
        for name in ('half_policy_name','quantization_policy_name','mma_policy_name'):
            if not getattr(self,name): raise ValueError(name+' must document the supplied policy')


def _matrix(x,rows,cols,name):
    if len(x)!=rows or any(len(row)!=cols for row in x):
        raise ValueError(f'{name}: expected {rows}x{cols}')


def _byte(x):
    if not isinstance(x,int) or isinstance(x,bool) or not 0<=x<=255:
        raise ValueError('FP8 policy must return one integer byte')
    return x


def main_pi(k: int) -> int:
    """Physical MMA K / packed low rank -> canonical MMA N-neuron.

    First16: 0,1,8,9,2,3,10,11,4,5,12,13,6,7,14,15.
    Applies independently per 16 channels, for both C32 and C64.
    """
    if k<0: raise ValueError('negative channel')
    return 16*(k//16)+8*((k//2)%2)+2*((k//4)%4)+k%2


def inverse_main_pi(c: int) -> int:
    """Canonical MMA N-neuron -> physical K / serialized low rank.

    This is the prior report's pi (inverseMainPI). First16:
    0,1,4,5,8,9,12,13,2,3,6,7,10,11,14,15.
    """
    if c<0: raise ValueError('negative channel')
    return 16*(c//16)+4*((c%8)//2)+2*((c//8)%2)+c%2


def basis_maps():
    return {
        'physical_k_to_canonical_input_c':tuple(main_pi(k) for k in range(32)),
        'canonical_input_c_to_physical_k':tuple(inverse_main_pi(c) for c in range(32)),
        'mma_output_n_to_canonical_output_n':tuple(range(64)),
        'mma_output_n_to_serialized_low_rank':tuple(inverse_main_pi(n) for n in range(64)),
        'serialized_low_rank_to_canonical_output_n':tuple(main_pi(r) for r in range(64)),
    }


def slab_offset(n: int,k: int) -> int:
    """Absolute block4 slab byte for W_N_K[n,k], not row-major.

    Output axis n=physical down MMA N-neuron, input axis k=physical MMA K.
    W_N_C[n,c] = slab[slab_offset(n,inverse_main_pi(c))].
    W_R_C[r,c] = slab[slab_offset(main_pi(r),inverse_main_pi(c))].
    W_R_K[r,k] = slab[slab_offset(main_pi(r),k)].
    These explicit both-axis views avoid assuming serialized rank == N.
    """
    if not (0<=n<64 and 0<=k<32): raise ValueError('down weight index')
    return (20656+512*(n//16)+16*(4*(n%8)+(k%16)//4)
            +8*((n//8)%2)+4*(k//16)+k%4)


def unpack_weights(slab: bytes):
    if len(slab)!=22720: raise ValueError('supply the exact 22720-byte block4 slab')
    return tuple(tuple(slab[slab_offset(n,k)] for k in range(32)) for n in range(64))


def weight_views(slab: bytes):
    nk=unpack_weights(slab)
    return {
        'N_K':nk,
        'N_canonicalC':tuple(tuple(nk[n][inverse_main_pi(c)] for c in range(32)) for n in range(64)),
        'serializedRank_canonicalC':tuple(tuple(nk[main_pi(r)][inverse_main_pi(c)] for c in range(32)) for r in range(64)),
        'serializedRank_physicalK':tuple(nk[main_pi(r)] for r in range(64)),
    }


def tile_rows_to_raster(tile_rows):
    """64x32 TL,TR,BL,BR four 4x4 tile rows -> 8x8 raster rows."""
    _matrix(tile_rows,64,32,'tile_rows')
    return tuple(tuple(tile_rows[16*(2*(y//4)+x//4)+4*(y%4)+x%4]) for y in range(8) for x in range(8))


def final_c_scalar_location(y,x,c):
    """Actual final C packed register/lane/half element for raster F[y,x,c]."""
    if not (0<=y<8 and 0<=x<8 and 0<=c<32): raise ValueError('final C index')
    m=4*(y%4)+x%4
    lane=4*(m%8)+(c%8)//2
    base=(2998 if y<4 else 4117)+8*(x//4)
    reg=base+4*(c//16)+(c//8)%2+2*(m//8)
    return {'lane':lane,'register':f'%r{reg}','half_element':c%2}


def unpack_final_c_fragments(lane_registers,half_from_bits):
    """32 dicts: actual %r2998..3013/%r4117..4132 -> packed uint32.

    Decodes half bits ONLY (never the high FP8 output) into raster64x32.
    Caller explicitly supplies the half representation policy.
    """
    if len(lane_registers)!=32: raise ValueError('one complete 32-lane warp required')
    out=[]
    for y in range(8):
        for x in range(8):
            row=[]
            for c in range(32):
                loc=final_c_scalar_location(y,x,c)
                word=lane_registers[loc['lane']][loc['register']]
                if not isinstance(word,int) or not 0<=word<2**32: raise ValueError('uint32 fragment required')
                row.append(half_from_bits((word>>(16*loc['half_element']))&0xffff))
            out.append(tuple(row))
    return tuple(out)


def pool_pair_coordinates(y,x):
    """Ordered source children at pooled LOCAL y/x, each in0..3.

    Horizontal pair order reverses for x>=2. First row reverses for odd y.
    Keep this even where finite half addition happens to be commutative.
    """
    if not (0<=y<4 and 0<=x<4): raise ValueError('pooled local coordinate')
    rho,sigma=y%2,x//2
    return ((2*y+rho,2*x+sigma),(2*y+rho,2*x+1-sigma),
            (2*y+1-rho,2*x+sigma),(2*y+1-rho,2*x+1-sigma))


def pooled_half(final_half,policies: Policies):
    """RN16(RN16(RN16(Fa+Fb)+RN16(Fc+Fd))*0.25).

    a,b,c,d are pool_pair_coordinates(y,x), NOT always TL,TR,BL,BR.
    Exact ordered children for ALL512 scalars are source selp/shfl/add/mul
    traces. Do not reassociate, divide early, use float32 pooling, or quantize
    F first. No hidden input-rounding step is inserted.
    """
    _matrix(final_half,64,32,'final_half raster')
    q=policies.half_from_bits(0x3400)  # +0.25 exactly; source f32 0x3e800000 -> RN half
    pooled=[]
    for y in range(4):
        for x in range(4):
            row=[]
            a,b,d,e=pool_pair_coordinates(y,x)
            for c in range(32):
                fa=final_half[a[0]*8+a[1]][c]
                fb=final_half[b[0]*8+b[1]][c]
                fd=final_half[d[0]*8+d[1]][c]
                fe=final_half[e[0]*8+e[1]][c]
                first_pair=policies.half_add_rn(fa,fb)
                second_pair=policies.half_add_rn(fd,fe)
                total=policies.half_add_rn(first_pair,second_pair)
                row.append(policies.half_mul_rn(total,q))
            pooled.append(tuple(row))
    return tuple(pooled)


@dataclass(frozen=True)
class DownsampleResult:
    pooled_half_canonicalC: tuple
    pooled_fp8_physicalK: tuple
    logical_D_mmaN_half: tuple
    logical_D_serializedRank_half: tuple
    low_fp8_mmaN: tuple
    low_packed_local_four_planes: bytes
    policies: tuple


def downsample(final_half,slab: bytes,policies: Policies) -> DownsampleResult:
    """64x32 final half -> 16x32 pool -> eight learned one-K32 -> 16x64.

    No complete block4 computation, no high-skip dependency, no padding writes.
    Actual slab is supplied, never automatically loaded or fitted.
    Native-equivalence status belongs to the supplied policies, not this API.
    """
    pool=pooled_half(final_half,policies)
    a=tuple(tuple(_byte(policies.fp8_e4m3_rn_satfinite(pool[m][main_pi(k)])) for k in range(32)) for m in range(16))
    w=unpack_weights(slab)
    zero=policies.half_from_bits(0x0000)
    c=tuple((zero,)*8 for _ in range(16))
    d=[[] for _ in range(16)]
    for group in range(8):
        dg=policies.mma_m16n8k32(a,w[8*group:8*group+8],c)
        _matrix(dg,16,8,'MMA D')
        for m in range(16): d[m].extend(dg[m])
    d=tuple(tuple(row) for row in d)
    qd=tuple(tuple(_byte(policies.fp8_e4m3_rn_satfinite(x)) for x in row) for row in d)
    packed=bytearray(1024)
    for m in range(16):
        for n in range(64):
            packed[16*(16*(n//16)+m)+inverse_main_pi(n)%16]=qd[m][n]
    return DownsampleResult(pool,a,d,tuple(tuple(row[main_pi(r)] for r in range(64)) for row in d),qd,bytes(packed),
                            (policies.half_policy_name,policies.quantization_policy_name,policies.mma_policy_name))


def _domain(H,W,X,Y,cx,cy):
    if min(H,W)<=0 or min(cx,cy)<0 or X%4 or Y%4:
        raise ValueError('requires positive H/W, nonnegative CTA xy, origins multiples4')
    # API domain is conservative: source integer wrap/negative-size launches not inferred.
    if max(H,W,abs(X)+8*cx,abs(Y)+8*cy)>=2**20 or 64*H*W>=2**31:
        raise ValueError('outside conservative no-overflow coordinate domain')


def low_writer_coordinates(H,W,X=0,Y=0,cx=0,cy=0):
    """Records each actual v2.b16 data store, in source instruction/lane order.

    Returns relative byte addresses only; no pointer/null/allocation assertion.
    Source line order 14137,14171,...14380; each data is four FP8 bytes.
    """
    _domain(H,W,X,Y,cx,cy)
    hd,wd=H//2,W//2; oy,ox=(Y+8*cy)//2,(X+8*cx)//2
    lines=(14137,14171,14205,14240,14275,14311,14345,14380)
    records=[]
    for plane in range(4):
        for d in range(2):
            for lane in range(32):
                m=lane//4+8*d; y=oy+m//4; x=ox+m%4
                ns=tuple(16*plane+8*u+2*(lane%4)+e for u in range(2) for e in range(2))
                valid=0<=y<hd and 0<=x<wd
                address=16*((plane*hd+y)*wd+x)+4*(lane%4)
                records.append({'line':lines[2*plane+d],'lane':lane,'m':m,'n':ns,'y':y,'x':x,
                                'valid':valid,'address':address if valid else None})
    return records


def low_global_stores(low_fp8_mmaN,H,W,X=0,Y=0,cx=0,cy=0):
    _matrix(low_fp8_mmaN,16,64,'low_fp8_mmaN')
    return tuple((r['address'],bytes(_byte(low_fp8_mmaN[r['m']][n]) for n in r['n']))
                 for r in low_writer_coordinates(H,W,X,Y,cx,cy) if r['valid'])


def high_skip_stores(final_half,quantize,H,W,X=0,Y=0,cx=0,cy=0):
    """Separate F -> Q8(F) high tiled-C fork. Never an input to downsample.

    Return individual byte records. Full4x4 tile guards, no high zero-fill.
    """
    _matrix(final_half,64,32,'final_half raster'); _domain(H,W,X,Y,cx,cy)
    oy,ox=Y+8*cy,X+8*cx; ty0,tx0=oy//4,ox//4
    writes=[]
    for y in range(8):
        for x in range(8):
            dy,dx=y//4,x//4; ty,tx=ty0+dy,tx0+dx
            valid=(oy>(-4 if dy==0 else -8) and ty<H//4 and
                   ox>(-4 if dx==0 else -8) and tx<W//4)
            if not valid: continue
            m=4*(y%4)+x%4
            for c in range(32):
                lane=4*(m%8)+(c%8)//2; j=2*(c//16)+m//8; u=(c//8)%2; e=c%2
                address=512*(ty*(W//4)+tx)+16*lane+4*j+2*u+e
                writes.append((address,_byte(quantize(final_half[y*8+x][c]))))
    return tuple(writes)


def padding_static_addresses(H,W,PH,PW,X=0,Y=0,grid_x=1,grid_y=1):
    """SEPARATE original padding address-set contract, NOT a data packer.

    Positive dimensions/padding/grid, origin multiples4, no overflow. Exact
    integer CFG is checked separately in source_suffix_proof/test_suffix.py.
    Eight16-byte planes, PH/PW strides, ceil(H/2),ceil(W/2) exclusion remain
    unchanged. This is a SET: does not impose CTA execution/publication order.
    The late source null check is not a learned-writer optional-output guard.
    No allocation exists here; these addresses cannot prove allocation OOB.
    """
    _domain(H,W,X,Y,grid_x-1,grid_y-1)
    if min(PH,PW,grid_x,grid_y)<=0 or max(PH,PW)>=2**20 or 128*PH*PW>=2**31:
        raise ValueError('padding/grid positive and conservatively bounded')
    hc,wc=(H+1)//2,(W+1)//2
    if PH<=hc and PW<=wc: return frozenset(),frozenset()
    pixels=set(); barriers=set()
    for cy in range(grid_y):
        for cx in range(grid_x):
            oy,ox=(Y+8*cy)//2,(X+8*cx)//2
            if (oy<PH and oy+4>hc) or (ox<PW and ox+4>wc):
                barriers.add((cx,cy))
                for y in range(oy,oy+4):
                    for x in range(ox,ox+4):
                        if 0<=y<PH and 0<=x<PW and (y>=hc or x>=wc): pixels.add((y,x))
    cover_y=min((Y+8*(grid_y-1))//2+4,PH)
    bottom_start=max(cover_y,hc)
    right_start=max(min((X+8*(grid_x-1))//2+4,PW),wc)
    pixels.update((y,x) for y in range(bottom_start,PH) for x in range(PW))
    pixels.update((y,x) for y in range(cover_y) for x in range(right_start,PW))
    addresses=frozenset(16*((p*PH+y)*PW+x)+b for p in range(8) for y,x in pixels for b in range(16))
    return addresses,frozenset(barriers)
