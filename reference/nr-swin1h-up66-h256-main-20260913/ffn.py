"""Source-equation FFN candidate: expand, cubic, grouped projection, seeded mixing.
Consumes/produces canonical-C packed64x256 FP8. Separate original-operand acceptance.
"""
from pathlib import Path
import sys,importlib.util
import numpy as np
HERE=Path(__file__).resolve().parent;V=HERE.parent
sys.path.insert(0,str(V/'nr-vit-expand-tensor-reference-20260912'));import expand_tensor_reference as num
spec=importlib.util.spec_from_file_location('_swin8_ffn_axes',V/'nr-vit-contract-layout-main-20260912/tensor_reference.py');axes=importlib.util.module_from_spec(spec);spec.loader.exec_module(axes)
def pi(k):return 16*(k//16)+2*((k%16)//4)+8*((k%4)//2)+k%2

def activate(x):
 if x.dtype!=np.float16 or not np.all(np.isfinite(x)):raise num.UnsupportedNonFinite('finite half activation')
 q=np.minimum(x.astype(np.float64),4);q=np.maximum(q,-4)
 a,b,c=np.array([0xab28,0x3728,0x3b28],dtype=np.uint16).view(np.float16).astype(np.float64)
 t=num._rn_half(a*np.abs(q)+b,'cubic first FMA')
 z=num._rn_half(q*t.astype(np.float64)+c,'cubic second FMA')
 return num._rn_half(x.astype(np.float64)*z.astype(np.float64),'cubic final multiply')

# Own1h two-layer FFN: source four interleaved hidden32 expand/contract steps.
M=64;C=32;E=128
m=np.arange(M)[:,None];c=np.arange(C)[None,:]
C_MAP=axes.c_offset(m,c,C);A_MAP=axes.a_offset(m,c,C)
EXP_MAP=axes.b_offset(c.T,np.arange(E)[None,:],E)
CONTRACT_MAP=4096+axes.b_offset(np.arange(E)[:,None],c,C)
def forward(x,slab,hooks=None):
 if not isinstance(x,np.ndarray) or x.dtype!=np.float16 or x.shape!=(64,32) or not np.all(np.isfinite(x)):raise ValueError('finite UNQUANTIZED prefix64x32 half')
 if type(slab)is not bytes or len(slab)!=22784:raise ValueError('own1h22784byte slab')
 code=num.fp8_rn_satfinite(x);values=num.decode_e4m3(code.tobytes(),np.arange(2048).reshape(64,32));a=values[:,pi(np.arange(32))]
 we=num.decode_e4m3(slab,EXP_MAP);wc=num.decode_e4m3(slab,CONTRACT_MAP);gamma=np.frombuffer(slab,dtype='<f2',count=32,offset=10256).copy()
 if not np.all(np.isfinite(gamma)):raise num.UnsupportedNonFinite('FFN seed gamma')
 expanded=num._rn_half(a@we,'oneK32 expand');activated=activate(expanded);ac=num.fp8_rn_satfinite(activated);av=num.decode_e4m3(ac.tobytes(),np.arange(8192).reshape(64,128))
 seed=num._rn_half(x.astype(np.float64)*gamma.astype(np.float64),'unquantizedP earlyFFNseed');out=seed.copy();before={};after={}
 for k in (0,32,64,96):
  before[k]=out.copy();out=num._rn_half(av[:,pi(np.arange(k,k+32))]@wc[k:k+32]+out.astype(np.float64),'contract carried K32');after[k]=out.copy()
 physical=np.empty(2048,dtype=np.uint8);physical[C_MAP]=num.fp8_rn_satfinite(out)
 return dict(input_unquantized=x.copy(),input_fp8=code,input_mma_axes=a,weight_expand=we,weight_contract=wc,gamma=gamma,expanded_half=expanded,activation_half=activated,activation_fp8=ac,residual_seed_half=seed,contract_before=before,contract_after=after,prequantization=out,physical=physical)
