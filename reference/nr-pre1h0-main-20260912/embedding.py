"""Own pre0 16→32 F16 embedding: exactly one K16 RN-half dot, not post70's head.
Input is canonical 4x4-tile row order [64,16]. Native TC parity remains open.
"""
import numpy as np
from f16_math import finite_units,round_half
K=np.arange(16)[:,None];N=np.arange(32)[None,:]
WEIGHT_MAP=8208+512*(N//16)+16*(4*(N%8)+(K%8)//2)+8*((N%16)//8)+4*(K//8)+2*(K%2)

def forward(features,slab):
 if not isinstance(features,np.ndarray) or features.dtype!=np.float16 or features.shape!=(64,16) or not np.all(np.isfinite(features)):raise ValueError('finite canonical64x16 half features')
 if type(slab)is not bytes or len(slab)!=21696:raise ValueError('own block0 slab21696')
 wbits=np.frombuffer(slab,dtype='<u2')[WEIGHT_MAP//2].copy();a=np.array([finite_units(int(v)) for v in features.view(np.uint16).ravel()],dtype=object).reshape(64,16);b=np.array([finite_units(int(v)) for v in wbits.ravel()],dtype=object).reshape(16,32);acc=a@b;out=np.array([round_half(int(v)) for v in acc.ravel()],dtype=np.uint16).reshape(64,32)
 return dict(input_half=features.copy(),weight_half=wbits.view(np.float16),output_half=out.view(np.float16))
