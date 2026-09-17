"""Own typed32→16 computed F16 head (only4live); canonical finalC input.
ExactK16 integer sums with two RN-half boundaries; not native F16MMA equivalence.
"""
import numpy as np
from f16_math import finite_units,round_half
K=np.arange(32)[:,None];N=np.arange(16)[None,:]
WEIGHT_MAP=20784+512*(K//16)+16*(4*(N%8)+(K%8)//2)+8*(N//8)+4*((K%16)//8)+2*(K%2)

def forward(features,slab):
 if not isinstance(features,np.ndarray) or features.dtype!=np.float16 or features.shape!=(64,32) or not np.all(np.isfinite(features)):raise ValueError('finite canonicalfinal64x32 half')
 if type(slab)is not bytes or len(slab)!=21808:raise ValueError('own21808byte slab')
 weights=np.frombuffer(slab,dtype='<u2')[WEIGHT_MAP//2].copy();a=np.array([finite_units(int(v)) for v in features.view(np.uint16).ravel()],dtype=object).reshape(64,32);b=np.array([finite_units(int(v)) for v in weights.ravel()],dtype=object).reshape(32,16);c=np.zeros((64,16),dtype=np.uint16);before={};after={}
 for k in (0,16):
  before[k]=c.copy();units=np.array([finite_units(int(v))<<24 for v in c.ravel()],dtype=object).reshape(64,16);summed=a[:,k:k+16]@b[k:k+16]+units;c=np.array([round_half(int(v)) for v in summed.ravel()],dtype=np.uint16).reshape(64,16);after[k]=c.copy()
 return dict(weight_bits=weights,weight_half=weights.view(np.float16),before=before,after=after,computed_half=c.view(np.float16),live_half=c[:,:4].copy().view(np.float16))
