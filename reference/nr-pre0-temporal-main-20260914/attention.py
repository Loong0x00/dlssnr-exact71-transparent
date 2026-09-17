"""Source-equation candidate: seeded QK, packed half map, pre-PV normalization,
two-K32 PV and early-residual-seeded final projection. Acceptance is separate.
"""
from pathlib import Path
import hashlib
import numpy as np
import numeric_primitives as num
import layout_axes as axes
HERE=Path(__file__).resolve().parent
def pi(x):return 16*(x//16)+2*((x%16)//4)+8*((x%4)//2)+x%2
m=np.arange(64)[:,None];c=np.arange(32)[None,:]
A_MAP=axes.a_offset(m,c,32);C_MAP=axes.c_offset(m,c,32);B_MAP=20592+axes.b_offset(c.T,c,32)
h=np.arange(1)[:,None,None];q=np.arange(64)[None,:,None];k=np.arange(64)[None,None,:]
BIAS_MAP=12384+8192*h+512*(4*(q//16)+k//16)+16*(4*(q%8)+(k%8)//2)+8*((k%16)//8)+4*((q%16)//8)+2*(k%2)
class ReciprocalHalfDomain:
 def __init__(self,b):
  if type(b)is not bytes or len(b)!=262144 or hashlib.sha256(b).hexdigest()!='1254eed4e932af3deb7894a185a159e4b95f863aa353afbabc603a3ba2abd53b':raise ValueError('pinned reciprocal measurement required')
  self.table=np.frombuffer(b,dtype='<u4')
 @classmethod
 def load(cls):return cls((HERE/'reciprocal-f32.bin').read_bytes())
 def evaluate(self,x):
  if x.dtype!=np.float16:raise ValueError('half normalizer required')
  bits=x.view(np.uint16)
  if not np.all((bits>=0x410)&(bits<0x7c00)):raise ValueError('reciprocal outside measured normalizer domain')
  raw=self.table[bits].copy();half=num._rn_half(raw.view(np.float32).astype(np.float64),'measured reciprocal to half')
  return raw,half

def forward(Q,K,Vv,skip,slab,*,reciprocal,hooks=None):
 for a in (Q,K,Vv):
  if not isinstance(a,np.ndarray) or a.dtype!=np.uint8 or a.shape!=(64,1,32):raise ValueError('normalized logical64x1x32 FP8 arrays required')
 if not isinstance(skip,np.ndarray) or skip.dtype!=np.float16 or skip.shape!=(64,32) or type(slab)is not bytes or len(slab)!=21696 or not isinstance(reciprocal,ReciprocalHalfDomain):raise ValueError('own skip/slab/policy required')
 decode=lambda x:num.decode_e4m3(x.tobytes(),np.arange(2048).reshape(64,1,32)).transpose(1,0,2)
 a,b,v=map(decode,(Q,K,Vv));bias=np.frombuffer(slab,dtype='<f2')[BIAS_MAP//2].copy()
 if not np.all(np.isfinite(bias)):raise num.UnsupportedNonFinite('finite QK seeds')
 idx=pi(np.arange(32));score=np.stack([num._rn_half(a[h][:,idx]@b[h][:,idx].T+bias[h].astype(np.float64),'seeded QK K32') for h in range(1)])
 constants=np.array([1027077105,1067877303,1065615360,1070129152],dtype=np.uint32).view(np.float32).astype(np.float16)
 affine=num._rn_half(score.astype(np.float64)*float(constants[0])+float(constants[1]),'attention affine FMA')
 floor=np.maximum(affine,constants[2]);clamp=np.minimum(floor,constants[3]).astype(np.float16)
 pairs=clamp.view(np.uint16).reshape(1,64,32,2).astype(np.uint32);packed=pairs[...,0]|(pairs[...,1]<<16)
 shifted=(packed.astype(np.uint64)<<5)&0xffffffff;word=((shifted+2146992128)&0xffffffff).astype(np.uint32)
 mapped_bits=np.stack([word&65535,word>>16],axis=-1).astype(np.uint16).reshape(1,64,64);mapped=mapped_bits.view(np.float16)
 if not np.all(np.isfinite(mapped)):raise num.UnsupportedNonFinite('packed map produced nonfinite')
 add=lambda x,y,desc:num._rn_half(x.astype(np.float64)+y.astype(np.float64),desc)
 pair=[add(mapped[...,i:i+8],mapped[...,i+8:i+16],'mapped local pair') for i in (0,16,32,48)]
 local01=add(pair[0],pair[1],'mapped local01');local012=add(local01,pair[2],'mapped local012');local=add(local012,pair[3],'mapped local0123').reshape(1,64,4,2)
 lanes01=add(local[...,0,:],local[...,1,:],'den lanes01');lanes012=add(lanes01,local[...,2,:],'den lanes012');lanes=add(lanes012,local[...,3,:],'den lanes0123')
 denominator=add(lanes[...,0],lanes[...,1],'den low high')
 eps=np.array([948045311],dtype=np.uint32).view(np.float32).astype(np.float16)[0];den_floor=np.maximum(denominator,eps).astype(np.float16)
 reciprocal32,inverse=reciprocal.evaluate(den_floor)
 prob=num._rn_half(mapped.astype(np.float64)*inverse[...,None].astype(np.float64),'normalize before probability FP8');pcodes=num.fp8_rn_satfinite(prob)
 pv_input=num.decode_e4m3(pcodes.tobytes(),np.arange(4096).reshape(1,64,64));pv=np.zeros((1,64,32),dtype=np.float16);pv_before={};pv_after={}
 for k0 in (0,32):
  pv_before[k0]=pv.copy();ii=pi(np.arange(k0,k0+32))
  pv=np.stack([num._rn_half(pv_input[h][:,ii]@v[h][ii]+pv[h].astype(np.float64),'PV perK32') for h in range(1)]);pv_after[k0]=pv.copy()
 flat=pv.transpose(1,0,2).reshape(64,32);pv_codes=num.fp8_rn_satfinite(flat);pv_phys=np.empty(2048,dtype=np.uint8);pv_phys[C_MAP]=pv_codes
 ma=num.decode_e4m3(pv_phys.tobytes(),A_MAP);weight=num.decode_e4m3(slab,B_MAP);residual=skip.astype(np.float64);gamma=np.frombuffer(slab,dtype='<f2',count=32,offset=21616).copy()
 if not np.all(np.isfinite(gamma)):raise num.UnsupportedNonFinite('finite final gamma')
 seed=num._rn_half(residual*gamma.astype(np.float64),'final early seed');out=seed.copy();fb={};fa={}
 for k0 in range(0,32,32):
  fb[k0]=out.copy();out=num._rn_half(ma[:,k0:k0+32]@weight[k0:k0+32]+out.astype(np.float64),'final seeded K32');fa[k0]=out.copy()
 result=dict(Q_mma_axes=a[:,:,idx],K_mma_axes=b[:,:,idx],V_axes=v,bias=bias,score=score,constants=constants,affine=affine,floor=floor,clamp=clamp,packed=packed,shifted=shifted.astype(np.uint32),mapped_words=word,mapped=mapped,local_pairs=pair,local01=local01,local012=local012,local=local,lanes01=lanes01,lanes012=lanes012,lanes=lanes,denominator=denominator,den_floor=den_floor,reciprocal_f32_bits=reciprocal32,inverse_half=inverse,probability_half=prob,probability_fp8=pcodes,PV_half=pv,PV_before=pv_before,PV_after=pv_after,PV_physical=pv_phys,final_A=ma,final_B=weight,residual=residual,gamma=gamma,seed=seed,final_before=fb,final_after=fa,prequantization=out)
 for key,value in result.items():
  if isinstance(value,np.ndarray):num._call_hook(hooks,key,value)
 return result
