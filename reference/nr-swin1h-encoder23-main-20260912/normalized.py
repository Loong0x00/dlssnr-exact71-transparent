"""Bounded Swin8h Q/K half-tree normalization candidate; source acceptance separate.
No VM/DLL forward. Measured primitive LUT is explicit, pinned and half-domain-only.
"""
from pathlib import Path
import sys,hashlib
import numpy as np
HERE=Path(__file__).resolve().parent;V=HERE.parent
sys.path.insert(0,str(V/'nr-vit-expand-tensor-reference-20260912'));import expand_tensor_reference as num
LUT_SHA='a43128fa609f3f3a3e2538726fcf6feac5f950b9d5ef8eb523df3fbf43c96011'
class RsqrtHalfDomain:
 def __init__(self,data):
  if type(data)is not bytes or len(data)!=65536*4 or hashlib.sha256(data).hexdigest()!=LUT_SHA:raise ValueError('pinned measured RSQRT bytes required')
  self.table=np.frombuffer(data,dtype='<u4')
 @classmethod
 def load(cls):return cls((V/'nr-native-normalization-20260911/runs/20260911-134003/rsqrt-f32.bin').read_bytes())
 def evaluate(self,x):
  if x.dtype!=np.float16 or not np.all(np.isfinite(x)) or not np.all(x>0):raise ValueError('exact positive finite-half expansion domain only')
  bits=x.view(np.uint16);raw=self.table[bits].copy();values=raw.view(np.float32)
  half=num._rn_half(values.astype(np.float64),'measured rsqrt to half')
  return raw,half

def tree(x,rsqrt):
 rn=num._rn_half
 square=rn(x.astype(np.float64)**2,'norm squares')
 pa=rn(square[...,8:16].astype(np.float64)+square[...,24:32].astype(np.float64),'norm pair8,24')
 pb=rn(square[...,0:8].astype(np.float64)+square[...,16:24].astype(np.float64),'norm pair0,16')
 local=rn(pa.astype(np.float64)+pb.astype(np.float64),'norm local').reshape(64,1,4,2)
 b2=rn(local.astype(np.float64)+local[:,:,np.arange(4)^2,:].astype(np.float64),'norm butterfly2')
 b1=rn(b2.astype(np.float64)+b2[:,:,np.arange(4)^1,:].astype(np.float64),'norm butterfly1')
 den=rn(b1[...,0].astype(np.float64)+b1[...,1].astype(np.float64),'norm low plus high')
 epsilon=np.array([948045311],dtype=np.uint32).view(np.float32).astype(np.float16)[0]
 floor=np.maximum(den,epsilon).astype(np.float16)
 root32,root16=rsqrt.evaluate(floor)
 if not np.all(den.view(np.uint16)==den[...,0,None].view(np.uint16)):raise ValueError('unexpected lane-dependent norm tree')
 factor=root16[...,0]
 normalized=rn(x.astype(np.float64)*factor[...,None].astype(np.float64),'norm factor multiply')
 return dict(square=square,pair_a=pa,pair_b=pb,local=local,butterfly2=b2,butterfly1=b1,denominator=den,epsilon=epsilon,floored=floor,rsqrt_f32_bits=root32,factor_half=root16,normalized=normalized)

def forward(raw,slab,*,rsqrt,hooks=None):
 if not isinstance(raw,np.ndarray) or raw.dtype!=np.float16 or raw.shape!=(64,1,3,32) or not np.all(np.isfinite(raw)):raise ValueError('finite raw64x8x3x32 half input required')
 if type(slab)is not bytes or len(slab)!=20672 or not isinstance(rsqrt,RsqrtHalfDomain):raise ValueError('own slab and measured policy required')
 q=tree(raw[:,:,0,:],rsqrt);k=tree(raw[:,:,1,:],rsqrt)
 sf=np.frombuffer(slab,dtype='<f4',count=1,offset=19552).copy();s=num._rn_half(sf.astype(np.float64),'Q scale f32 to half')
 qh=num._rn_half(q['normalized'].astype(np.float64)*s[None,:,None].astype(np.float64),'Q scale after normalization')
 kh=k['normalized'];vh=raw[:,:,2,:].copy()
 result=dict(Q_tree=q,K_tree=k,Q_scale_f32=sf,Q_scale_half=s,Q_normalized_half=q['normalized'],Q_half=qh,K_half=kh,V_half=vh,Q=num.fp8_rn_satfinite(qh),K=num.fp8_rn_satfinite(kh),V=num.fp8_rn_satfinite(vh))
 for name,value in result.items():
  if isinstance(value,np.ndarray):num._call_hook(hooks,name,value)
 for name,st in [('Q',q),('K',k)]:
  for key,value in st.items():
   if isinstance(value,np.ndarray):num._call_hook(hooks,name+'.'+key,value)
 return result
