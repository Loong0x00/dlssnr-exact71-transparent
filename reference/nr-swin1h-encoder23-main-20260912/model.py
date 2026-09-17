"""Ordinary1h NN public references, bounded canonical uint8E4M3[64,64,32].
Current-static expected origins are defaults, NOT live mutable-instance guarantees.
"""
from pathlib import Path
from types import MappingProxyType
from collections.abc import Mapping
import hashlib,json,struct
import numpy as np
import nn,ffn,qkv,normalized,attention
CHECKPOINT_SHA='890a53f05473dd6fd0980d1ca8501cfd84c1f6d7330b9531ca0981b6a6778d0e'
ORIGINS={2:(-4,-4),3:(-4,0)}
PINS={2:'379da9fb2027273f2dffa52a970a3950e97ca2d720fdf089fe4a1af409bf81c8',3:'8f26be9b75d09960141bbf561a70e9628c269cc901934efdce362426ebbe1589'}
def c_map(h,w):
 y,x=np.meshgrid(np.arange(h),np.arange(w),indexing='ij');m=16*((y//4)*(w//4)+x//4)+4*(y%4)+x%4
 return ffn.axes.c_offset(m[:,:,None],np.arange(32)[None,None,:],32)
GLOBAL_C=c_map(64,64);LOCAL_C=c_map(8,8)
def load_slabs(path,blocks):
 with Path(path).open('rb') as f:
  digest=hashlib.sha256()
  for b in iter(lambda:f.read(1048576),b''):digest.update(b)
  if digest.hexdigest()!=CHECKPOINT_SHA:raise ValueError('whole checkpoint identity')
  f.seek(0);n=struct.unpack('<Q',f.read(8))[0]
  if n>1048576:raise ValueError('bounded header')
  hd=json.loads(f.read(n));slabs={}
  for i in blocks:
   lo,hi=hd[f'block{i}.layer0.layer']['data_offsets']
   if hi-lo!=20672:raise ValueError('ordinary1h named range')
   f.seek(8+n+lo);slabs[i]=f.read(hi-lo)
 return slabs
class Standard1hReference:
 def __init__(self,block,slab,rsqrt,reciprocal):
  if block not in PINS or type(slab)is not bytes or len(slab)!=20672 or hashlib.sha256(slab).hexdigest()!=PINS[block]:raise ValueError('original named ordinary1h block slab')
  if not isinstance(rsqrt,normalized.RsqrtHalfDomain) or not isinstance(reciprocal,attention.ReciprocalHalfDomain):raise ValueError('pinned measured half-domain LUT policies')
  self.block=block;self._slab=slab;self._rs=rsqrt;self._rc=reciprocal
 @classmethod
 def from_checkpoint(cls,path,*,block=2):return cls(block,load_slabs(path,[block])[block],normalized.RsqrtHalfDomain.load(),attention.ReciprocalHalfDomain.load())
 def weights(self):
  b=self._slab;d=ffn.num.decode_e4m3;i32=np.argsort(ffn.pi(np.arange(32)));i128=np.argsort(ffn.pi(np.arange(128)));a={'ffn.expand':d(b,ffn.EXP_MAP[i32]),'ffn.contract':d(b,ffn.CONTRACT_MAP[i128]),'ffn.seed_gamma':np.frombuffer(b,dtype='<f2',count=32,offset=8208),'qkv.projection':d(b,qkv.B_MAP[i32]),'qkv.Q_scale_f32':np.frombuffer(b,dtype='<f4',count=1,offset=19552),'attention.seed':np.frombuffer(b,dtype='<f2')[attention.BIAS_MAP//2],'attention.final_projection':d(b,attention.B_MAP[i32]),'attention.seed_gamma':np.frombuffer(b,dtype='<f2',count=32,offset=20592)}
  return MappingProxyType({k:ffn.num._readonly_snapshot(x) for k,x in a.items()})
 def forward_packed(self,features,*,origin=None,hooks=None,keep_stages=False):
  if hooks is not None and not isinstance(hooks,Mapping):raise ValueError('named hooks mapping')
  x,y=ORIGINS[self.block] if origin is None else origin
  r=nn.forward(features,self._slab,x=x,y=y,rsqrt=self._rs,reciprocal=self._rc,keep_stages=bool(hooks) or keep_stages)
  if hooks:
   for (cx,cy),z in r['stages'].items():
    for name,a in [('input',z['ingress'][LOCAL_C]),('ffn',z['ffn']['physical'][LOCAL_C]),('ffn_unquantized',z['ffn']['prequantization']),('Q',z['norm']['Q']),('K',z['norm']['K']),('V',z['norm']['V']),('probability',z['attention']['probability_half']),('PV',z['attention']['PV_half']),('output',z['physical'][LOCAL_C])]:ffn.num._call_hook(hooks,f'window.{cx},{cy}.{name}',a)
   ffn.num._call_hook(hooks,'output',np.frombuffer(r['output'],dtype=np.uint8)[GLOBAL_C])
  result=dict(output_packed=r['output'],origin=(x,y))
  if keep_stages:result['stages']=r['stages']
  return result
 def forward(self,features,**kwargs):
  if not isinstance(features,np.ndarray) or features.dtype!=np.uint8 or features.shape!=(64,64,32):raise ValueError('canonicalC uint8E4M3[64,64,32]')
  packed=np.empty(131072,dtype=np.uint8);packed[GLOBAL_C]=features;r=self.forward_packed(packed.tobytes(),**kwargs);r['output']=np.frombuffer(r['output_packed'],dtype=np.uint8)[GLOBAL_C].copy();return r
class Standard1hStack:
 def __init__(self,slabs):
  if set(slabs)!=set(ORIGINS):raise ValueError('two explicit blocks2/3 required')
  rs=normalized.RsqrtHalfDomain.load();rc=attention.ReciprocalHalfDomain.load();self.blocks=MappingProxyType({i:Standard1hReference(i,b,rs,rc) for i,b in slabs.items()})
 @classmethod
 def from_checkpoint(cls,path):return cls(load_slabs(path,ORIGINS))
 def forward_packed(self,features,*,hooks=None):
  outputs={}
  for i in ORIGINS:
   features=self.blocks[i].forward_packed(features)['output_packed'];outputs[i]=features
   if hooks:ffn.num._call_hook(hooks,f'block{i}.output',np.frombuffer(features,dtype=np.uint8)[GLOBAL_C])
  return dict(output_packed=features,block_outputs=outputs)
