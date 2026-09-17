"""Transparent pre0 first-counter/no-temporal/no-mask, normal-output CPU reference.
Sampler and nonnative noise policy are explicit. Not live image/native/fullNR parity.
Matrices exposed on canonical neuron input/output axes, not serialized-F16 guesses.
"""
from pathlib import Path
from types import MappingProxyType
from collections.abc import Mapping
import hashlib,json,struct
import numpy as np
import nn,ffn,qkv,normalized,attention,embedding
import preprocess_candidate as pre
SLAB_SHA='5fe2ab86289f7b21253d00368ce7a95f005ec3c8476abd7db22550b918962a5c'
CHECKPOINT_SHA='890a53f05473dd6fd0980d1ca8501cfd84c1f6d7330b9531ca0981b6a6778d0e'

def c_map(h,w):
 y,x=np.indices((h,w));m=16*((y//4)*(w//4)+x//4)+4*(y%4)+x%4
 return ffn.axes.c_offset(m[:,:,None],np.arange(32)[None,None,:],32)

class Preblock0FirstReference:
 def __init__(self,slab,*,approx,rsqrt,reciprocal):
  if type(slab)is not bytes or len(slab)!=21696 or hashlib.sha256(slab).hexdigest()!=SLAB_SHA:raise ValueError('own named block0 slab')
  if not isinstance(approx,pre.ApproxPolicy) or not isinstance(rsqrt,normalized.RsqrtHalfDomain) or not isinstance(reciprocal,attention.ReciprocalHalfDomain):raise ValueError('explicit ingress/learned arithmetic policies')
  self._slab=slab;self._approx=approx;self._rs=rsqrt;self._rc=reciprocal
 @classmethod
 def from_checkpoint(cls,path,*,approx):
  with Path(path).open('rb') as f:
   h=hashlib.sha256()
   for b in iter(lambda:f.read(1048576),b''):h.update(b)
   if h.hexdigest()!=CHECKPOINT_SHA:raise ValueError('checkpoint identity')
   f.seek(0);n=struct.unpack('<Q',f.read(8))[0]
   if not 0<n<1048576:raise ValueError('bounded header')
   head=json.loads(f.read(n));lo,hi=head['block0.layer0.layer']['data_offsets']
   if hi-lo!=21696:raise ValueError('named range')
   f.seek(8+n+lo);slab=f.read(hi-lo)
  return cls(slab,approx=approx,rsqrt=normalized.RsqrtHalfDomain.load(),reciprocal=attention.ReciprocalHalfDomain.load())
 def weights(self):
  b=self._slab;d=ffn.num.decode_e4m3;i32=np.argsort(ffn.pi(np.arange(32)));i128=np.argsort(ffn.pi(np.arange(128)));a={'embedding':np.frombuffer(b,dtype='<f2')[embedding.WEIGHT_MAP//2],'ffn.expand':d(b,ffn.EXP_MAP[i32]),'ffn.contract':d(b,ffn.CONTRACT_MAP[i128]),'ffn.seed_gamma':np.frombuffer(b,dtype='<f2',count=32,offset=9232),'qkv.projection':d(b,qkv.B_MAP[i32]),'qkv.Q_scale_f32':np.frombuffer(b,dtype='<f4',count=1,offset=20576),'attention.bias':np.frombuffer(b,dtype='<f2')[attention.BIAS_MAP[0]//2],'attention.final_projection':d(b,attention.B_MAP[i32]),'attention.seed_gamma':np.frombuffer(b,dtype='<f2',count=32,offset=21616)}
  return MappingProxyType({k:ffn.num._readonly_snapshot(v) for k,v in a.items()})
 def forward(self,args,*,sampler,hooks=None,keep_stages=False):
  if hooks is not None and not isinstance(hooks,Mapping):raise ValueError('named probe mapping')
  h,w=nn.geometry(args);r=nn.forward(args,self._slab,sampler=sampler,approx=self._approx,rsqrt=self._rs,reciprocal=self._rc,keep_stages=bool(hooks) or keep_stages);skip=r['skip_C_packed'][c_map(h,w)];low=r['primary_A_planar'].transpose(1,2,0,3).reshape(h//2,w//2,32)
  if hooks:
   for (cx,cy),z in r['stages'].items():
    for name,value in [('features',z['embedding']['input_half']),('embedding',z['embedding']['output_half']),('FFN_unquantized',z['ffn']['prequantization']),('Q',z['norm']['Q']),('K',z['norm']['K']),('V',z['norm']['V']),('probability',z['attention']['probability_half']),('PV',z['attention']['PV_half']),('final_unquantized',z['attention']['prequantization']),('low_unquantized',z['low_half_C'])]:ffn.num._call_hook(hooks,f'window.{cx},{cy}.{name}',value)
   ffn.num._call_hook(hooks,'skip_C',skip);ffn.num._call_hook(hooks,'primary_A',low)
  out=dict(primary_A=low,primary_A_planar=r['primary_A_planar'],skip_C=skip,skip_C_packed=r['skip_C_packed'],primary_port=0,skip_port=1,scope='firstcounter0 source callback, no-temporal/no-mask normal output family; no native allocation/sampler/visibility claim')
  if keep_stages:out['stages']=r['stages']
  return out
