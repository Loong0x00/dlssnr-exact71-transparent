"""Transparent BASE INPVIEW reference: logical canonical-C neurons or packed planar-A.
Eight immutable canonical weights; explicit half-domain policies. Not tilesync/native.
"""
from pathlib import Path
from types import MappingProxyType
from collections.abc import Mapping
import hashlib,json,struct
import numpy as np
import nn,ffn,qkv,normalized,attention
SLAB_SHA='0270fb8f406d3d0a916953ddaf6ec5d29328fe95b9d6cb408ca0c25c91108e3a'
CHECKPOINT_SHA='890a53f05473dd6fd0980d1ca8501cfd84c1f6d7330b9531ca0981b6a6778d0e'
def c_map(h,w):
 y,x=np.indices((h,w));m=16*((y//4)*(w//4)+x//4)+4*(y%4)+x%4
 return ffn.axes.c_offset(m[:,:,None],np.arange(32)[None,None,:],32)
LOCAL_C=c_map(8,8)
class Inpview1hReference:
 def __init__(self,slab,*,rsqrt,reciprocal,variant='base'):
  if variant!='base':raise ValueError('tilesync publication requires independent synchronization contract')
  if type(slab)is not bytes or len(slab)!=20672 or hashlib.sha256(slab).hexdigest()!=SLAB_SHA:raise ValueError('own named block1 slab')
  if not isinstance(rsqrt,normalized.RsqrtHalfDomain) or not isinstance(reciprocal,attention.ReciprocalHalfDomain):raise ValueError('explicit pinned measured half-domain policies')
  self._slab=slab;self._rs=rsqrt;self._rc=reciprocal
 @classmethod
 def from_checkpoint(cls,path):
  with Path(path).open('rb') as f:
   if hashlib.file_digest(f,'sha256').hexdigest()!=CHECKPOINT_SHA:raise ValueError('whole checkpoint identity')
   f.seek(0);n=struct.unpack('<Q',f.read(8))[0]
   if not 0<n<1048576:raise ValueError('bounded header')
   hd=json.loads(f.read(n));lo,hi=hd['block1.layer0.layer']['data_offsets']
   if [lo,hi]!=[21696,42368]:raise ValueError('named block1 range')
   f.seek(8+n+lo);b=f.read(hi-lo)
  return cls(b,rsqrt=normalized.RsqrtHalfDomain.load(),reciprocal=attention.ReciprocalHalfDomain.load())
 def weights(self):
  b=self._slab;d=ffn.num.decode_e4m3;i32=np.argsort(ffn.pi(np.arange(32)));i128=np.argsort(ffn.pi(np.arange(128)));a={'ffn.expand':d(b,ffn.EXP_MAP[i32]),'ffn.contract':d(b,ffn.CONTRACT_MAP[i128]),'ffn.seed_gamma':np.frombuffer(b,dtype='<f2',count=32,offset=8208),'qkv.projection':d(b,qkv.B_MAP[i32]),'qkv.Q_scale_f32':np.frombuffer(b,dtype='<f4',count=1,offset=19552),'attention.bias':np.frombuffer(b,dtype='<f2')[attention.BIAS_MAP[0]//2],'attention.final_projection':d(b,attention.B_MAP[i32]),'attention.seed_gamma':np.frombuffer(b,dtype='<f2',count=32,offset=20592)}
  return MappingProxyType({k:ffn.num._readonly_snapshot(v) for k,v in a.items()})
 def forward_packed(self,features,*,h=64,w=64,origin=(0,0),source_extent=(0,0),hooks=None):
  if hooks is not None and not isinstance(hooks,Mapping):raise ValueError('named immutable probes mapping')
  if len(origin)!=2 or len(source_extent)!=2 or any(type(v)is not int or not -2**31<=v<2**31 for v in source_extent):raise ValueError('signed source extent fields')
  x,y=origin;nn.standard_nn.geometry(h,w,x,y);hs,ws=nn.source_shape(h,w,source_extent)
  if type(features)is not bytes or len(features)!=hs*ws*32:raise ValueError('exact planarA source span')
  if np.any((np.frombuffer(features,np.uint8)&127)==127):raise ValueError('finite E4M3 domain only')
  r=nn.forward(features,self._slab,h=h,w=w,x=x,y=y,source_extent=source_extent,rsqrt=self._rs,reciprocal=self._rc,keep_stages=bool(hooks));output=np.frombuffer(r['output'],np.uint8)[c_map(h,w)]
  if hooks:
   for (cx,cy),z in r['stages'].items():
    for name,value in [('input',z['ingress'][LOCAL_C]),('FFN_unquantized',z['ffn']['prequantization']),('Q',z['norm']['Q']),('K',z['norm']['K']),('V',z['norm']['V']),('probability',z['attention']['probability_half']),('PV',z['attention']['PV_half']),('final_unquantized',z['attention']['prequantization']),('output',z['physical'][LOCAL_C])]:ffn.num._call_hook(hooks,f'window.{cx},{cy}.{name}',value)
   ffn.num._call_hook(hooks,'output',output)
  return dict(output=output,output_packed=r['output'],source_extent=(hs,ws),origin=(x,y),variant='base')
 def forward(self,features,*,h=64,w=64,origin=(0,0),hooks=None):
  """Canonical logical uint8 E4M3[Hs,Ws,32], not physical-A channel order."""
  if not isinstance(features,np.ndarray) or features.dtype!=np.uint8 or features.ndim!=3 or features.shape[2]!=32:raise ValueError('canonicalC uint8[Hs,Ws,32]')
  hs,ws,_=features.shape
  if not 1<=hs<=64 or not 1<=ws<=64:raise ValueError('bounded source dimensions')
  physical=features[:,:,ffn.pi(np.arange(32))].reshape(hs,ws,2,16).transpose(2,0,1,3).copy().tobytes();extent=(0,0) if (hs,ws)==(h,w) else (hs,ws)
  return self.forward_packed(physical,h=h,w=w,origin=origin,source_extent=extent,hooks=hooks)
