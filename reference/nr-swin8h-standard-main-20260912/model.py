"""Bounded ordinary8h NN API. Static expected block49 shift, not live instance state.
Source/weights/lifetimes outside this component are not silently inferred.
"""
from pathlib import Path
from types import MappingProxyType
from collections.abc import Mapping
import sys,importlib.util,struct,json,hashlib
import numpy as np
HERE=Path(__file__).resolve().parent

def load_engine():
 saved=dict(sys.modules);paths=sys.path[:]
 try:
  sys.path.insert(0,str(HERE))
  for name in ('ffn','qkv','normalized','attention'):
   spec=importlib.util.spec_from_file_location('_nr_std8_'+name,HERE/(name+'.py'));m=importlib.util.module_from_spec(spec);sys.modules[spec.name]=m;spec.loader.exec_module(m);sys.modules[name]=m
  spec=importlib.util.spec_from_file_location('_nr_std8_engine',HERE/'nn.py');m=importlib.util.module_from_spec(spec);sys.modules[spec.name]=m;spec.loader.exec_module(m);return m
 finally:
  for key in set(sys.modules)-set(saved):del sys.modules[key]
  sys.modules.update(saved);sys.path[:]=paths
engine=load_engine()
PINS={16:'37c5928b15aa456cac02de082e96cd2b7c3e512915f17914000cded6858a86d8',49:'e55e43e0c806c5119af85dff8e1e83835c0b6ddf4c3c16c2abe416d884f96120'}
class Standard8hReference:
 def __init__(self,slab,block,rsqrt,reciprocal,origin):
  if block not in PINS or type(slab)is not bytes or hashlib.sha256(slab).hexdigest()!=PINS[block]:raise ValueError('exact admitted named slab')
  if not isinstance(rsqrt,engine.normalized.RsqrtHalfDomain) or not isinstance(reciprocal,engine.attention.ReciprocalHalfDomain):raise ValueError('pinned policies')
  x,y=origin;engine.geometry(8,8,x,y)
  self._slab=slab;self.block=block;self.origin=(x,y);self._rsqrt=rsqrt;self._rcp=reciprocal
 @classmethod
 def from_checkpoint(cls,path,block=49,origin=None):
  if block not in PINS:raise ValueError('only source/weight-admitted blocks16/49')
  if origin is None:
   if block!=49:raise ValueError('explicit origin required; this block host configuration not proved here')
   origin=(-4,-4) # current-static factory table1, not a live capture
  with Path(path).open('rb') as f:
   h=hashlib.sha256()
   for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
   if h.hexdigest()!='890a53f05473dd6fd0980d1ca8501cfd84c1f6d7330b9531ca0981b6a6778d0e':raise ValueError('checkpoint identity')
   f.seek(0);n=struct.unpack('<Q',f.read(8))[0]
   if n>1024*1024:raise ValueError('bounded header')
   hd=json.loads(f.read(n));lo,hi=hd[f'block{block}.layer0.layer']['data_offsets'];f.seek(8+n+lo);slab=f.read(hi-lo)
  return cls(slab,block,engine.normalized.RsqrtHalfDomain.load(),engine.attention.ReciprocalHalfDomain.load(),origin)
 def forward_packed(self,features,*,h=8,w=8,keep_stages=False,hooks=None):
  if hooks is not None and not isinstance(hooks,Mapping):raise ValueError('named callback mapping')
  x,y=self.origin;result=engine.forward(features,self._slab,h=h,w=w,x=x,y=y,rsqrt=self._rsqrt,reciprocal=self._rcp,keep_stages=keep_stages or hooks is not None)
  if hooks:
   for (cx,cy),window in result['stages'].items():
    for part in ('ffn','raw','norm','attention'):
     for key,value in window[part].items():
      if isinstance(value,np.ndarray):engine.ffn.num._call_hook(hooks,f'cta{cx},{cy}.{part}.{key}',value)
  if not keep_stages:result.pop('stages',None)
  return result
 def forward(self,features,*,keep_stages=False,hooks=None):
  if not isinstance(features,np.ndarray) or features.dtype!=np.uint8 or features.ndim!=3 or features.shape[2]!=256:raise ValueError('canonical[H,W,256] E4M3uint8')
  h,w,_=features.shape;engine.geometry(h,w,*self.origin);yy,xx=np.meshgrid(np.arange(h),np.arange(w),indexing='ij');m=16*((yy//4)*(w//4)+xx//4)+4*(yy%4)+xx%4;mp=engine.ffn.axes.c_offset(m[:,:,None],np.arange(256)[None,None,:],256)
  packed=np.empty(h*w*256,dtype=np.uint8);packed[mp]=features;out=self.forward_packed(packed.tobytes(),h=h,w=w,keep_stages=keep_stages,hooks=hooks);out['canonical']=np.frombuffer(out['output'],dtype=np.uint8)[mp].copy();return out
 def weights(self):
  b=self._slab;f=engine.ffn;q=engine.qkv;a=engine.attention;d=f.num.decode_e4m3
  xs={'ffn.expand':d(b,f.EXP_MAP),'ffn.grouped':d(b,f.GROUP_MAP),'ffn.mix':d(b,f.MIX_MAP),'ffn.gamma':np.frombuffer(b,dtype='<f2',count=256,offset=360464),'qkv.projection':d(b,q.B_MAP),'qkv.Q_scale_f32':np.frombuffer(b,dtype='<f4',count=8,offset=623136),'attention.seed':np.frombuffer(b,dtype='<f2')[a.BIAS_MAP//2],'attention.final':d(b,a.B_MAP),'attention.gamma':np.frombuffer(b,dtype='<f2',count=256,offset=688704)}
  return MappingProxyType({k:f.num._readonly_snapshot(v) for k,v in xs.items()})
