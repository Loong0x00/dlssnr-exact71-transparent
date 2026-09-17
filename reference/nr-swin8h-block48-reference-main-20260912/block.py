"""Complete bounded block48 NN reference, NOT complete NR/native equivalence.
Canonical inputs: uint8 E4M3[4,4,512] and skip[8,8,256]; output[8,8,256].
All forward operations are tensor math; local helper imports remain a packaging gap.
"""
from pathlib import Path
from collections.abc import Mapping
from types import MappingProxyType
import importlib.util,sys,hashlib,struct,json
import numpy as np
HERE=Path(__file__).resolve().parent;V=HERE.parent

def load(name,path):
 spec=importlib.util.spec_from_file_location('_nr_b48_'+name,path);m=importlib.util.module_from_spec(spec);sys.modules[spec.name]=m;spec.loader.exec_module(m);return m
up=load('upsample',V/'nr-swin8h-up-main-20260912/prefix_tensor.py')
ffn=load('ffn',V/'nr-swin8h-ffn-tensor-main-20260912/ffn.py')
raw=load('raw',V/'nr-swin8h-qkv-tensor-main-20260912/qkv.py')
norm=load('norm',V/'nr-swin8h-qkv-normalized-main-20260912/normalized.py')
attn=load('attention',V/'nr-swin8h-attention-tensor-main-20260912/attention.py')
SLAB_SHA='7e832b24266b5565b4660a32a4789ce3b15d5835e01df4872e0cee52f82dfcb7'
CHECKPOINT_SHA='890a53f05473dd6fd0980d1ca8501cfd84c1f6d7330b9531ca0981b6a6778d0e'
class Block48Reference:
 def __init__(self,slab,rsqrt,reciprocal):
  if type(slab)is not bytes or len(slab)!=820784 or hashlib.sha256(slab).hexdigest()!=SLAB_SHA:raise ValueError('original named block48 slab required')
  if not isinstance(rsqrt,norm.RsqrtHalfDomain) or not isinstance(reciprocal,attn.ReciprocalHalfDomain):raise ValueError('pinned half-domain policies required')
  self._slab=slab;self._rsqrt=rsqrt;self._reciprocal=reciprocal
 @classmethod
 def from_checkpoint(cls,path):
  with Path(path).open('rb') as f:
   digest=hashlib.sha256()
   for b in iter(lambda:f.read(1024*1024),b''):digest.update(b)
   if digest.hexdigest()!=CHECKPOINT_SHA:raise ValueError('checkpoint identity')
   f.seek(0);size=struct.unpack('<Q',f.read(8))[0]
   if size>1024*1024:raise ValueError('bounded header')
   header=json.loads(f.read(size));lo,hi=header['block48.layer0.layer']['data_offsets'];f.seek(8+size+lo);slab=f.read(hi-lo)
  return cls(slab,norm.RsqrtHalfDomain.load(),attn.ReciprocalHalfDomain.load())
 @staticmethod
 def _hooks(hooks,prefix):
  if hooks is None:return None
  if not isinstance(hooks,Mapping):raise ValueError('mapping of named callbacks required')
  return {k[len(prefix)+1:]:v for k,v in hooks.items() if isinstance(k,str) and k.startswith(prefix+'.')}
 def weights(self):
  b=self._slab;decode=up.num.decode_e4m3
  arrays={'upsample.projection':decode(b,up.B_MAP),'upsample.skip_gamma':np.frombuffer(b,dtype='<f2',count=256,offset=492032),
   'ffn.expand':decode(b,ffn.EXP_MAP),'ffn.grouped':decode(b,ffn.GROUP_MAP),'ffn.mix':decode(b,ffn.MIX_MAP),'ffn.seed_gamma':np.frombuffer(b,dtype='<f2',count=256,offset=491520),
   'qkv.projection':decode(b,raw.B_MAP),'qkv.Q_scale_f32':np.frombuffer(b,dtype='<f4',count=8,offset=754688),
   'attention.seed':np.frombuffer(b,dtype='<f2')[attn.BIAS_MAP//2],'attention.final_projection':decode(b,attn.B_MAP),'attention.seed_gamma':np.frombuffer(b,dtype='<f2',count=256,offset=820256)}
  return MappingProxyType({k:up.num._readonly_snapshot(v) for k,v in arrays.items()})
 def forward_packed(self,low,skip,*,hooks=None,keep_stages=False):
  """P0 channel16-planar bytes, P24 canonical-C tile bytes; I/O confined to loading/construction."""
  p=up.forward(low,skip,self._slab,hooks=self._hooks(hooks,'upsample'))
  f=ffn.forward(p['physical'].tobytes(),self._slab,hooks=self._hooks(hooks,'ffn'))
  r=raw.forward_raw(f['physical'].tobytes(),self._slab,hooks=self._hooks(hooks,'qkv_raw'))
  n=norm.forward(r['warp_bank_view'],self._slab,rsqrt=self._rsqrt,hooks=self._hooks(hooks,'qkv_norm'))
  a=attn.forward(n['Q'],n['K'],n['V'],f['physical'].tobytes(),self._slab,reciprocal=self._reciprocal,hooks=self._hooks(hooks,'attention'))
  result={'output_packed':a['physical'].tobytes()}
  if keep_stages:result['stages']={'upsample':p,'ffn':f,'qkv_raw':r,'qkv_norm':n,'attention':a}
  return result
 def forward(self,features,skip,*,hooks=None,keep_stages=False):
  for x,shape in ((features,(4,4,512)),(skip,(8,8,256))):
   if not isinstance(x,np.ndarray) or x.dtype!=np.uint8 or x.shape!=shape:raise ValueError('canonical E4M3 uint8 tensor geometry')
  low=np.ascontiguousarray(features.reshape(4,4,32,16).transpose(2,0,1,3)).tobytes()
  packed=np.empty(16384,dtype=np.uint8);packed[up.C_MAP]=skip
  result=self.forward_packed(low,packed.tobytes(),hooks=hooks,keep_stages=keep_stages)
  result['output']=np.frombuffer(result['output_packed'],dtype=np.uint8)[up.C_MAP].copy()
  return result
