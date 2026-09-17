"""Inspectable eight-block N64 ViT source-policy reference, not whole NR model.
Canonical feature API and optional exact 2D byte adapters. No DLL/VM forward.
"""
from pathlib import Path
from types import MappingProxyType
import importlib.util,sys,json,struct,hashlib
import numpy as np
HERE=Path(__file__).resolve().parent;V=HERE.parent

def module(name,path):
 spec=importlib.util.spec_from_file_location(name,V/path);m=importlib.util.module_from_spec(spec);sys.modules[name]=m;spec.loader.exec_module(m);return m
sys.path.insert(0,str(V/'nr-vit-n64-tensor-main-20260912'));import predict as core
out_repack=module('stack_output_repack','nr-vit-output-repack-main-20260912/repack.py')
in_repack=module('stack_input_repack','nr-vit-repack-reference-20260912/repack_2d_to_1d_fp8_reference.py')
CHECKPOINT_SHA='890a53f05473dd6fd0980d1ca8501cfd84c1f6d7330b9531ca0981b6a6778d0e'
SIZES=(4194320,4196352,3145856,2,1050624)

def scoped(hooks,prefix):return None if hooks is None else {k[len(prefix)+1:]:f for k,f in hooks.items() if k.startswith(prefix+'.')}
def unpack(data):
 if type(data)is not bytes or len(data)!=65536:raise ValueError('N64 physical feature span')
 return np.frombuffer(data,dtype=np.uint8)[core.projection.C_MAP].copy()
def pack(data):
 if not isinstance(data,np.ndarray) or data.dtype!=np.uint8 or data.shape!=(64,1024):raise ValueError('canonical uint8 E4M3[64,1024] required')
 if np.any((data&127)==127):raise ValueError('finite E4M3 codes required')
 out=np.empty(65536,dtype=np.uint8);out[core.projection.C_MAP]=data;return out.tobytes()

class ViTStack64:
 def __init__(self,blocks,ranges):
  self.blocks=MappingProxyType({i:tuple(blocks[i]) for i in range(31,39)});self.ranges=MappingProxyType(ranges)
  g=json.loads((V/'nr-outer-graph-main-20260912/RETYPED_FACTORY_GRAPH.json').read_text())['blocks']
  for i in range(31,39):
   if g[i]['primary_inputs']!=[dict(block=i-1,export=0,producer=dict(block=i-1,layer=4,output=0))]:raise ValueError('source-backed primary chain changed')
   for data,size in zip(self.blocks[i],SIZES):
    if type(data)is not bytes or len(data)!=size:raise ValueError('own operator slab shape')
 @classmethod
 def from_checkpoint(cls,path):
  path=Path(path);hash=hashlib.sha256()
  with path.open('rb') as f:
   for chunk in iter(lambda:f.read(1048576),b''):hash.update(chunk)
  if hash.hexdigest()!=CHECKPOINT_SHA:raise ValueError('original pinned checkpoint required')
  blocks={};ranges={}
  with path.open('rb') as f:
   n=struct.unpack('<Q',f.read(8))[0]
   if n>1048576:raise ValueError('bounded checkpoint header')
   header=json.loads(f.read(n))
   for i in range(31,39):
    blocks[i]=[]
    for j,size in enumerate(SIZES):
     name=f'block{i}.layer{j}.layer';a,b=header[name]['data_offsets']
     if a<0 or b-a!=size:raise ValueError('named tensor range')
     f.seek(8+n+a);data=f.read(size);blocks[i].append(data);ranges[name]=dict(offset=8+n+a,bytes=size,sha256=hashlib.sha256(data).hexdigest())
  return cls(blocks,ranges)
 def forward(self,features,hooks=None,keep_stages=False):
  physical=pack(features);records=[];kept={}
  for i in range(31,39):
   slabs=self.blocks[i];prefix=f'block{i}';before=physical
   e=core.expand.forward(before,slabs[0],scoped(hooks,prefix+'.expand'));F=e['output_physical'].tobytes()
   c=core.contract.forward(F,before,slabs[1],scoped(hooks,prefix+'.contract'));P=c['output_physical'].tobytes()
   q=core.qkv.forward(P,slabs[2],scoped(hooks,prefix+'.qkv'))
   a=core.attention.forward(*(q[n+'_physical'].tobytes() for n in ('Q','K','V')),scoped(hooks,prefix+'.attention'))
   y=core.projection.forward(a['output_physical'].tobytes(),P,slabs[4],scoped(hooks,prefix+'.projection'));physical=y['output_physical'].tobytes()
   records.append(dict(block=i,input_sha256=hashlib.sha256(before).hexdigest(),output_sha256=hashlib.sha256(physical).hexdigest(),bytes=len(physical)))
   if keep_stages:kept[i]=dict(expand=e,contract=c,qkv=q,attention=a,projection=y)
  result=dict(output_fp8=unpack(physical),output_1d_physical=physical,output_2d_physical=out_repack.forward(physical,8,8),blocks=records)
  if keep_stages:result['stages']=kept
  return result
 def forward_head_2d(self,head_bytes,**kwargs):
  if type(head_bytes)is not bytes or len(head_bytes)!=65536:raise ValueError('complete8x8x1024 head byte span required')
  # Actual block31 source-defined input-repack, then block38 output-repack.
  return self.forward(unpack(bytes(in_repack.run_repack(head_bytes,8,8))),**kwargs)
