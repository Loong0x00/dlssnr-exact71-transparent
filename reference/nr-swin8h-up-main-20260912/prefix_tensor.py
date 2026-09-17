"""SOURCE-derived candidate, bounded8x8 only. Not yet the whole Swin NN lift.
P0 is channel16-planar4x4x512; P24 is tile-packed8x8x256.
Extra512->256 perK32 half projection, unquantized nearest2x, late learned skip.
"""
from pathlib import Path
import sys,importlib.util
import numpy as np
HERE=Path(__file__).resolve().parent;V=HERE.parent
sys.path.insert(0,str(V/'nr-vit-expand-tensor-reference-20260912'));import expand_tensor_reference as num
spec=importlib.util.spec_from_file_location('_up8_axes',V/'nr-vit-contract-layout-main-20260912/tensor_reference.py');axes=importlib.util.module_from_spec(spec);spec.loader.exec_module(axes)
m=np.arange(16)[:,None];k=np.arange(512)[None,:];n=np.arange(256)[None,:]
A_MAP=16*((k//16*4+m//4)*4+m%4)+k%16
B_MAP=360448+axes.b_offset(k.T,n,256)
y,x=np.meshgrid(np.arange(8),np.arange(8),indexing='ij');tile=16*((y//4)*2+x//4)+4*(y%4)+x%4
C_MAP=axes.c_offset(tile[:,:,None],np.arange(256)[None,None,:],256)
def forward(source,skip,slab,hooks=None):
 for b,nbytes in ((source,8192),(skip,16384),(slab,820784)):
  if type(b)is not bytes or len(b)!=nbytes:raise ValueError('prefix exact8x8 source/skip/slab spans')
 a=num.decode_e4m3(source,A_MAP);b=num.decode_e4m3(slab,B_MAP);s=num.decode_e4m3(skip,C_MAP)
 gamma=np.frombuffer(slab,dtype='<f2',count=256,offset=492032).copy()
 if not np.all(np.isfinite(gamma)):raise num.UnsupportedNonFinite('prefix finite gamma')
 carry=np.zeros((16,256),dtype=np.float16);before={};after={}
 for k0 in range(0,512,32):
  before[k0]=carry.copy();carry=num._rn_half(a[:,k0:k0+32]@b[k0:k0+32]+carry.astype(np.float64),'prefix perK32');after[k0]=carry.copy()
  num._call_hook(hooks,f'after_K{k0+32}',carry)
 low=carry.reshape(4,4,256);near=low.repeat(2,0).repeat(2,1)
 scaled=num._rn_half(s*gamma.astype(np.float64),'prefix late skip multiply')
 pre=num._rn_half(near.astype(np.float64)+scaled.astype(np.float64),'prefix late skip add')
 out=num.fp8_rn_satfinite(pre);physical=np.empty(16384,dtype=np.uint8);physical[C_MAP]=out
 result=dict(input_mma_axes=a,weight_mma_axes=b,skip=s,gamma=gamma,low=low,nearest=near,scaled_skip=scaled,prequantization=pre,output=out,physical=physical,before=before,after=after)
 for name in ('input_mma_axes','weight_mma_axes','skip','gamma','low','nearest','scaled_skip','prequantization','output'):num._call_hook(hooks,name,result[name])
 return result
