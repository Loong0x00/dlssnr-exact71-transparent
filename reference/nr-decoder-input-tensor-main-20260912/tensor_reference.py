"""Source-equation candidate: low-res1x1 projection,2x nearest view, late skip.
Acceptance requires the separate actual operand/intermediate checker.
"""
from pathlib import Path
import sys,importlib.util,json,hashlib
import numpy as np
HERE=Path(__file__).resolve().parent;V=HERE.parent
sys.path.insert(0,str(V/'nr-vit-expand-tensor-reference-20260912'));import expand_tensor_reference as num
spec=importlib.util.spec_from_file_location('decoder_axes',V/'nr-vit-contract-layout-main-20260912/tensor_reference.py');axes=importlib.util.module_from_spec(spec);spec.loader.exec_module(axes)
H=W=8;HO=WO=16;K=1024;N=512

def tile_m(y,x,width):return 16*((y//4)*(width//4)+x//4)+4*(y%4)+x%4
im=np.arange(64)[:,None];ik=np.arange(1024)[None,:];oc=np.arange(512)[None,:]
A_MAP=axes.a_offset(im,ik,1024);B_MAP=axes.b_offset(ik.T,oc,512);C_MAP=axes.c_offset(np.arange(256)[:,None],oc,512)
y,x=np.meshgrid(np.arange(16),np.arange(16),indexing='ij');FULL_M=tile_m(y,x,16);LOW_M=tile_m(y//2,x//2,8)

def forward(source,skip,slab,hooks=None):
 for b,n in ((source,65536),(skip,131072),(slab,525312)):
  if type(b)is not bytes or len(b)!=n:raise ValueError('own decoder input/skip/slab span')
 a=num.decode_e4m3(source,A_MAP);weight=num.decode_e4m3(slab,B_MAP);residual=num.decode_e4m3(skip,C_MAP);coefficient=np.frombuffer(slab,dtype='<f2',offset=524288).copy()
 if not np.all(np.isfinite(coefficient)):raise num.UnsupportedNonFinite('decoder finite coefficients')
 for key,value in [('input_mma_axes',a),('weight_mma_axes',weight),('skip_output_axes',residual),('residual_coefficient',coefficient)]:num._call_hook(hooks,key,value)
 parts=[];combined=None
 for z in range(4):
  carry=np.zeros((64,512),dtype=np.float16)
  for k0 in range(z*256,(z+1)*256,32):carry=num._rn_half(a[:,k0:k0+32]@weight[k0:k0+32]+carry.astype(np.float64),'decoder perK32')
  parts.append(carry);num._call_hook(hooks,f'partial_z{z}',carry)
  combined=carry.copy() if combined is None else num._rn_half(combined.astype(np.float64)+carry.astype(np.float64),'decoder ordered half split merge')
 up=np.empty((256,512),dtype=np.float16);up[FULL_M]=combined[LOW_M]
 scaled=num._rn_half(residual*coefficient.astype(np.float64),'decoder late learned skip multiply')
 pre=num._rn_half(up.astype(np.float64)+scaled.astype(np.float64),'decoder late skip add')
 out=num.fp8_rn_satfinite(pre);physical=np.empty(131072,dtype=np.uint8);physical[C_MAP]=out
 for key,value in [('low_half',combined),('upsampled_half',up),('scaled_skip_half',scaled),('prequantization',pre),('output',out)]:num._call_hook(hooks,key,value)
 return dict(input_mma_axes=a,weight_mma_axes=weight,skip_output_axes=residual,residual_coefficient=coefficient,partials=parts,low_half=combined,upsampled_half=up,scaled_skip_half=scaled,prequantization=pre,output=out,output_physical=physical)
