"""Own2h source-derived local prefix128→64; actual operand proof is separate."""
from pathlib import Path
import sys,importlib.util
import numpy as np
HERE=Path(__file__).resolve().parent;V=HERE.parent
sys.path.insert(0,str(V/'nr-vit-expand-tensor-reference-20260912'));import expand_tensor_reference as num
sp=importlib.util.spec_from_file_location('_4h_axes',V/'nr-vit-contract-layout-main-20260912/tensor_reference.py');axes=importlib.util.module_from_spec(sp);sp.loader.exec_module(axes)
m=np.arange(16)[:,None];k=np.arange(128)[None,:];n=np.arange(64)[None,:]
A_MAP=16*((k//16*4+m//4)*4+m%4)+k%16
B_MAP=28672+axes.b_offset(k.T,n,64)
y,x=np.meshgrid(np.arange(8),np.arange(8),indexing='ij');tile=16*((y//4)*2+x//4)+4*(y%4)+x%4
C_MAP=axes.c_offset(tile[:,:,None],np.arange(64)[None,None,:],64)
def forward(source,skip,slab,hooks=None):
 for b,nbytes in ((source,2048),(skip,4096),(slab,70048)):
  if type(b)is not bytes or len(b)!=nbytes:raise ValueError('4h local prefix spans')
 a=num.decode_e4m3(source,A_MAP);b=num.decode_e4m3(slab,B_MAP);s=num.decode_e4m3(skip,C_MAP);gamma=np.frombuffer(slab,dtype='<f2',count=64,offset=36992).copy()
 if not np.all(np.isfinite(gamma)):raise num.UnsupportedNonFinite('prefix gamma')
 carry=np.zeros((16,64),dtype=np.float16);before={};after={}
 for k0 in range(0,128,32):
  before[k0]=carry.copy();carry=num._rn_half(a[:,k0:k0+32]@b[k0:k0+32]+carry.astype(np.float64),'4h prefix K32');after[k0]=carry.copy()
 low=carry.reshape(4,4,64);near=low.repeat(2,0).repeat(2,1);scaled=num._rn_half(s*gamma.astype(np.float64),'4h late skip');pre=num._rn_half(near.astype(np.float64)+scaled.astype(np.float64),'4h late add');codes=num.fp8_rn_satfinite(pre);physical=np.empty(4096,dtype=np.uint8);physical[C_MAP]=codes
 result=dict(input_mma_axes=a,weight_mma_axes=b,skip=s,gamma=gamma,before=before,after=after,low=low,nearest=near,scaled_skip=scaled,prequantization=pre,output=codes,physical=physical)
 for name,value in result.items():
  if isinstance(value,np.ndarray):num._call_hook(hooks,name,value)
 return result
