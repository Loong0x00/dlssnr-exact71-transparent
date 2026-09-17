"""Raw pre-normalization64x768 projection; bank identities require consumers.
This is not yet the normalized Q/K/V operator or full attention.
"""
from pathlib import Path
import sys,importlib.util
import numpy as np
HERE=Path(__file__).resolve().parent;V=HERE.parent
sys.path.insert(0,str(V/'nr-vit-expand-tensor-reference-20260912'));import expand_tensor_reference as num
spec=importlib.util.spec_from_file_location('_swin8_qkv_axes',V/'nr-vit-contract-layout-main-20260912/tensor_reference.py');axes=importlib.util.module_from_spec(spec);spec.loader.exec_module(axes)
A_MAP=axes.a_offset(np.arange(64)[:,None],np.arange(64)[None,:],64)
B_MAP=28832+axes.b_offset(np.arange(64)[:,None],np.arange(192)[None,:],192)
def forward_raw(features,slab,hooks=None):
 if type(features)is not bytes or len(features)!=4096 or type(slab)is not bytes or len(slab)!=69936:raise ValueError('own64x256/slab contract')
 a=num.decode_e4m3(features,A_MAP);b=num.decode_e4m3(slab,B_MAP);c=np.zeros((64,192),dtype=np.float16);before={};after={}
 num._call_hook(hooks,'input_mma_axes',a);num._call_hook(hooks,'weight_matrix',b)
 for k in range(0,64,32):
  before[k]=c.copy();c=num._rn_half(a[:,k:k+32]@b[k:k+32]+c.astype(np.float64),'raw projection perK32');after[k]=c.copy();num._call_hook(hooks,f'after_K{k+32}',c)
 banks=c.reshape(64,2,3,32);num._call_hook(hooks,'raw_projected_half',c);num._call_hook(hooks,'warp_bank_view',banks)
 return dict(input_mma_axes=a,weight_matrix=b,raw_projected_half=c,warp_bank_view=banks,before=before,after=after)
