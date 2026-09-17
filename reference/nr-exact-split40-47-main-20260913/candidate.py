"""Transparent split16 block23 arithmetic candidate; first validates layer0 dense output."""
from pathlib import Path
import importlib.util,numpy as np
P=Path(__file__).resolve().parent;V=P.parent
spec=importlib.util.spec_from_file_location('_exactnum23',V/'nr-swin8h-ds22-main-20260913/ffn.py');f=importlib.util.module_from_spec(spec);spec.loader.exec_module(f);num=f.num
def matmul_k32(a,w,seed=None,label='K32'):
 if a.dtype not in (np.dtype(np.float16),np.dtype(np.float64)) or w.dtype not in (np.dtype(np.float16),np.dtype(np.float64)) or a.shape[1]!=w.shape[0] or a.shape[1]%32:raise ValueError('exact finite/K32 matrix')
 c=np.zeros((a.shape[0],w.shape[1]),np.float16) if seed is None else seed.copy()
 for k in range(0,a.shape[1],32):c=num._rn_half(a[:,k:k+32]@w[k:k+32]+c.astype(np.float64),label)
 return c
def quant_decode(x):
 q=num.fp8_rn_satfinite(x);return q,num.decode_e4m3(q.tobytes(),np.arange(q.size).reshape(q.shape))
def activation(x):
 q=np.minimum(x.astype(np.float64),4);q=np.maximum(q,-4);a,b,c=np.array([0xab28,0x3728,0x3b28],np.uint16).view(np.float16).astype(np.float64);t=num._rn_half(a*np.abs(q)+b,'split gate FMA1');z=num._rn_half(q*t.astype(np.float64)+c,'split gate FMA2');return num._rn_half(x.astype(np.float64)*z.astype(np.float64),'split gate mul')
def unpack_planar(buf,h,w):
 if type(buf)is not bytes or len(buf)!=h*w*512:raise ValueError('planar512')
 return np.frombuffer(buf,np.uint8).reshape(32,h,w,16).transpose(1,2,0,3).reshape(h*w,512).copy()
def packed_c_offset(h,w,y,x,c):
 m=4*(y%4)+x%4;lane=4*(m%8)+(c%8)//2;j=4*((c%32)//16)+2*(m//8)+(c%16)//8
 return 8192*((y//4)*(w//4)+x//4)+1024*(c//64)+512*((c%64)//32)+16*lane+2*j+c%2
def pack_c(a,h,w):
 if a.shape!=(h*w,512) or a.dtype!=np.uint8:raise ValueError('logical codes')
 out=np.empty(h*w*512,np.uint8)
 for y in range(h):
  for x in range(w):
   for c in range(512):out[packed_c_offset(h,w,y,x,c)]=a[y*w+x,c]
 return out.tobytes()
class Block23Candidate:
 def __init__(self,path=P/'block23-logical.npz'):
  z=np.load(path)
  try:self.w={k:z[k].copy() for k in z.files}
  finally:z.close()
  if self.w['first_projection_weight'].shape!=(512,512):raise ValueError('logical bundle')
 def layer0(self,source,h,w):
  xcodes=unpack_planar(source,h,w);x=num.decode_e4m3(xcodes.tobytes(),np.arange(xcodes.size).reshape(xcodes.shape));hidden=matmul_k32(x,self.w['first_projection_weight'],label='split23 first K32');hidden_codes,hidden=quant_decode(hidden);groups=[];details=[]
  for g in range(8):
   e=matmul_k32(hidden[:,64*g:64*(g+1)],self.w['group_expand_weight'][g],label='split23 expand K32');act=activation(e);o=matmul_k32(act,self.w['group_project_weight'][g],label='split23 project K32');q,d=quant_decode(o);groups.append(q);details.append(dict(expand=e,activation=act,prequantization=o))
  logical=np.concatenate(groups,axis=1);return dict(output=pack_c(logical,h,w),logical=logical,first_prequantization=hidden,first_codes=hidden_codes,groups=details)
