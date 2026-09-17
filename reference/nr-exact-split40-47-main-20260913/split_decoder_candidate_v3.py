"""Transparent standard split16 arithmetic, parameterized by logical block bundle."""
import numpy as np
from candidate_decoder_v3 import *
class StandardSplitCandidate(Block23CandidateV3):
 def __init__(self,path):super().__init__(path)
 def layer0(self,source,h,w):
  xcodes=unpack_a(source,h,w);x=num.decode_e4m3(xcodes.tobytes(),np.arange(xcodes.size).reshape(xcodes.shape));hidden=matmul_k32(x,self.w['first_projection_weight'],label='split standard first K32');hidden_codes,hidden=quant_decode(hidden);groups=[];details=[]
  for g in range(8):
   e=matmul_k32(hidden[:,64*g:64*(g+1)],self.w['group_expand_weight'][g],label='split standard expand K32');act_pre=activation(e);act_codes,act=quant_decode(act_pre);o=matmul_k32(act,self.w['group_project_weight'][g],label='split standard project K32');q,d=quant_decode(o);groups.append(q);details.append(dict(expand=e,activation_prequantization=act_pre,activation_codes=act_codes,prequantization=o))
  logical=np.concatenate(groups,axis=1);return dict(output=pack_c(logical,h,w),logical=logical,first_prequantization=hidden,first_codes=hidden_codes,groups=details)
 def layer1(self,source,skip,h,w):
  codes=unpack_a(source,h,w);x=num.decode_e4m3(codes.tobytes(),np.arange(codes.size).reshape(codes.shape));skip_codes=unpack_c(skip,h,w);sf=num.decode_e4m3(skip_codes.tobytes(),np.arange(skip_codes.size).reshape(skip_codes.shape));gamma=self.w['ffn_cos_skip'].astype(np.float64);seed=num._rn_half(sf*gamma[None,:],'split standard layer1 residual mul');pre=matmul_k32(x,self.w['weight3'][pi_index(512)],seed=seed,label='split standard layer1 K32');q,d=quant_decode(pre);return dict(output=pack_c(q,h,w),logical=q,prequantization=pre,seed=seed)
 def layer2_shifted(self,source,h,w,origin,*,rsqrt,reciprocal):
  x0,y0=origin;gx=(w-x0+7)//8;gy=(h-y0+7)//8;global_codes=unpack_a(source,h,w);result=np.empty((h*w,512),np.uint8);seen=set();windows=[]
  for cy in range(gy):
   for cx in range(gx):
    ox=x0+8*cx;oy=y0+8*cy;window=np.zeros((64,512),np.uint8)
    for ly in range(8):
     for lx in range(8):
      yy=oy+ly;xx=ox+lx
      if 0<=yy<h and 0<=xx<w:window[8*ly+lx]=global_codes[yy*w+xx]
    raw=np.empty(32768,np.uint8)
    for ly in range(8):
     for lx in range(8):
      for ch in range(512):raw[packed_a_offset(8,8,ly,lx,ch)]=window[8*ly+lx,ch]
    z=super().layer2(raw.tobytes(),8,8,rsqrt=rsqrt,reciprocal=reciprocal);windows.append(z)
    for ly in range(8):
     for lx in range(8):
      yy=oy+ly;xx=ox+lx
      if 0<=yy<h and 0<=xx<w:
       if (yy,xx) in seen:raise ValueError('duplicate shifted output')
       seen.add((yy,xx));result[yy*w+xx]=z['logical'][8*ly+lx]
  if len(seen)!=h*w:raise ValueError('missing shifted output')
  return dict(output=pack_c(result,h,w),logical=result,windows=windows,origin=origin)
 def forward_shifted(self,source,h,w,origin,*,rsqrt,reciprocal):
  l0=self.layer0(source,h,w);l1=self.layer1(l0['output'],source,h,w);l2=self.layer2_shifted(l1['output'],h,w,origin,rsqrt=rsqrt,reciprocal=reciprocal);l3=self.layer3(l2['output'],l1['output'],h,w);return dict(output=l3['output'],layers={0:l0,1:l1,2:l2,3:l3})
