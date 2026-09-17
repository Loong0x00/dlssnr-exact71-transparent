"""Transparent block30 including retained-half projection pool and512→1024 head."""
import numpy as np
from split_standard_candidate import *
def pack_c_channels(a,h,w,channels):
 if a.shape!=(h*w,channels) or a.dtype!=np.uint8 or channels%32:raise ValueError('logical C codes')
 out=np.empty(h*w*channels,np.uint8)
 for y in range(h):
  for x in range(w):
   m=4*(y%4)+x%4;tile=(y//4)*(w//4)+x//4
   for c in range(channels):
    lane=4*(m%8)+(c%8)//2;j=4*((c%32)//16)+2*(m//8)+(c%16)//8;off=16*channels*tile+512*(c//32)+16*lane+2*j+c%2;out[off]=a[y*w+x,c]
 return out.tobytes()
class Block30Candidate(StandardSplitCandidate):
 def layer3_pool(self,source,skip,h,w):
  z=super().layer3(source,skip,h,w);c=z['prequantization'].reshape(h,w,512);rn=num._rn_half;low=np.empty((h//2,w//2,512),np.float16)
  for y in range(h//2):
   for x in range(w//2):
    p0=rn(c[2*y,2*x].astype(np.float64)+c[2*y,2*x+1].astype(np.float64),'block30 pool pair0');p1=rn(c[2*y+1,2*x].astype(np.float64)+c[2*y+1,2*x+1].astype(np.float64),'block30 pool pair1');s=rn(p0.astype(np.float64)+p1.astype(np.float64),'block30 pool sum');low[y,x]=rn(s.astype(np.float64)*.25,'block30 pool quarter')
  q=num.fp8_rn_satfinite(low.reshape(-1,512));z['pool_half']=low;z['pool_logical']=q;z['pool_output']=pack_c(q,h//2,w//2);return z
 def head(self,source,h,w):
  codes=unpack_a(source,h,w);a=num.decode_e4m3(codes.tobytes(),np.arange(codes.size).reshape(codes.shape));pre=matmul_k32(a,self.w['head_weight'][pi_index(512)],label='block30 head K32');q,d=quant_decode(pre);return dict(output=pack_c_channels(q,h,w,1024),logical=q,prequantization=pre)
 def forward_complete(self,source,h,w,origin,*,rsqrt,reciprocal):
  l0=self.layer0(source,h,w);l1=self.layer1(l0['output'],source,h,w);l2=self.layer2_shifted(l1['output'],h,w,origin,rsqrt=rsqrt,reciprocal=reciprocal);l3=self.layer3_pool(l2['output'],l1['output'],h,w);head=self.head(l3['pool_output'],h//2,w//2);return dict(output=head['output'],layers={0:l0,1:l1,2:l2,3:l3,4:head})
