"""Exact source-ordered block8 unquantized64→pool→FP8→learned128 transition."""
import numpy as np
import ffn
Y,X=np.indices((8,8));TILE=16*((Y//4)*2+X//4)+4*(Y%4)+X%4

def serialized(c):
 c=np.asarray(c);return 16*(c//16)+4*((c%8)//2)+2*((c//8)%2)+c%2

def weight_offset(n,c):
 n=np.asarray(n);k=serialized(np.asarray(c));inner=16*(4*(n%8)+(k%16)//4)+8*((n%16)//8)+4*((k%32)//16)+k%4
 return 61744+4096*(k//32)+1024*(n//32)+512*((n%32)//16)+inner

def pool(final_half):
 if not isinstance(final_half,np.ndarray) or final_half.dtype!=np.float16 or final_half.shape!=(64,64):raise ValueError('local tile-row64x64 final half')
 grid=final_half[TILE];out=np.empty((16,64),np.float16);rn=ffn.num._rn_half
 for y in range(4):
  for x in range(4):
   rho=y%2;sigma=x//2;a=grid[2*y+rho,2*x+sigma];b=grid[2*y+rho,2*x+1-sigma];c=grid[2*y+1-rho,2*x+sigma];d=grid[2*y+1-rho,2*x+1-sigma];ab=rn(a.astype(np.float64)+b.astype(np.float64),'DS8 ordered pair0');cd=rn(c.astype(np.float64)+d.astype(np.float64),'DS8 ordered pair1');total=rn(ab.astype(np.float64)+cd.astype(np.float64),'DS8 pair sum');out[4*y+x]=rn(total.astype(np.float64)*.25,'DS8 quarter')
 return out

def forward(final_half,slab):
 if type(slab)is not bytes or len(slab)!=69936:raise ValueError('own block8 slab')
 pooled=pool(final_half);codes=ffn.num.fp8_rn_satfinite(pooled);p=ffn.num.decode_e4m3(codes.tobytes(),np.arange(1024).reshape(16,64));n=np.arange(128)[None,:];c=np.arange(64)[:,None];wm=weight_offset(n,c);w=ffn.num.decode_e4m3(slab,wm);out=np.zeros((16,128),np.float16);before={};after={}
 for k0 in (0,32):before[k0]=out.copy();out=ffn.num._rn_half(p[:,k0:k0+32]@w[k0:k0+32]+out.astype(np.float64),'DS8 learned down carriedK32');after[k0]=out.copy()
 q=ffn.num.fp8_rn_satfinite(out);local=np.empty((8,4,4,16),np.uint8)
 for canonical in range(128):r=int(serialized(canonical));local[r//16,:,:,r%16]=q.reshape(4,4,128)[:,:,canonical]
 return dict(pooled_half=pooled,pooled_fp8=codes,weight=w,weight_offsets=wm,before=before,after=after,prequantization=out,output_fp8=q,local_planar=local)
