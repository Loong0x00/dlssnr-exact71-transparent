"""Block14 exact ordered pool and learned128→256 four-K32 transition."""
import numpy as np
import ffn
Y,X=np.indices((8,8));TILE=16*((Y//4)*2+X//4)+4*(Y%4)+X%4
def old_channel(k):
 k=np.asarray(k);return 16*(k//16)+2*((k%16)//4)+8*((k%4)//2)+k%2
def out_channel(n):
 n=np.asarray(n);return 16*(n//16)+4*((n%8)//2)+2*((n%16)//8)+n%2
def weight_map():
 result=np.empty((128,256),np.int64)
 for q in range(4):
  for w in range(4):
   for t in range(2):
    for k in range(32):
     for n in range(32):
      lane=4*(n%8)+(k%16)//4;ng=n//8;off=197168+8192*q+1024*w+4096*t+512*(ng//2)+16*lane+8*(ng%2)+4*(k//16)+k%4;result[32*q+int(old_channel(k)),128*t+32*w+int(out_channel(n))]=off
 assert len(set(result.ravel()))==32768 and set(result.ravel())==set(range(197168,229936));return result
WEIGHT_MAP=weight_map()
def pool(final):
 if not isinstance(final,np.ndarray) or final.dtype!=np.float16 or final.shape!=(64,128):raise ValueError('tile rows64x128 half')
 grid=final[TILE];out=np.empty((16,128),np.float16);rn=ffn.num._rn_half
 for y in range(4):
  for x in range(4):
   rho=y%2;sigma=x//2;a=grid[2*y+rho,2*x+sigma];b=grid[2*y+rho,2*x+1-sigma];c=grid[2*y+1-rho,2*x+sigma];d=grid[2*y+1-rho,2*x+1-sigma];ab=rn(a.astype(float)+b.astype(float),'DS14pair0');cd=rn(c.astype(float)+d.astype(float),'DS14pair1');total=rn(ab.astype(float)+cd.astype(float),'DS14sum');out[4*y+x]=rn(total.astype(float)*.25,'DS14quarter')
 return out
def forward(final,slab):
 if type(slab)is not bytes or len(slab)!=229936:raise ValueError('block14 slab')
 pooled=pool(final);codes=ffn.num.fp8_rn_satfinite(pooled);p=ffn.num.decode_e4m3(codes.tobytes(),np.arange(2048).reshape(16,128));w=ffn.num.decode_e4m3(slab,WEIGHT_MAP);out=np.zeros((16,256),np.float16);before={};after={}
 for k0 in (0,32,64,96):before[k0]=out.copy();out=ffn.num._rn_half(p[:,k0:k0+32]@w[k0:k0+32]+out.astype(float),'DS14down carriedK32');after[k0]=out.copy()
 q=ffn.num.fp8_rn_satfinite(out);local=q.reshape(4,4,16,16).transpose(2,0,1,3).copy();return dict(pooled_half=pooled,pooled_fp8=codes,weight=w,before=before,after=after,prequantization=out,output_fp8=q,local_planar=local)
