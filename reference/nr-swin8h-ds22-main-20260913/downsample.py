"""Block22 exact retained-half pool and learned256→512 eight-K32 transition."""
import numpy as np
import ffn
Y,X=np.indices((8,8));TILE=16*((Y//4)*2+X//4)+4*(Y%4)+X%4
def old(k):
 k=np.asarray(k);return 16*(k//16)+2*((k%16)//4)+8*((k%4)//2)+k%2
def outc(n):
 n=np.asarray(n);return 16*(n//16)+4*((n%8)//2)+2*((n%16)//8)+n%2
def make_map():
 r=np.empty((256,512),np.int64)
 for q in range(8):
  for w in range(8):
   for t in range(2):
    for k in range(32):
     for n in range(32):
      lane=4*(n%8)+(k%16)//4;ng=n//8;off=689216+16384*q+1024*w+8192*t+512*(ng//2)+16*lane+8*(ng%2)+4*(k//16)+k%4;r[32*q+int(old(k)),256*t+32*w+int(outc(n))]=off
 assert set(r.ravel())==set(range(689216,820288));return r
WEIGHT_MAP=make_map()
def pool(final):
 if not isinstance(final,np.ndarray) or final.dtype!=np.float16 or final.shape!=(64,256):raise ValueError('tile64x256 half')
 grid=final[TILE];out=np.empty((16,256),np.float16);rn=ffn.num._rn_half
 for y in range(4):
  for x in range(4):
   rho=y%2;sigma=x//2;a=grid[2*y+rho,2*x+sigma];b=grid[2*y+rho,2*x+1-sigma];c=grid[2*y+1-rho,2*x+sigma];d=grid[2*y+1-rho,2*x+1-sigma];ab=rn(a.astype(float)+b.astype(float),'DS22pair0');cd=rn(c.astype(float)+d.astype(float),'DS22pair1');total=rn(ab.astype(float)+cd.astype(float),'DS22sum');out[4*y+x]=rn(total.astype(float)*.25,'DS22quarter')
 return out
def forward(final,slab):
 if type(slab)is not bytes or len(slab)!=820288:raise ValueError('block22 slab')
 pooled=pool(final);codes=ffn.num.fp8_rn_satfinite(pooled);p=ffn.num.decode_e4m3(codes.tobytes(),np.arange(4096).reshape(16,256));w=ffn.num.decode_e4m3(slab,WEIGHT_MAP);o=np.zeros((16,512),np.float16);before={};after={}
 for k0 in range(0,256,32):before[k0]=o.copy();o=ffn.num._rn_half(p[:,k0:k0+32]@w[k0:k0+32]+o.astype(float),'DS22down carriedK32');after[k0]=o.copy()
 q=ffn.num.fp8_rn_satfinite(o);local=q.reshape(4,4,32,16).transpose(2,0,1,3).copy();return dict(pooled_half=pooled,pooled_fp8=codes,weight=w,before=before,after=after,prequantization=o,output_fp8=q,local_planar=local)
