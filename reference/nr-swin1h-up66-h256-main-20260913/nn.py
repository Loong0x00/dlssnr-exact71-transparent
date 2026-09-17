"""Own1h NN candidate. Unquantized prefix/FFN residual forks, not a scaled2h model."""
import numpy as np
import prefix,ffn,qkv,normalized,attention

def geometry(h,w,x,y):
 if any(type(i)is not int for i in (h,w,x,y)) or not(8<=h<=256 and 8<=w<=256 and h%8==w%8==0 and x in (0,-4) and y in (0,-4)):raise ValueError('bounded highmultiples8 and XY0/-4')
 return (w-x+7)//8,(h-y+7)//8

def crops(low,skip,*,h,w,x,y,cta):
 if type(low)is not bytes or len(low)!=h//2*(w//2)*64 or type(skip)is not bytes or len(skip)!=h*w*32:raise ValueError('exact A-low and C-highskip spans')
 gx,gy=geometry(h,w,x,y);cx,cy=cta
 if not(0<=cx<gx and 0<=cy<gy):raise ValueError('CTA')
 # Admitted origins are multiples4, so signed truncating division coincides here.
 c,ly,lx=np.indices((4,4,4));yy=(y+8*cy)//2+ly;xx=(x+8*cx)//2+lx;valid=(yy>=0)&(yy<h//2)&(xx>=0)&(xx<w//2)
 lowview=np.zeros((4,4,4,16),dtype=np.uint8);full=np.frombuffer(low,dtype=np.uint8).reshape(4,h//2,w//2,16);lowview[valid]=full[c[valid],yy[valid],xx[valid]]
 off=np.arange(2048);tile=off//512;ty=(y+8*cy)//4+tile//2;tx=(x+8*cx)//4+tile%2;real=(ty>=0)&(ty<h//4)&(tx>=0)&(tx<w//4);mp=512*(ty*(w//4)+tx)+off%512
 sk=np.zeros(2048,dtype=np.uint8);sk[real]=np.frombuffer(skip,dtype=np.uint8)[mp[real]]
 return lowview.tobytes(),sk.tobytes(),np.where(real,mp,-1)

def forward_window(low,skip,slab,*,rsqrt,reciprocal):
 p=prefix.forward(low,skip,slab);unquantized=np.empty((64,32),dtype=np.float16);unquantized[prefix.tile.ravel()]=p['prequantization'].reshape(64,32)
 f=ffn.forward(unquantized,slab);r=qkv.forward_raw(f['physical'].tobytes(),slab);n=normalized.forward(r['warp_bank_view'],slab,rsqrt=rsqrt);a=attention.forward(n['Q'],n['K'],n['V'],f['prequantization'],slab,reciprocal=reciprocal)
 return dict(prefix=p,ffn=f,raw=r,norm=n,attention=a,physical=a['physical'])

def forward(low,skip,slab,*,h=64,w=64,x=0,y=0,rsqrt,reciprocal,keep_stages=False):
 gx,gy=geometry(h,w,x,y);out=np.empty(h*w*32,dtype=np.uint8);seen=np.zeros(h*w*32,dtype=np.bool_);stages={}
 for cy in range(gy):
  for cx in range(gx):
   a,b,mp=crops(low,skip,h=h,w=w,x=x,y=y,cta=(cx,cy));z=forward_window(a,b,slab,rsqrt=rsqrt,reciprocal=reciprocal);valid=mp>=0
   if np.any(seen[mp[valid]]):raise ValueError('overlapping C output')
   seen[mp[valid]]=True;out[mp[valid]]=z['physical'][valid]
   if keep_stages:stages[(cx,cy)]=z
 if not np.all(seen):raise ValueError('unproduced output')
 return dict(output=out.tobytes(),stages=stages)
