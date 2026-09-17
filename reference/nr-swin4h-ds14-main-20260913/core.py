"""Ordinary4h source-equation NN: full64-query windows, original sourcezero ingress.
No VM/DLL in forward. Own scalar/range and numerical proof remain separate artifacts.
"""
import numpy as np
import ffn,qkv,normalized,attention

def geometry(h,w,x,y):
 if any(type(i)is not int for i in (h,w,x,y)) or not(8<=h<=32 and 8<=w<=32 and h%8==w%8==0 and x in (-4,0) and y in (-4,0)):raise ValueError('admitted positive multiples8 and XY0/-4')
 return (w-x+7)//8,(h-y+7)//8

def tile_sources(h,w,x,y,cx,cy):
 gx,gy=geometry(h,w,x,y)
 if type(cx)is not int or type(cy)is not int or not 0<=cx<gx or not 0<=cy<gy:raise ValueError('CTA bounds')
 off=np.arange(8192);tile=off//2048;ty=(y+8*cy)//4+tile//2;tx=(x+8*cx)//4+tile%2;valid=(ty>=0)&(ty<h//4)&(tx>=0)&(tx<w//4)
 return np.where(valid,2048*(ty*(w//4)+tx)+off%2048,-1)

def ingress(features,*,h,w,x,y,cta):
 if type(features)is not bytes or len(features)!=h*w*128:raise ValueError('canonical-C128 packed feature span')
 mp=tile_sources(h,w,x,y,*cta);out=np.zeros(8192,dtype=np.uint8);valid=mp>=0;out[valid]=np.frombuffer(features,dtype=np.uint8)[mp[valid]];return out

def forward_window(features,slab,*,rsqrt,reciprocal):
 f=ffn.forward(features,slab);r=qkv.forward_raw(f['physical'].tobytes(),slab);n=normalized.forward(r['warp_bank_view'],slab,rsqrt=rsqrt);a=attention.forward(n['Q'],n['K'],n['V'],f['physical'].tobytes(),slab,reciprocal=reciprocal)
 return dict(ffn=f,raw=r,norm=n,attention=a,physical=a['physical'])

def forward(features,slab,*,h=16,w=16,x,y,rsqrt,reciprocal,keep_stages=False):
 gx,gy=geometry(h,w,x,y);out=np.empty(h*w*128,dtype=np.uint8);seen=np.zeros(h*w*128,dtype=np.bool_);stages={}
 for cy in range(gy):
  for cx in range(gx):
   a=ingress(features,h=h,w=w,x=x,y=y,cta=(cx,cy));z=forward_window(a.tobytes(),slab,rsqrt=rsqrt,reciprocal=reciprocal);mp=tile_sources(h,w,x,y,cx,cy);valid=mp>=0
   if np.any(seen[mp[valid]]):raise ValueError('overlapping output writes')
   out[mp[valid]]=z['physical'][valid];seen[mp[valid]]=True
   if keep_stages:z['ingress']=a;stages[(cx,cy)]=z
 if not np.all(seen):raise ValueError('missing original output tiles')
 return dict(output=out.tobytes(),stages=stages)
