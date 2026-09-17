"""Ordinary8h NN with explicit bounded window ingress/scatter, not host shift inference."""
import numpy as np
import ffn,qkv,normalized,attention

def geometry(h,w,x,y):
 if any(type(i)is not int for i in (h,w,x,y)) or not(8<=h<=32 and 8<=w<=32 and h%8==w%8==0 and x in (-4,0) and y in (-4,0)):raise ValueError('bounded standard geometry')
 return (w-x+7)//8,(h-y+7)//8

def tile_sources(h,w,x,y,cx,cy):
 gx,gy=geometry(h,w,x,y)
 if not(0<=cx<gx and 0<=cy<gy):raise ValueError('CTA domain')
 result=[]
 for ty in range(2):
  for tx in range(2):
   yy=(y+8*cy)//4+ty;xx=(x+8*cx)//4+tx
   result.append(4096*(yy*(w//4)+xx) if 0<=yy<h//4 and 0<=xx<w//4 else None)
 return result

def ingress(features,h,w,x,y,cx,cy):
 if type(features)is not bytes or len(features)!=h*w*256:raise ValueError('packed global feature span')
 out=bytearray(16384)
 for i,off in enumerate(tile_sources(h,w,x,y,cx,cy)):
  if off is not None:out[i*4096:(i+1)*4096]=features[off:off+4096]
 return bytes(out)

def forward_window(features,slab,*,rsqrt,reciprocal):
 f=ffn.forward(features,slab);r=qkv.forward_raw(f['physical'].tobytes(),slab);n=normalized.forward(r['warp_bank_view'],slab,rsqrt=rsqrt)
 a=attention.forward(n['Q'],n['K'],n['V'],f['physical'].tobytes(),slab,reciprocal=reciprocal)
 return dict(ffn=f,raw=r,norm=n,attention=a,physical=a['physical'])

def forward(features,slab,*,h=8,w=8,x=0,y=0,rsqrt,reciprocal,keep_stages=False):
 gx,gy=geometry(h,w,x,y);out=bytearray(h*w*256);seen=set();stages={}
 for cy in range(gy):
  for cx in range(gx):
   window=ingress(features,h,w,x,y,cx,cy);result=forward_window(window,slab,rsqrt=rsqrt,reciprocal=reciprocal)
   for i,off in enumerate(tile_sources(h,w,x,y,cx,cy)):
    if off is not None:
     if off in seen:raise ValueError('duplicate output tile')
     seen.add(off);out[off:off+4096]=result['physical'][i*4096:(i+1)*4096].tobytes()
   if keep_stages:stages[(cx,cy)]=dict(input_window=window,**result)
 if seen!=set(range(0,len(out),4096)):raise ValueError('unproduced output tile')
 return dict(output=bytes(out),stages=stages)
