"""Transparent block8 DS: verified2H core and exact unquantized pooling/down transition."""
import numpy as np
import core,downsample

def forward_window(features,slab,*,rsqrt,reciprocal):
 z=core.forward_window(features,slab,rsqrt=rsqrt,reciprocal=reciprocal);z['down']=downsample.forward(z['attention']['prequantization'],slab);return z

def forward(features,slab,*,h=32,w=32,x=0,y=-4,rsqrt,reciprocal,keep_stages=False):
 gx,gy=core.geometry(h,w,x,y)
 if type(features)is not bytes or len(features)!=h*w*64 or type(slab)is not bytes or len(slab)!=69936:raise ValueError('block8 exact spans')
 high=np.empty(h*w*64,np.uint8);hs=np.zeros(len(high),bool);low=np.empty((8,h//2,w//2,16),np.uint8);ls=np.zeros((h//2,w//2),bool);stages={}
 for cy in range(gy):
  for cx in range(gx):
   a=core.ingress(features,h=h,w=w,x=x,y=y,cta=(cx,cy));z=forward_window(a.tobytes(),slab,rsqrt=rsqrt,reciprocal=reciprocal);mp=core.tile_sources(h,w,x,y,cx,cy);valid=mp>=0
   if hs[mp[valid]].any():raise ValueError('duplicate high')
   high[mp[valid]]=z['physical'][valid];hs[mp[valid]]=True
   oy=(y+8*cy)//2;ox=(x+8*cx)//2
   for ly in range(4):
    for lx in range(4):
     yy=oy+ly;xx=ox+lx
     if 0<=yy<h//2 and 0<=xx<w//2:
      if ls[yy,xx]:raise ValueError('duplicate low pixel')
      low[:,yy,xx]=z['down']['local_planar'][:,ly,lx];ls[yy,xx]=True
   if keep_stages:z['ingress']=a;stages[cx,cy]=z
 if not hs.all() or not ls.all():raise ValueError('missing normal outputs')
 return dict(high_C=high.tobytes(),low_A=low.tobytes(),stages=stages)
