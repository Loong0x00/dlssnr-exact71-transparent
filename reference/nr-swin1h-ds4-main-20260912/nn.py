"""Transparent DS4 full-window core plus unquantized ordered pooling/learned projection.
Only conservative no-padding normal output domain. No VM/DLL in forward.
"""
import numpy as np
import core,downsample,down_policies

def forward_window(features,slab,*,rsqrt,reciprocal):
 z=core.forward_window(features,slab,rsqrt=rsqrt,reciprocal=reciprocal);final=z['attention']['prequantization'];raster=downsample.tile_rows_to_raster(final);d=downsample.downsample(raster,slab,down_policies.policies());z['down']=d;return z

def forward(features,slab,*,h=64,w=64,x=0,y=-4,rsqrt,reciprocal,keep_stages=False):
 gx,gy=core.geometry(h,w,x,y)
 if type(features)is not bytes or len(features)!=h*w*32 or type(slab)is not bytes or len(slab)!=22720:raise ValueError('own DS4 source/slab span')
 high=np.empty(h*w*32,np.uint8);hs=np.zeros(len(high),bool);low=np.empty(h//2*(w//2)*64,np.uint8);ls=np.zeros(len(low),bool);stages={}
 for cy in range(gy):
  for cx in range(gx):
   a=core.ingress(features,h=h,w=w,x=x,y=y,cta=(cx,cy));z=forward_window(a.tobytes(),slab,rsqrt=rsqrt,reciprocal=reciprocal);mp=core.tile_sources(h,w,x,y,cx,cy);valid=mp>=0
   if hs[mp[valid]].any():raise ValueError('duplicate high source stores')
   high[mp[valid]]=z['physical'][valid];hs[mp[valid]]=True
   for off,b in downsample.low_global_stores(z['down'].low_fp8_mmaN,h,w,x,y,cx,cy):
    if not 0<=off<=len(low)-len(b) or ls[off:off+len(b)].any():raise ValueError('low normal writer bounds/overlap')
    low[off:off+len(b)]=np.frombuffer(b,np.uint8);ls[off:off+len(b)]=True
   if keep_stages:z['ingress']=a;stages[cx,cy]=z
 if not hs.all() or not ls.all():raise ValueError('missing source normal output bytes')
 return dict(high_C=high.tobytes(),low_A=low.tobytes(),stages=stages)
