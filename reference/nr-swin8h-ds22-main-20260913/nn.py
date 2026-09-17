"""Transparent block22 DS from verified8H core plus exact transition."""
import numpy as np
import core,downsample
def forward(features,slab,*,h=8,w=8,x=0,y=-4,rsqrt,reciprocal):
 gx,gy=core.geometry(h,w,x,y)
 if type(features)is not bytes or len(features)!=h*w*256 or type(slab)is not bytes or len(slab)!=820288:raise ValueError('block22 spans')
 high=bytearray(h*w*256);hs=set();low=np.empty((32,h//2,w//2,16),np.uint8);ls=set()
 for cy in range(gy):
  for cx in range(gx):
   a=core.ingress(features,h,w,x,y,cx,cy);z=core.forward_window(a,slab,rsqrt=rsqrt,reciprocal=reciprocal);d=downsample.forward(z['attention']['prequantization'],slab)
   for i,off in enumerate(core.tile_sources(h,w,x,y,cx,cy)):
    if off is not None:
     if off in hs:raise ValueError('duplicate high')
     hs.add(off);high[off:off+4096]=z['physical'][i*4096:(i+1)*4096].tobytes()
   oy=(y+8*cy)//2;ox=(x+8*cx)//2
   for ly in range(4):
    for lx in range(4):
     yy=oy+ly;xx=ox+lx
     if 0<=yy<h//2 and 0<=xx<w//2:
      if (yy,xx) in ls:raise ValueError('duplicate low')
      ls.add((yy,xx));low[:,yy,xx]=d['local_planar'][:,ly,lx]
 if hs!=set(range(0,len(high),4096)) or len(ls)!=(h//2)*(w//2):raise ValueError('missing outputs')
 return dict(high_C=bytes(high),low_A=low.tobytes())
