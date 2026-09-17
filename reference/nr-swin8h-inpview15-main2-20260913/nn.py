"""Block15 exact source planar→shared staging plus verified8H learned core."""
import numpy as np
import standard_nn
def ingress(features,*,h,w,x,y,cta,source_extent=(0,0)):
 eh,ew=source_extent;eh=eh if eh>0 else h;ew=ew if ew>0 else w
 if not(1<=eh<=16 and 1<=ew<=16) or type(features)is not bytes or len(features)!=eh*ew*256:raise ValueError('exact sixteen-planar source')
 cx,cy=cta;src=np.frombuffer(features,np.uint8);out=np.zeros(16384,np.uint8)
 for wy in range(8):
  for j in range(16):
   for lane in range(32):
    yy=y+8*cy+4*(j//8)+2*(j%2)+lane//16;xx=x+8*cx+4*((j%8)//4)+(lane//4)%4;valid=(eh==1 or 0<=yy<eh) and (ew==1 or 0<=xx<ew);sy=0 if eh==1 else yy;sx=0 if ew==1 else xx;group=2*wy+(j%4)//2;stage=4096*(j//4)+512*wy+16*lane+4*(j%4)
    if valid:
     c=16*group+4*(lane%4);off=16*((group*eh+sy)*ew+sx)+4*(lane%4);out[stage:stage+4]=src[off:off+4]
 return out
def forward(features,slab,*,h=8,w=8,x=0,y=0,source_extent=(0,0),rsqrt,reciprocal):
 gx,gy=standard_nn.geometry(h,w,x,y);out=bytearray(h*w*256);seen=set()
 for cy in range(gy):
  for cx in range(gx):
   a=ingress(features,h=h,w=w,x=x,y=y,cta=(cx,cy),source_extent=source_extent);z=standard_nn.forward_window(a.tobytes(),slab,rsqrt=rsqrt,reciprocal=reciprocal)
   for i,off in enumerate(standard_nn.tile_sources(h,w,x,y,cx,cy)):
    if off is not None:
     if off in seen:raise ValueError('duplicate')
     seen.add(off);out[off:off+4096]=z['physical'][i*4096:(i+1)*4096].tobytes()
 if seen!=set(range(0,len(out),4096)):raise ValueError('missing')
 return bytes(out)
