"""Own4h OUTVIEW NN: full-window core + guarded physical-A output.
Static C-slot/code relation pi is separately derived; actual operand/address proof follows.
"""
from pathlib import Path
import numpy as np
import standard_nn as engine
HERE=Path(__file__).resolve().parent
PI=engine.attention.pi(np.arange(128));yy,xx=np.meshgrid(np.arange(8),np.arange(8),indexing='ij');mm=16*((yy//4)*2+xx//4)+4*(yy%4)+xx%4
C_MAP=engine.ffn.axes.c_offset(mm[:,:,None],np.arange(128)[None,None,:],128)
def extents(h,w,output_extent):
 oh,ow=output_extent
 if any(type(v)is not int or not -(1<<31)<=v<(1<<31) for v in (oh,ow)):raise ValueError('signed output extents')
 oh=oh if oh>0 else h;ow=ow if ow>0 else w
 if not(1<=oh<=64 and 1<=ow<=64):raise ValueError('bounded effective output extents')
 return oh,ow

def forward(features,slab,*,h=16,w=16,x=0,y=-4,output_extent=(0,0),rsqrt,reciprocal,keep_stages=False):
 gx,gy=engine.geometry(h,w,x,y);oh,ow=extents(h,w,output_extent)
 if oh>y+8*gy or ow>x+8*gx:raise ValueError('requested output includes unproduced coordinates')
 out=np.empty((8,oh,ow,16),dtype=np.uint8);seen=np.zeros((oh,ow),dtype=np.bool_);stages={}
 for cy in range(gy):
  for cx in range(gx):
   inp=engine.ingress(features,h=h,w=w,x=x,y=y,cta=(cx,cy));z=engine.forward_window(inp.tobytes(),slab,rsqrt=rsqrt,reciprocal=reciprocal);logical=z['physical'][C_MAP]
   for ly in range(8):
    for lx in range(8):
     gy0=y+8*cy+ly;gx0=x+8*cx+lx
     if 0<=gy0<oh and 0<=gx0<ow:
      if seen[gy0,gx0]:raise ValueError('overlapping OUTVIEW coordinate')
      seen[gy0,gx0]=True;out[:,gy0,gx0,:]=logical[ly,lx,PI].reshape(8,16)
   if keep_stages:z['ingress']=inp;stages[(cx,cy)]=z
 if not np.all(seen):raise ValueError('unproduced output')
 return dict(output=out.tobytes(),output_A_basis=out.transpose(1,2,0,3).reshape(oh,ow,128).copy(),extent=(oh,ow),stages=stages)
