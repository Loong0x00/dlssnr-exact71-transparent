"""OUTVIEW candidate: core tensor math plus coordinate guards and A-basis plane output.
The output's channel permutation must be proved against actual final-source neurons.
"""
from pathlib import Path
import sys,importlib.util
import numpy as np
HERE=Path(__file__).resolve().parent;V=HERE.parent
sp=importlib.util.spec_from_file_location('_outview_standard_nn',V/'nr-swin8h-standard-main-20260912/model.py');m=importlib.util.module_from_spec(sp);sys.modules[sp.name]=m;sp.loader.exec_module(m);engine=m.engine
PI=engine.attention.pi(np.arange(256))
y,x=np.meshgrid(np.arange(8),np.arange(8),indexing='ij');tile=16*((y//4)*2+x//4)+4*(y%4)+x%4
C_MAP=engine.ffn.axes.c_offset(tile[:,:,None],np.arange(256)[None,None,:],256)

def extents(h,w,output_extent):
 oh,ow=output_extent
 if any(type(v)is not int or not -(1<<31)<=v<(1<<31) for v in (oh,ow)):raise ValueError('signed ABI output extents')
 oh=oh if oh>0 else h;ow=ow if ow>0 else w
 if not(1<=oh<=32 and 1<=ow<=32):raise ValueError('bounded selected output dimensions')
 return oh,ow

def forward(features,slab,*,h=8,w=8,x=0,y=-4,output_extent=(0,0),rsqrt,reciprocal,keep_stages=False):
 gx,gy=engine.geometry(h,w,x,y);oh,ow=extents(h,w,output_extent)
 if oh>y+8*gy or ow>x+8*gx:raise ValueError('requested output includes unproduced coordinates')
 physical=np.empty((16,oh,ow,16),dtype=np.uint8);seen=np.zeros((oh,ow),dtype=np.bool_);stages={}
 for cy in range(gy):
  for cx in range(gx):
   window=engine.ingress(features,h,w,x,y,cx,cy);r=engine.forward_window(window,slab,rsqrt=rsqrt,reciprocal=reciprocal);local=r['physical'][C_MAP]
   for yy in range(8):
    for xx in range(8):
     oy=y+8*cy+yy;ox=x+8*cx+xx
     if 0<=oy<oh and 0<=ox<ow:
      if seen[oy,ox]:raise ValueError('duplicate output coordinate')
      seen[oy,ox]=True;physical[:,oy,ox,:]=local[yy,xx,PI].reshape(16,16)
   if keep_stages:stages[(cx,cy)]=dict(input_window=window,**r)
 if not np.all(seen):raise ValueError('unproduced output')
 return dict(output=physical.tobytes(),output_A_basis=physical.transpose(1,2,0,3).reshape(oh,ow,256).copy(),stages=stages,extent=(oh,ow))
