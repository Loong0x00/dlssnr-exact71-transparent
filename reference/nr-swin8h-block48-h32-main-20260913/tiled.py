"""Transparent tiled block48 H16low/H32skip→H32 output using exact local NN."""
from pathlib import Path
import importlib.util,sys
import numpy as np
P=Path(__file__).resolve().parent;V=P.parent
spec=importlib.util.spec_from_file_location('_b48h32',V/'nr-swin8h-block48-reference-main-20260912/block.py');b=importlib.util.module_from_spec(spec);sys.modules[spec.name]=b;spec.loader.exec_module(b)
def coff(y,x,c,h,w,C):
 m=4*(y%4)+x%4;tile=(y//4)*(w//4)+x//4;lane=4*(m%8)+(c%8)//2;j=4*((c%32)//16)+2*(m//8)+(c%16)//8
 return 16*C*tile+512*(c//32)+16*lane+2*j+c%2
def poff(y,x,c,h,w,C):return 16*((c//16*h+y)*w+x)+c%16
class Block48H32:
 def __init__(self,checkpoint):self.local=b.Block48Reference.from_checkpoint(checkpoint)
 def forward(self,low,skip):
  if type(low)is not bytes or len(low)!=16*16*512 or type(skip)is not bytes or len(skip)!=32*32*256:raise ValueError('connected H16/H32 spans')
  out=bytearray(32*32*256);seen=set();tiles=[]
  for cy in range(4):
   for cx in range(4):
    lp=bytearray(4*4*512);sp=bytearray(8*8*256)
    for y in range(4):
     for x in range(4):
      for c in range(512):lp[poff(y,x,c,4,4,512)]=low[poff(4*cy+y,4*cx+x,c,16,16,512)]
    for y in range(8):
     for x in range(8):
      for c in range(256):sp[coff(y,x,c,8,8,256)]=skip[coff(8*cy+y,8*cx+x,c,32,32,256)]
    z=self.local.forward_packed(bytes(lp),bytes(sp));tiles.append(z['output_packed'])
    for y in range(8):
     for x in range(8):
      for c in range(256):
       dst=coff(8*cy+y,8*cx+x,c,32,32,256)
       if dst in seen:raise ValueError('duplicate')
       seen.add(dst);out[dst]=z['output_packed'][coff(y,x,c,8,8,256)]
  if len(seen)!=len(out):raise ValueError('missing')
  return dict(output=bytes(out),tiles=tiles)
