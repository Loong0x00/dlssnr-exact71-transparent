"""Exact transparent split blocks23..30 at H16, producing H8/N64 ViT input."""
from pathlib import Path
import sys,importlib.util,numpy as np
P=Path(__file__).resolve().parent;V=P.parent;S=V/'nr-exact-split23-30-main-20260913';sys.path.insert(0,str(S))
from candidate_v3 import Block23CandidateV3,HalfLut,unpack_a,packed_a_offset,pack_c
from split_standard_candidate import StandardSplitCandidate
from block30_candidate import Block30Candidate
sys.path.pop(0)
spec=importlib.util.spec_from_file_location('_h16_repack',V/'nr-vit-repack-reference-20260912/repack_2d_to_1d_fp8_reference.py');repack=importlib.util.module_from_spec(spec);sys.modules[spec.name]=repack;spec.loader.exec_module(repack)
ORIGINS={24:(-4,-4),25:(-4,0),26:(0,-4),27:(0,0),28:(-4,-4),29:(-4,0),30:(0,-4)}
def _block23_attention_grid(model,source,h,w,rs,rc):
 global_codes=unpack_a(source,h,w);result=np.empty((h*w,512),np.uint8);seen=set();windows=[]
 for oy in range(0,h,8):
  for ox in range(0,w,8):
   raw=np.zeros(8*8*512,np.uint8)
   for ly in range(8):
    for lx in range(8):
     for ch in range(512):raw[packed_a_offset(8,8,ly,lx,ch)]=global_codes[(oy+ly)*w+ox+lx,ch]
   z=model.layer2(raw.tobytes(),8,8,rsqrt=rs,reciprocal=rc);windows.append(z)
   for ly in range(8):
    for lx in range(8):
     yy,xx=oy+ly,ox+lx
     if (yy,xx) in seen:raise ValueError('duplicate block23 output')
     seen.add((yy,xx));result[yy*w+xx]=z['logical'][8*ly+lx]
 if len(seen)!=h*w:raise ValueError('missing block23 output')
 return {'output':pack_c(result,h,w),'logical':result,'windows':windows,'origin':(0,0)}
class ExactSplit23to30H16:
 def __init__(self):
  self.b23=Block23CandidateV3(V/'nr-split16h-block23-dense-main-20260913/block23-logical.npz');self.standard={i:StandardSplitCandidate(V/f'nr-split16h-block{i}-dense-main-20260913/block{i}-logical.npz') for i in range(24,30)};self.b30=Block30Candidate(V/'nr-split16h-block30-dense-main-20260913/block30-logical-full.npz');self.rs=HalfLut.load('rsqrt');self.rc=HalfLut.load('reciprocal')
 def forward(self,source,keep_blocks=False):
  if type(source)is not bytes or len(source)!=16*16*512:raise ValueError('H16 planarA C512 source')
  out={};l0=self.b23.layer0(source,16,16);l1=self.b23.layer1(l0['output'],source,16,16);l2=_block23_attention_grid(self.b23,l1['output'],16,16,self.rs,self.rc);l3=self.b23.layer3(l2['output'],l1['output'],16,16);z={'output':l3['output'],'layers':{0:l0,1:l1,2:l2,3:l3}};out[23]=z;x=z['output']
  for i in range(24,30):z=self.standard[i].forward_shifted(x,16,16,ORIGINS[i],rsqrt=self.rs,reciprocal=self.rc);out[i]=z;x=z['output']
  z=self.b30.forward_complete(x,16,16,ORIGINS[30],rsqrt=self.rs,reciprocal=self.rc);out[30]=z;head=z['output'];packed=bytes(repack.run_repack(head,8,8));r={'output':packed,'head':head,'retained_high':z['layers'][3]['output'],'logical_N':64,'padded_N':64,'scope':'exact transparent CPU H16 split23..30 -> H8/N64'}
  if keep_blocks:r['blocks']=out
  return r
