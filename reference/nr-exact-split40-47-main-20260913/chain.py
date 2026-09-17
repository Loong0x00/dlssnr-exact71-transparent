from pathlib import Path
from split_decoder_candidate_v3 import StandardSplitCandidate,HalfLut
P=Path(__file__).resolve().parent;V=P.parent;ORIGINS={40:(0,0),41:(-4,-4),42:(-4,0),43:(0,-4),44:(0,0),45:(-4,-4),46:(-4,0),47:(0,-4)}
class ExactSplit40to47:
 def __init__(self):self.blocks={i:StandardSplitCandidate(V/f'nr-split16h-block{i}-dense-main-20260913/block{i}-logical.npz') for i in range(40,48)};self.rs=HalfLut.load('rsqrt');self.rc=HalfLut.load('reciprocal')
 def forward(self,source):
  if type(source)is not bytes or len(source)!=131072:raise ValueError('H16C512 exact source')
  rows={};value=source
  for i in range(40,48):z=self.blocks[i].forward_shifted(value,16,16,ORIGINS[i],rsqrt=self.rs,reciprocal=self.rc);rows[i]=z;value=z['output']
  return dict(output=value,blocks=rows,scope='transparent exact CPU split40..47 H16; no native sync')
