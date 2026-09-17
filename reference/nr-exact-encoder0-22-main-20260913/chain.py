"""Unified exact bounded CPU encoder0..22, assembled from independently accepted modules.
No DLL/community arithmetic/VM in forward; firstcounter0 procedural ingress remains explicit.
"""
from pathlib import Path
import importlib,sys
V=Path(__file__).resolve().parent.parent
NAMES=('model','nn','core','standard_nn','ffn','qkv','normalized','attention','embedding','downsample','down_policies','dependencies','runtime_dependencies','prefix','f16_math','preprocess_candidate','approx_policies','input_fixture')
def load(folder,wanted):
 saved={n:sys.modules.get(n) for n in NAMES};paths=sys.path[:]
 try:
  for n in NAMES:sys.modules.pop(n,None)
  sys.path.insert(0,str(V/folder));return {n:importlib.import_module(n) for n in wanted}
 finally:
  for n in NAMES:
   if saved[n] is None:sys.modules.pop(n,None)
   else:sys.modules[n]=saved[n]
  sys.path[:]=paths
PRE=load('nr-pre1h0-main-20260912',('model','input_fixture'))
B1=load('nr-swin1h-inpview-main-20260912',('model',))
B23=load('nr-swin1h-encoder23-main-20260912',('model',))
D4=load('nr-swin1h-ds4-main-20260912',('nn','normalized','attention'))
B5=load('nr-swin2h-inpview5-main-20260913',('nn','normalized','attention'))
B67=load('nr-swin2h-encoder67-main-20260913',('nn','normalized','attention'))
D8=load('nr-swin2h-ds8-main-20260913',('nn','normalized','attention'))
B9=load('nr-swin4h-inpview9-main-20260913',('nn','normalized','attention'))
B1013=load('nr-swin4h-encoder10-13-main-20260913',('nn','normalized','attention'))
D14=load('nr-swin4h-ds14-main-20260913',('nn','normalized','attention'))
B15=load('nr-swin8h-inpview15-main2-20260913',('nn','normalized','attention'))
B1621=load('nr-swin8h-encoder16-21-main-20260913',('nn','normalized','attention'))
D22=load('nr-swin8h-ds22-main-20260913',('nn','normalized','attention'))
class ExactEncoder0to22:
 def __init__(self,checkpoint):
  self.pre=PRE['model'].Preblock0FirstReference.from_checkpoint(checkpoint,approx=PRE['input_fixture'].approx());self.b1=B1['model'].Inpview1hReference.from_checkpoint(checkpoint);self.b23=B23['model'].Standard1hStack.from_checkpoint(checkpoint);self.modules=[]
  for tag,mods,blocks in [('d4',D4,(4,)),('b5',B5,(5,)),('b67',B67,(6,7)),('d8',D8,(8,)),('b9',B9,(9,)),('b1013',B1013,(10,11,12,13)),('d14',D14,(14,)),('b15',B15,(15,)),('b1621',B1621,(16,17,18,19,20,21)),('d22',D22,(22,))]:
   root=V/({'d4':'nr-swin1h-ds4-main-20260912','b5':'nr-swin2h-inpview5-main-20260913','b67':'nr-swin2h-encoder67-main-20260913','d8':'nr-swin2h-ds8-main-20260913','b9':'nr-swin4h-inpview9-main-20260913','b1013':'nr-swin4h-encoder10-13-main-20260913','d14':'nr-swin4h-ds14-main-20260913','b15':'nr-swin8h-inpview15-main2-20260913','b1621':'nr-swin8h-encoder16-21-main-20260913','d22':'nr-swin8h-ds22-main-20260913'}[tag]);slabs={i:(root/f'block{i}-slab.bin').read_bytes() for i in blocks};rs=mods['normalized'].RsqrtHalfDomain.load();rc=mods['attention'].ReciprocalHalfDomain.load();self.modules.append((tag,mods,slabs,rs,rc))
 def forward(self,args,*,sampler):
  out={};z=self.pre.forward(args,sampler=sampler);value=z['primary_A_planar'].tobytes();out[0]=dict(primary=value,skip=z['skip_C_packed'].tobytes());z=self.b1.forward_packed(value);value=z['output_packed'];out[1]=value;z=self.b23.forward_packed(value)
  for i,b in z['block_outputs'].items():out[i]=b
  value=z['output_packed']
  for tag,m,s,rs,rc in self.modules:
   if tag=='d4':z=m['nn'].forward(value,s[4],rsqrt=rs,reciprocal=rc);out[4]=dict(high=z['high_C'],low=z['low_A']);value=z['low_A']
   elif tag=='b5':z=m['nn'].forward(value,s[5],rsqrt=rs,reciprocal=rc);value=z['output'];out[5]=value
   elif tag=='b67':
    for i,o in ((6,(-4,-4)),(7,(-4,0))):value=m['nn'].forward(value,s[i],h=32,w=32,x=o[0],y=o[1],rsqrt=rs,reciprocal=rc)['output'];out[i]=value
   elif tag=='d8':z=m['nn'].forward(value,s[8],rsqrt=rs,reciprocal=rc);out[8]=dict(high=z['high_C'],low=z['low_A']);value=z['low_A']
   elif tag=='b9':value=m['nn'].forward(value,s[9],rsqrt=rs,reciprocal=rc)['output'];out[9]=value
   elif tag=='b1013':
    for i,o in ((10,(-4,-4)),(11,(-4,0)),(12,(0,-4)),(13,(0,0))):value=m['nn'].forward(value,s[i],h=16,w=16,x=o[0],y=o[1],rsqrt=rs,reciprocal=rc)['output'];out[i]=value
   elif tag=='d14':z=m['nn'].forward(value,s[14],rsqrt=rs,reciprocal=rc);out[14]=dict(high=z['high_C'],low=z['low_A']);value=z['low_A']
   elif tag=='b15':value=m['nn'].forward(value,s[15],rsqrt=rs,reciprocal=rc);out[15]=value
   elif tag=='b1621':
    for i,o in ((16,(-4,-4)),(17,(-4,0)),(18,(0,-4)),(19,(0,0)),(20,(-4,-4)),(21,(-4,0))):value=m['nn'].forward(value,s[i],h=8,w=8,x=o[0],y=o[1],rsqrt=rs,reciprocal=rc)['output'];out[i]=value
   elif tag=='d22':z=m['nn'].forward(value,s[22],rsqrt=rs,reciprocal=rc);out[22]=dict(high=z['high_C'],low=z['low_A']);value=z['low_A']
  return dict(output=value,blocks=out,scope='firstcounter0/absentT1..T4 procedural sampler, exactCPU base variants0..22; no native wait/history/fullNR')
