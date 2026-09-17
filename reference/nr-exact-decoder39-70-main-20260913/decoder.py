"""Transparent CPU decoder39-output -> block70 logical RGBA for restored geometry.
Exact local numerical families only. Encoder skips and texture policy are explicit inputs.
"""
from pathlib import Path
import sys,importlib.util,hashlib,json,struct
V=Path(__file__).resolve().parent.parent
CLEAR=('nn','standard_nn','prefix','ffn','qkv','normalized','attention','head','dependencies','vm','literal','candidate','candidate_v3','candidate_decoder_v3','split_decoder_candidate_v3','block','model','f16_math','f16_policy','f32_ops','ieee_compare')
def _load(tag,path,name):
 saved={n:sys.modules.get(n) for n in CLEAR};old=sys.path[:]
 try:
  for n in CLEAR:sys.modules.pop(n,None)
  sys.path.insert(0,str(path));spec=importlib.util.spec_from_file_location(tag,path/name);m=importlib.util.module_from_spec(spec);sys.modules[tag]=m;spec.loader.exec_module(m);return m
 finally:
  for n,x in saved.items():
   if x is None:sys.modules.pop(n,None)
   else:sys.modules[n]=x
  sys.path[:]=old
class Family:
 def __init__(self,tag,dirname):
  p=V/dirname;self.nn=_load(tag,p,'nn.py');core=getattr(self.nn,'engine',self.nn)
  self.normalized=getattr(self.nn,'normalized',core.normalized);self.attention=getattr(self.nn,'attention',core.attention)
  self.rs=self.normalized.RsqrtHalfDomain.load();self.rc=self.attention.ReciprocalHalfDomain.load()
class ExactDecoder39to70NullTexture:
 def __init__(self,checkpoint):
  cp=Path(checkpoint);h=hashlib.sha256(cp.read_bytes()).hexdigest()
  if h!='890a53f05473dd6fd0980d1ca8501cfd84c1f6d7330b9531ca0981b6a6778d0e':raise ValueError('checkpoint identity')
  with cp.open('rb') as f:n=struct.unpack('<Q',f.read(8))[0];header=json.loads(f.read(n));base=8+n
  def slab(block):
   lo,hi=header[f'block{block}.layer0.layer']['data_offsets']
   with cp.open('rb') as f:f.seek(base+lo);return f.read(hi-lo)
  self.slab={i:slab(i) for i in range(49,71)}
  self.split=_load('_decoder_split40_47',V/'nr-exact-split40-47-main-20260913','chain.py').ExactSplit40to47()
  b48mod=_load('_decoder_b48',V/'nr-swin8h-block48-h32-main-20260913','tiled.py');self.b48=b48mod.Block48H32(cp)
  self.f8=Family('_decoder_f8','nr-swin8h-decoder49-54-h32-main-20260913')
  self.o8=Family('_decoder_o8','nr-swin8h-outview55-h32-main-20260913')
  self.u4=Family('_decoder_u4','nr-swin4h-up56-h64-main-20260913')
  self.f4=Family('_decoder_f4','nr-swin4h-decoder57-60-h64-main-20260913')
  self.o4=Family('_decoder_o4','nr-swin4h-outview61-h64-main-20260913')
  self.u2=Family('_decoder_u2','nr-swin2h-up62-h128-main-20260913')
  self.f2=Family('_decoder_f2','nr-swin2h-decoder63-64-h128-main-20260913')
  self.o2=Family('_decoder_o2','nr-swin2h-outview65-h128-main-20260913')
  self.u1=Family('_decoder_u1','nr-swin1h-up66-h256-main-20260913')
  self.f1=Family('_decoder_f1','nr-swin1h-decoder67-68-h256-main-20260913')
  self.o1=Family('_decoder_o1','nr-swin1h-outview69-h256-main-20260913')
  self.post=Family('_decoder_post','nr-post1h70-h512-main-20260913')
 def forward(self,block39,skip22,skip14,skip8,skip4,skip0,*,textures=None,keep_blocks=False):
  if textures not in (None,(None,None,None)):raise NotImplementedError('live block70 TEX/history family is not restored')
  blocks={39:block39};z=self.split.forward(block39);x=z['output'];blocks.update({i:r['output'] for i,r in z['blocks'].items()})
  x=self.b48.forward(x,skip22)['output'];blocks[48]=x
  for i,(ox,oy) in {49:(-4,-4),50:(-4,0),51:(0,-4),52:(0,0),53:(-4,-4),54:(-4,0)}.items():x=self.f8.nn.forward(x,self.slab[i],h=32,w=32,x=ox,y=oy,rsqrt=self.f8.rs,reciprocal=self.f8.rc)['output'];blocks[i]=x
  x=self.o8.nn.forward(x,self.slab[55],h=32,w=32,x=0,y=-4,rsqrt=self.o8.rs,reciprocal=self.o8.rc)['output'];blocks[55]=x
  x=self.u4.nn.forward(x,skip14,self.slab[56],h=64,w=64,rsqrt=self.u4.rs,reciprocal=self.u4.rc);blocks[56]=x
  for i,(ox,oy) in {57:(0,-4),58:(0,0),59:(-4,-4),60:(-4,0)}.items():x=self.f4.nn.forward(x,self.slab[i],h=64,w=64,x=ox,y=oy,rsqrt=self.f4.rs,reciprocal=self.f4.rc)['output'];blocks[i]=x
  x=self.o4.nn.forward(x,self.slab[61],h=64,w=64,x=0,y=-4,rsqrt=self.o4.rs,reciprocal=self.o4.rc)['output'];blocks[61]=x
  x=self.u2.nn.forward(x,skip8,self.slab[62],h=128,w=128,rsqrt=self.u2.rs,reciprocal=self.u2.rc);blocks[62]=x
  for i,(ox,oy) in {63:(-4,-4),64:(-4,0)}.items():x=self.f2.nn.forward(x,self.slab[i],h=128,w=128,x=ox,y=oy,rsqrt=self.f2.rs,reciprocal=self.f2.rc)['output'];blocks[i]=x
  x=self.o2.nn.forward(x,self.slab[65],h=128,w=128,x=0,y=-4,rsqrt=self.o2.rs,reciprocal=self.o2.rc)['output'];blocks[65]=x
  x=self.u1.nn.forward(x,skip4,self.slab[66],h=256,w=256,x=0,y=0,rsqrt=self.u1.rs,reciprocal=self.u1.rc)['output'];blocks[66]=x
  for i,(ox,oy) in {67:(-4,-4),68:(-4,0)}.items():x=self.f1.nn.forward(x,self.slab[i],h=256,w=256,x=ox,y=oy,rsqrt=self.f1.rs,reciprocal=self.f1.rc)['output'];blocks[i]=x
  x=self.o1.nn.forward(x,self.slab[69],h=256,w=256,x=0,y=-4,rsqrt=self.o1.rs,reciprocal=self.o1.rc)['output'];blocks[69]=x
  z=self.post.nn.forward(x,skip0,self.slab[70],h=512,w=512,ox=-4,oy=-4,rsqrt=self.post.rs,reciprocal=self.post.rc);blocks[70]=z['output_RGBA_f32_bits'].tobytes()
  result={'output_RGBA_f32_bits':z['output_RGBA_f32_bits'],'learned_RGB_logit_half':z['learned_RGB_logit_half'],'scope':'exact CPU transparent decoder40..70; explicit skip inputs; null-TEX logical surface'}
  if keep_blocks:result['blocks']=blocks
  return result
