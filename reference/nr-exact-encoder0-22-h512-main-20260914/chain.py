"""Transparent CPU encoder0..22 at the complete-chain H512/H256/.../H16 geometry."""
from pathlib import Path
import sys,importlib.util,types,struct,numpy as np
P=Path(__file__).resolve().parent;V=P.parent;src=V/'nr-exact-encoder0-22-main-20260913';old=sys.path[:];sys.path.insert(0,str(src));spec=importlib.util.spec_from_file_location('_base_encoder512',src/'chain.py');E=importlib.util.module_from_spec(spec);sys.modules[spec.name]=E;spec.loader.exec_module(E);sys.path[:]=old
def geom(limit):
 def f(h,w,x,y):
  if any(type(i)is not int for i in (h,w,x,y)) or not(8<=h<=limit and 8<=w<=limit and h%8==w%8==0 and x in (-4,0) and y in (-4,0)):raise ValueError('bounded extended geometry')
  return (w-x+7)//8,(h-y+7)//8
 return f
def source_shape(limit):
 def f(h,w,extent):
  eh,ew=extent;eh=eh if eh>0 else h;ew=ew if ew>0 else w
  if not(1<=eh<=limit and 1<=ew<=limit):raise ValueError('bounded source extent')
  return eh,ew
 return f
def patch_tree(root,limit):
 seen=set()
 def rec(m,depth):
  if id(m) in seen or depth>4:return
  seen.add(id(m))
  if hasattr(m,'geometry'):m.geometry=geom(limit)
  if hasattr(m,'source_shape'):m.source_shape=source_shape(limit)
  if hasattr(m,'shape'):m.shape=source_shape(limit)
  for x in vars(m).values():
   if isinstance(x,types.ModuleType) and getattr(x,'__file__',None) and '/reviews/nr-' in str(x.__file__):rec(x,depth+1)
 rec(root,0)
def pre_geometry(args):
 if type(args)is not bytes or len(args)!=264:raise ValueError('own264ABI')
 h,w=struct.unpack_from('<2i',args,240);ah,aw=struct.unpack_from('<2i',args,256)
 if not(8<=h<=512 and 8<=w<=512 and h%8==w%8==0 and (ah,aw)==(h//2,w//2)):raise ValueError('bounded H512 pre geometry')
 if struct.unpack_from('<Q',args,200)[0] or any(struct.unpack_from('<4Q',args,8)):raise ValueError('firstcounter0/absent temporal only')
 return h,w
E.PRE['model'].nn.geometry=pre_geometry
for root,limit in [(E.B1['model'],256),(E.B23['model'],256),(E.D4['nn'],256),(E.B5['nn'],128),(E.B67['nn'],128),(E.D8['nn'],128),(E.B9['nn'],64),(E.B1013['nn'],64),(E.D14['nn'],64),(E.B15['nn'],32),(E.B1621['nn'],32),(E.D22['nn'],32)]:patch_tree(root,limit)
def ingress15_h32(features,*,h,w,x,y,cta,source_extent=(0,0)):
 eh,ew=source_extent;eh=eh if eh>0 else h;ew=ew if ew>0 else w
 if not(1<=eh<=32 and 1<=ew<=32) or type(features)is not bytes or len(features)!=eh*ew*256:raise ValueError('exact extended planar source')
 cx,cy=cta;src=np.frombuffer(features,np.uint8);out=np.zeros(16384,np.uint8)
 for wy in range(8):
  for j in range(16):
   for lane in range(32):
    yy=y+8*cy+4*(j//8)+2*(j%2)+lane//16;xx=x+8*cx+4*((j%8)//4)+(lane//4)%4;valid=(eh==1 or 0<=yy<eh) and (ew==1 or 0<=xx<ew);sy=0 if eh==1 else yy;sx=0 if ew==1 else xx;group=2*wy+(j%4)//2;stage=4096*(j//4)+512*wy+16*lane+4*(j%4)
    if valid:
     off=16*((group*eh+sy)*ew+sx)+4*(lane%4);out[stage:stage+4]=src[off:off+4]
 return out
E.B15['nn'].ingress=ingress15_h32
class ExactEncoder0to22H512(E.ExactEncoder0to22):
 def forward(self,args=None,*,sampler=None,precomputed=None,keep_blocks=True):
  out={}
  if precomputed is None:
   if args is None or sampler is None:raise ValueError('first ingress requires args and sampler')
   z=self.pre.forward(args,sampler=sampler);primary=z['primary_A_planar'];skip=z['skip_C_packed']
  else:
   if args is not None or sampler is not None or not isinstance(precomputed,dict) or set(precomputed)!=set(('primary_A_planar','skip_C_packed')):raise ValueError('exact precomputed block0 boundary')
   primary=precomputed['primary_A_planar'];skip=precomputed['skip_C_packed']
  x=primary if type(primary)is bytes else primary.tobytes();skip=skip if type(skip)is bytes else skip.tobytes()
  if len(x)!=256*256*32 or len(skip)!=512*512*32:raise ValueError('H512 block0 boundary sizes')
  out[0]={'primary':x,'skip':skip};z=self.b1.forward_packed(x,h=256,w=256);x=z['output_packed'];out[1]=x
  for i,o in ((2,(-4,-4)),(3,(-4,0))):
   b=self.b23.blocks[i];x=E.B23['model'].nn.forward(x,b._slab,h=256,w=256,x=o[0],y=o[1],rsqrt=b._rs,reciprocal=b._rc)['output'];out[i]=x
  for tag,m,s,rs,rc in self.modules:
   if tag=='d4':z=m['nn'].forward(x,s[4],h=256,w=256,rsqrt=rs,reciprocal=rc);out[4]={'high':z['high_C'],'low':z['low_A']};x=z['low_A']
   elif tag=='b5':z=m['nn'].forward(x,s[5],h=128,w=128,rsqrt=rs,reciprocal=rc);x=z['output'];out[5]=x
   elif tag=='b67':
    for i,o in ((6,(-4,-4)),(7,(-4,0))):x=m['nn'].forward(x,s[i],h=128,w=128,x=o[0],y=o[1],rsqrt=rs,reciprocal=rc)['output'];out[i]=x
   elif tag=='d8':z=m['nn'].forward(x,s[8],h=128,w=128,rsqrt=rs,reciprocal=rc);out[8]={'high':z['high_C'],'low':z['low_A']};x=z['low_A']
   elif tag=='b9':x=m['nn'].forward(x,s[9],h=64,w=64,rsqrt=rs,reciprocal=rc)['output'];out[9]=x
   elif tag=='b1013':
    for i,o in ((10,(-4,-4)),(11,(-4,0)),(12,(0,-4)),(13,(0,0))):x=m['nn'].forward(x,s[i],h=64,w=64,x=o[0],y=o[1],rsqrt=rs,reciprocal=rc)['output'];out[i]=x
   elif tag=='d14':z=m['nn'].forward(x,s[14],h=64,w=64,rsqrt=rs,reciprocal=rc);out[14]={'high':z['high_C'],'low':z['low_A']};x=z['low_A']
   elif tag=='b15':x=m['nn'].forward(x,s[15],h=32,w=32,rsqrt=rs,reciprocal=rc);out[15]=x
   elif tag=='b1621':
    for i,o in ((16,(-4,-4)),(17,(-4,0)),(18,(0,-4)),(19,(0,0)),(20,(-4,-4)),(21,(-4,0))):x=m['nn'].forward(x,s[i],h=32,w=32,x=o[0],y=o[1],rsqrt=rs,reciprocal=rc)['output'];out[i]=x
   elif tag=='d22':z=m['nn'].forward(x,s[22],h=32,w=32,rsqrt=rs,reciprocal=rc);out[22]={'high':z['high_C'],'low':z['low_A']};x=z['low_A']
  return {'output':x,'blocks':out,'scope':'transparent firstcounter0 H512 encoder0..22; no history/native'}
def synthetic_args_sampler():
 a=bytearray(E.PRE['input_fixture'].args(512,512));struct.pack_into('<f',a,180,2/512);return bytes(a),E.PRE['input_fixture'].sampler()
