"""Pure first/reset-frame public-CUDA pre0 stage for the pinned SM120 target."""
from pathlib import Path
import importlib.util,sys,struct,hashlib,numpy as np
ROOT=Path(__file__).resolve().parent
CHECKPOINT_SHA='890a53f05473dd6fd0980d1ca8501cfd84c1f6d7330b9531ca0981b6a6778d0e'
def _load(tag,path):
 spec=importlib.util.spec_from_file_location(tag,path);m=importlib.util.module_from_spec(spec);sys.modules[tag]=m;spec.loader.exec_module(m);return m
def _load_pre():
 names=('nn','ffn','qkv','normalized','attention','embedding','preprocess_candidate','approx_policies','f16_math')
 saved={n:sys.modules.get(n) for n in names};old=sys.path[:]
 try:
  for n in names:sys.modules.pop(n,None)
  d=ROOT/'nr-pre1h0-main-20260912';sys.path.insert(0,str(d));return _load('_public_sm120_pre_model',d/'model.py')
 finally:
  sys.path[:]=old
  for n,x in saved.items():
   if x is None:sys.modules.pop(n,None)
   else:sys.modules[n]=x
PRE=_load_pre();HMMA=_load('_public_sm120_hmma',ROOT/'sm120_hmma.py')
def _read_table(inp,out):
 raw=inp.read_bytes();u=struct.unpack('<'+'I'*(len(raw)//4),raw);n=u[0];z=struct.unpack('<'+'I'*n,out.read_bytes())
 if len(u)!=1+3*n:return None
 return {tuple(u[1+3*i:4+3*i]):z[i] for i in range(n)}
class PublicPre0SM120:
 """Exact pinned 512x512 source -> 512x576/256x288 block0 producer.

 Constructor I/O loads the checkpoint and immutable recovered instruction tables.
 ``forward`` itself performs array arithmetic only and no file/process/device I/O.
 """
 HANDLE=1
 def __init__(self,checkpoint=ROOT/'weights/dlssnr-310.8-weights.safetensors',lut_dir=ROOT/'native_approx'):
  checkpoint=Path(checkpoint)
  if hashlib.sha256(checkpoint.read_bytes()).hexdigest()!=CHECKPOINT_SHA:raise ValueError('checkpoint identity')
  lut={}
  for i in (1,2,6,7):
   z=_read_table(Path(lut_dir)/f'APPROX_PHASE{i}_INPUT.bin',Path(lut_dir)/f'APPROX_PHASE{i}_OUTPUT.bin')
   if z is None:raise ValueError('approx table shape')
   lut.update(z)
  self._lut=lut;ops={'div.approx.ftz.f32':0,'lg2.approx.ftz.f32':1,'sqrt.approx.ftz.f32':2,'sin.approx.ftz.f32':3,'cos.approx.ftz.f32':4}
  def evaluate(op,args):
   key=(ops[op],args[0],args[1] if len(args)>1 else 0)
   if key not in self._lut:raise ValueError('SM120 approximation domain miss')
   return self._lut[key]
  policy=PRE.pre.ApproxPolicy('recovered SM120 complete 512x576 first-frame domain',True,evaluate)
  def geometry(args):
   if type(args)is not bytes or len(args)!=264:raise ValueError('264-byte ABI')
   h,w=struct.unpack_from('<2i',args,240);lh,lw=struct.unpack_from('<2i',args,256);sh,sw=struct.unpack_from('<2i',args,208)
   if (sh,sw,h,w,lh,lw)!=(512,512,512,576,256,288):raise ValueError('pinned public padded geometry')
   return h,w
  PRE.nn.geometry=geometry;self._geometry=geometry
  n=PRE.normalized
  def tree(v,rsqrt):
   rn=n.num._rn_half;s24=rn(v[...,24:32].astype(np.float64)**2,'sq24');s16=rn(v[...,16:24].astype(np.float64)**2,'sq16');pa=rn(v[...,8:16].astype(np.float64)*v[...,8:16].astype(np.float64)+s24.astype(np.float64),'HFMA8+24');pb=rn(v[...,:8].astype(np.float64)*v[...,:8].astype(np.float64)+s16.astype(np.float64),'HFMA0+16');local=rn(pa.astype(np.float64)+pb.astype(np.float64),'local').reshape(64,1,4,2);b2=rn(local.astype(np.float64)+local[:,:,np.arange(4)^2,:].astype(np.float64),'b2');b1=rn(b2.astype(np.float64)+b2[:,:,np.arange(4)^1,:].astype(np.float64),'b1');den=rn(b1[...,0].astype(np.float64)+b1[...,1].astype(np.float64),'den');eps=np.array([948045311],np.uint32).view(np.float32).astype(np.float16)[0];floor=np.maximum(den,eps).astype(np.float16);r32,r16=rsqrt.evaluate(floor);norm=rn(v.astype(np.float64)*r16[...,0,None].astype(np.float64),'factor');return dict(square24=s24,square16=s16,pair_a=pa,pair_b=pb,local=local,butterfly2=b2,butterfly1=b1,denominator=den,epsilon=eps,floored=floor,rsqrt_f32_bits=r32,factor_half=r16,normalized=norm)
  n.tree=tree
  def embed(features,slab):
   if features.shape!=(64,16) or features.dtype!=np.float16:raise ValueError('embedding input')
   w=np.frombuffer(slab,dtype='<u2')[PRE.embedding.WEIGHT_MAP//2].copy();o=HMMA.mma_bits(features.view(np.uint16),w);return dict(input_half=features.copy(),weight_half=w.view(np.float16),output_half=o.view(np.float16))
  PRE.embedding.forward=embed;self._model=PRE.Preblock0FirstReference.from_checkpoint(checkpoint,approx=policy)
 @staticmethod
 def args(*,tone=.25,structure=.75,style=0,skin=-1.,use_auto_mask=False):
  if type(style)is not int or not 0<=style<=128:raise ValueError('style integer 0..128')
  vals=(tone,structure,skin)
  if not all(np.isfinite(x) for x in vals):raise ValueError('finite controls')
  a=b=-1.0
  if use_auto_mask:a=skin if skin>=0 else structure;b=structure
  q=bytearray(264);struct.pack_into('<Q',q,0,PublicPre0SM120.HANDLE);struct.pack_into('<6f',q,136,0.,0.,512.,512.,1/512,1/512);struct.pack_into('<2fI',q,160,1/512,1/512,0);struct.pack_into('<5fIf',q,172,tone,structure,style/128.,a,b,1,1/16);struct.pack_into('<Q',q,200,0);struct.pack_into('<2i',q,208,512,512);struct.pack_into('<2Q',q,216,0x40000000,0x20000000);struct.pack_into('<2i',q,240,512,576);struct.pack_into('<Q',q,248,0x50000000);struct.pack_into('<2i',q,256,256,288);return bytes(q)
 def forward(self,color_rgba_f32,*,tone=.25,structure=.75,style=0,skin=-1.,use_auto_mask=False):
  if not isinstance(color_rgba_f32,np.ndarray) or color_rgba_f32.dtype!=np.float32 or color_rgba_f32.shape!=(512,512,4) or not np.all(np.isfinite(color_rgba_f32)):raise ValueError('finite float32[512,512,4] color required')
  tex=np.ascontiguousarray(color_rgba_f32).view(np.uint32).copy();handle=self.HANDLE
  def sample(h,xb,yb):
   if h!=handle:raise ValueError('texture identity')
   x=struct.unpack('<f',struct.pack('<I',xb))[0];y=struct.unpack('<f',struct.pack('<I',yb))[0];ix=min(511,max(0,int(np.floor(x*512))));iy=min(511,max(0,int(np.floor(y*512))));return tuple(map(int,tex[iy,ix]))
  PRE.nn.geometry=self._geometry;sampler=PRE.pre.SamplerPolicy('exact RGBA32F normalized point/clamp array',True,sample);z=self._model.forward(self.args(tone=tone,structure=structure,style=style,skin=skin,use_auto_mask=use_auto_mask),sampler=sampler)
  return {'primary_A_planar':z['primary_A_planar'].tobytes(),'skip_C_packed':z['skip_C_packed'].tobytes()}
