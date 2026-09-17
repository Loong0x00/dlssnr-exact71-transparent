"""Transparent second-frame pre0 and public RGBA16F-RTZ history contract."""
from pathlib import Path
import importlib.util,sys,struct,hashlib,json,numpy as np
ROOT=Path(__file__).resolve().parent;T=ROOT/'nr-pre0-temporal-main-20260914';CP=ROOT/'weights/dlssnr-310.8-weights.safetensors'
def _load(tag,path):s=importlib.util.spec_from_file_location(tag,path);m=importlib.util.module_from_spec(s);sys.modules[tag]=m;s.loader.exec_module(m);return m
def _temporal_module():
 names=('attention','embedding','ffn','normalized','qkv','temporal_preblock');saved={n:sys.modules.get(n) for n in names};old=sys.path[:]
 try:
  for n in names:sys.modules.pop(n,None)
  sys.path.insert(0,str(T));return _load('_package_public_temporal',T/'pure_array_model.py')
 finally:
  sys.path[:]=old
  for n,x in saved.items():
   if x is None:sys.modules.pop(n,None)
   else:sys.modules[n]=x
M=_temporal_module();HMMA=_load('_package_temporal_hmma',ROOT/'sm120_hmma.py')
def _table(d,i):
 b=(d/f'APPROX_PHASE{i}_INPUT.bin').read_bytes();u=struct.unpack('<'+'I'*(len(b)//4),b);n=u[0];o=struct.unpack('<'+'I'*n,(d/f'APPROX_PHASE{i}_OUTPUT.bin').read_bytes());return {tuple(u[1+3*j:4+3*j]):o[j] for j in range(n)}
def _rtz(bits):
 sign=(bits>>16)&0x8000;exp=(bits>>23)&255;frac=bits&0x7fffff
 if exp==255:return None if frac else sign|0x7c00
 sig,power=(frac,-149) if exp==0 else (0x800000|frac,exp-150)
 if sig==0:return sign
 top=sig.bit_length()-1+power;step=max(-24,top-10);shift=step-power;q=sig>>shift if shift>0 else sig<<-shift
 if q==0:return sign
 if q>=2048:q>>=1;step+=1
 if step==-24 and q<1024:return sign|q
 e=step+25
 if e>=31:return sign|0x7bff
 if e<=0:return sign|(q>>(1-e))
 return sign|(e<<10)|(q-1024)
def rgba32f_to_history_rgba16f_rtz(words):
 if not isinstance(words,np.ndarray) or words.dtype!=np.uint32 or words.shape!=(512,512,4):raise ValueError('uint32 RGBA F32 bits required')
 def one(x):
  y=_rtz(int(x))
  if y is None:raise ValueError('NaN history')
  return y
 flat=words.ravel();out=np.fromiter((one(x) for x in flat),dtype=np.uint16,count=flat.size).reshape(words.shape);out.setflags(write=False);return out
class PublicTemporalPreSM120:
 """Pinned invocation counter 1: T0 Color, T1 RGBA16F history, T2 zero MVec."""
 def __init__(self,checkpoint=CP,lut_dir=ROOT/'native_approx'):
  checkpoint=Path(checkpoint)
  if hashlib.sha256(checkpoint.read_bytes()).hexdigest()!='890a53f05473dd6fd0980d1ca8501cfd84c1f6d7330b9531ca0981b6a6778d0e':raise ValueError('checkpoint')
  with checkpoint.open('rb') as f:hn=struct.unpack('<Q',f.read(8))[0];h=json.loads(f.read(hn));lo,hi=h['block0.layer0.layer']['data_offsets'];f.seek(8+hn+lo);slab=f.read(hi-lo)
  self.net=M.Preblock0TemporalArrayReference(slab);self.rs,self.rc=self.net.local_numeric_policies();lut={};d=Path(lut_dir)
  for i in (1,2,3,4,5,6,7,8,9):lut.update(_table(d,i))
  ops={'div.approx.ftz.f32':0,'rcp.approx.ftz.f32':5,'lg2.approx.ftz.f32':1,'sqrt.approx.ftz.f32':2,'sin.approx.ftz.f32':3,'cos.approx.ftz.f32':4}
  def ev(op,args):
   k=(ops[op],args[0],args[1] if len(args)>1 else 0)
   if k not in lut:raise ValueError('SM120 temporal approximation domain miss')
   return lut[k]
  self.approx=M.ApproxPolicy('recovered SM120 second-frame domain',True,ev)
  def geom(a):
   sh,sw=struct.unpack_from('<2i',a,208);oh,ow=struct.unpack_from('<2i',a,240);lh,lw=struct.unpack_from('<2i',a,256)
   if (sh,sw,oh,ow,lh,lw)!=(512,512,512,576,256,288):raise ValueError('padded temporal geometry')
   return oh,ow
  M._geometry=geom
  n=M.normalized
  def tree(v,rsqrt):
   rn=n.num._rn_half;s24=rn(v[...,24:32].astype(np.float64)**2,'sq24');s16=rn(v[...,16:24].astype(np.float64)**2,'sq16');pa=rn(v[...,8:16].astype(np.float64)*v[...,8:16].astype(np.float64)+s24.astype(np.float64),'HFMA8+24');pb=rn(v[...,:8].astype(np.float64)*v[...,:8].astype(np.float64)+s16.astype(np.float64),'HFMA0+16');local=rn(pa.astype(np.float64)+pb.astype(np.float64),'local').reshape(64,1,4,2);b2=rn(local.astype(np.float64)+local[:,:,np.arange(4)^2,:].astype(np.float64),'b2');b1=rn(b2.astype(np.float64)+b2[:,:,np.arange(4)^1,:].astype(np.float64),'b1');den=rn(b1[...,0].astype(np.float64)+b1[...,1].astype(np.float64),'den');eps=np.array([948045311],np.uint32).view(np.float32).astype(np.float16)[0];floor=np.maximum(den,eps).astype(np.float16);r32,r16=rsqrt.evaluate(floor);return dict(square24=s24,square16=s16,pair_a=pa,pair_b=pb,local=local,butterfly2=b2,butterfly1=b1,denominator=den,epsilon=eps,floored=floor,rsqrt_f32_bits=r32,factor_half=r16,normalized=rn(v.astype(np.float64)*r16[...,0,None].astype(np.float64),'factor'))
  n.tree=tree
  def embed(features,slab):w=np.frombuffer(slab,dtype='<u2')[M.embedding.WEIGHT_MAP//2].copy();o=HMMA.mma_bits(features.view(np.uint16),w);return dict(input_half=features.copy(),weight_half=w.view(np.float16),output_half=o.view(np.float16))
  M.embedding.forward=embed
 @staticmethod
 def args():
  a=bytearray(264);struct.pack_into('<5Q',a,0,1,5,2,0,0)
  for off in (40,64,136):struct.pack_into('<6f',a,off,0,0,512,512,1/512,1/512)
  struct.pack_into('<2fI',a,160,1/512,1/512,0);struct.pack_into('<5fIfQ',a,172,.25,.75,0.,-1.,-1.,1,1/16,1);struct.pack_into('<2i',a,208,512,512);struct.pack_into('<2Q',a,216,0x40000000,0x20000000);struct.pack_into('<2iQ2i',a,240,512,576,0x50000000,256,288);return bytes(a)
 def forward(self,color,history_half):
  if not isinstance(color,np.ndarray) or color.dtype!=np.float32 or color.shape!=(512,512,4) or not np.all(np.isfinite(color)):raise ValueError('color')
  if not isinstance(history_half,np.ndarray) or history_half.dtype!=np.uint16 or history_half.shape!=(512,512,4):raise ValueError('history half bits')
  cb=np.ascontiguousarray(color).view(np.uint32);hf=history_half.view(np.float16).astype(np.float32).view(np.uint32);T0,T1,T2=1,5,2
  def coord(bits):
   x=struct.unpack('<f',struct.pack('<I',bits))[0]*512-.5;r=round(x)
   if x!=r:raise ValueError('zero-motion history coordinate left exact texel center')
   return min(511,max(0,int(r)))
  def sample(h,u,v):
   x,y=coord(u),coord(v)
   if h==T0:return tuple(map(int,cb[y,x]))
   if h==T1:return tuple(map(int,hf[y,x]))
   if h==T2:return (0,0,0,0)
   raise ValueError('resource')
  sampler=M.SamplerPolicy('point Color, exact-center linear RGBA16F history, zero RG motion',True,sample);state=M.ActualFrontendState(counter_before=1,source_view=T0,previous_output_view=T1,motion_view=T2,control_mask_view=0,depth_view=4,block0_no_edge_route=True,model_history_enabled=True,effective_reset=False,global_history_enabled=True,history_view_created=True,motion_view_created=True);z=self.net.forward_actual(self.args(),state=state,sampler=sampler,approx=self.approx,rsqrt=self.rs,reciprocal=self.rc);return {'primary_A_planar':z.primary_A_planar.tobytes(),'skip_C_packed':z.skip_C_packed.tobytes(),'counter_after':z.counter_after}
