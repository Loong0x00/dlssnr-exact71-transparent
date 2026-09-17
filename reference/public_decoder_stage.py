"""Pure padded decoder39..70 for the pinned first/reset public SM120 route."""
from pathlib import Path
import importlib.util,sys,types,struct,numpy as np,hashlib,json
ROOT=Path(__file__).resolve().parent;CP=ROOT/'weights/dlssnr-310.8-weights.safetensors'
def load(tag,path):s=importlib.util.spec_from_file_location(tag,path);m=importlib.util.module_from_spec(s);sys.modules[tag]=m;s.loader.exec_module(m);return m
DECMOD=load('_package_padded_decoder',ROOT/'nr-exact-decoder39-70-main-20260913/decoder.py');D39=load('_package_padded_d39',ROOT/'nr-decoder-input-tensor-main-20260912/tensor_reference.py');HMMA=load('_package_padded_hmma',ROOT/'sm120_hmma.py')
def geometry(h,w,x,y):
 if any(type(z)is not int for z in(h,w,x,y)) or not(8<=h<=576 and 8<=w<=576 and x in(-4,0) and y in(-4,0)):raise ValueError('padded decoder geometry')
 return (w-x+7)//8,(h-y+7)//8
def shape(h,w,extent):
 eh,ew=extent;eh=eh if eh>0 else h;ew=ew if ew>0 else w
 if not(1<=eh<=576 and 1<=ew<=576):raise ValueError('padded decoder extent')
 return eh,ew
def norm_patch(n):
 def tree(v,rsqrt):
  rn=n.num._rn_half;s24=rn(v[...,24:32].astype(np.float64)**2,'sq24');s16=rn(v[...,16:24].astype(np.float64)**2,'sq16');pa=rn(v[...,8:16].astype(np.float64)*v[...,8:16].astype(np.float64)+s24.astype(np.float64),'HFMA8+24');pb=rn(v[...,:8].astype(np.float64)*v[...,:8].astype(np.float64)+s16.astype(np.float64),'HFMA0+16');local=rn(pa.astype(np.float64)+pb.astype(np.float64),'local').reshape(v.shape[0],v.shape[1],4,2);b2=rn(local.astype(np.float64)+local[:,:,np.arange(4)^2,:].astype(np.float64),'b2');b1=rn(b2.astype(np.float64)+b2[:,:,np.arange(4)^1,:].astype(np.float64),'b1');den=rn(b1[...,0].astype(np.float64)+b1[...,1].astype(np.float64),'den');eps=np.array([948045311],np.uint32).view(np.float32).astype(np.float16)[0];floor=np.maximum(den,eps).astype(np.float16);r32,r16=rsqrt.evaluate(floor);return dict(square24=s24,square16=s16,pair_a=pa,pair_b=pb,local=local,butterfly2=b2,butterfly1=b1,denominator=den,epsilon=eps,floored=floor,rsqrt_f32_bits=r32,factor_half=r16,normalized=rn(v.astype(np.float64)*r16[...,0,None].astype(np.float64),'factor'))
 n.tree=tree
def _walk(root,seen=None,d=0):
 if seen is None:seen=set()
 if id(root) in seen or d>9:return
 seen.add(id(root))
 if isinstance(root,types.ModuleType):
  if hasattr(root,'geometry'):root.geometry=geometry
  if hasattr(root,'source_shape'):root.source_shape=shape
  if hasattr(root,'shape'):root.shape=shape
  if hasattr(root,'extents'):root.extents=shape
 vals=vars(root).values() if isinstance(root,types.ModuleType) else root.values() if isinstance(root,dict) else root if isinstance(root,(list,tuple)) else ()
 for z in vals:
  if isinstance(z,(types.ModuleType,dict,list,tuple)):_walk(z,seen,d+1)
def _read_lut(inp,out):
 u=struct.unpack('<'+'I'*(inp.stat().st_size//4),inp.read_bytes());n=u[0];z=struct.unpack('<'+'I'*n,out.read_bytes());return {tuple(u[1+3*i:4+3*i]):z[i] for i in range(n)}
class PublicDecoderSM120:
 def __init__(self,checkpoint=CP,lut_dir=ROOT/'native_approx'):
  checkpoint=Path(checkpoint)
  if hashlib.sha256(checkpoint.read_bytes()).hexdigest()!='890a53f05473dd6fd0980d1ca8501cfd84c1f6d7330b9531ca0981b6a6778d0e':raise ValueError('checkpoint identity')
  def own_geometry(h,w,x,y):
   if any(type(z)is not int for z in(h,w,x,y)) or not(8<=h<=576 and 8<=w<=576 and x in(-4,0) and y in(-4,0)):raise ValueError('padded decoder geometry')
   return (w-x+7)//8,(h-y+7)//8
  def own_shape(h,w,extent):
   eh,ew=extent;eh=eh if eh>0 else h;ew=ew if ew>0 else w
   if not(1<=eh<=576 and 1<=ew<=576):raise ValueError('padded decoder extent')
   return eh,ew
  self._geometry=own_geometry;self._shape=own_shape;self.dm=DECMOD.ExactDecoder39to70NullTexture(checkpoint);dm=self.dm;_walk(DECMOD)
  for f in (dm.f8,dm.o8,dm.u4,dm.f4,dm.o4,dm.u2,dm.f2,dm.o2,dm.u1,dm.f1,dm.o1,dm.post):_walk(f.nn);norm_patch(f.normalized)
  def patch_up(fam):
   p=fam.nn.prefix;old=p.forward
   def forward(source,skip,slab,hooks=None):
    r=old(source,skip,slab,hooks);pre=p.num._rn_half(r['nearest'].astype(np.float64)+r['skip']*r['gamma'].astype(np.float64),'sm120 up HFMA');codes=p.num.fp8_rn_satfinite(pre);physical=np.empty_like(r['physical']);physical[p.C_MAP]=codes;r.update(prequantization=pre,output=codes,physical=physical);return r
   p.forward=forward
  for f in (dm.u4,dm.u2,dm.u1):patch_up(f)
  sb=next(iter(dm.split.blocks.values()));self.sg=sb.layer2.__func__.__globals__;snum=self.sg['num']
  def snorm(v,rsqrt):
   rn=snum._rn_half;s24=rn(v[...,24:32].astype(np.float64)**2,'sq24');s16=rn(v[...,16:24].astype(np.float64)**2,'sq16');pa=rn(v[...,8:16].astype(np.float64)*v[...,8:16].astype(np.float64)+s24.astype(np.float64),'HFMA8+24');pb=rn(v[...,:8].astype(np.float64)*v[...,:8].astype(np.float64)+s16.astype(np.float64),'HFMA0+16');local=rn(pa.astype(np.float64)+pb.astype(np.float64),'local').reshape(64,16,4,2);b2=rn(local.astype(np.float64)+local[:,:,np.arange(4)^2,:].astype(np.float64),'b2');b1=rn(b2.astype(np.float64)+b2[:,:,np.arange(4)^1,:].astype(np.float64),'b1');den=rn(b1[...,0].astype(np.float64)+b1[...,1].astype(np.float64),'den');eps=np.array([948045311],np.uint32).view(np.float32).astype(np.float16)[0];floor=np.maximum(den,eps).astype(np.float16);raw,factor=rsqrt.evaluate(floor);return dict(normalized=rn(v.astype(np.float64)*factor[...,0,None].astype(np.float64),'factor'),denominator=den,floor=floor,raw=raw)
  self.sg['_norm_tree']=snorm
  # Block48 local SM120 contracts.
  self.b48=dm.b48;self.local=self.b48.local;self.bm=self.local.forward_packed.__func__.__globals__;pfx=self.bm['up'];old_b48=pfx.forward
  def pref(source,skip,slab,hooks=None):
   r=old_b48(source,skip,slab,hooks);pre=pfx.num._rn_half(r['nearest'].astype(np.float64)+r['skip']*r['gamma'].astype(np.float64),'b48 HFMA');codes=pfx.num.fp8_rn_satfinite(pre);physical=np.empty_like(r['physical']);physical[pfx.C_MAP]=codes;r.update(prequantization=pre,output=codes,physical=physical);return r
  pfx.forward=pref;norm_patch(self.bm['norm'])
  # Post fused prefix and recovered SM120 HMMA.
  post=dm.post;p=post.nn.prefix;old_post=p.forward
  def postpref(main,skip,slab,*,h,w,ox,oy,cta):
   r=old_post(main,skip,slab,h=h,w=w,ox=ox,oy=oy,cta=cta);z=p.ffn.num._rn_half(r['skip_half'].astype(np.float64)*r['skip_gamma'].astype(np.float64)+r['scaled_main'].astype(np.float64),'post HFMA');logical=np.empty((64,32),np.float16);logical[p.TILE.ravel()]=z.reshape(64,32);r.update(combined_grid=z,combined_half=logical);return r
  p.forward=postpref
  def headfn(features,slab):
   w=np.frombuffer(slab,dtype='<u2')[post.nn.head.WEIGHT_MAP//2].copy();a=features.view(np.uint16);first=HMMA.mma_bits(a[:,:16],w[:16]);last=HMMA.mma_bits(a[:,16:],w[16:],first);return dict(weight_bits=w,weight_half=w.view(np.float16),before={0:np.zeros_like(first),16:first.copy()},after={0:first.copy(),16:last.copy()},computed_half=last.view(np.float16),live_half=last[:,:4].copy().view(np.float16))
  post.nn.head.forward=headfn
  with checkpoint.open('rb') as f:
   hn=struct.unpack('<Q',f.read(8))[0];head=json.loads(f.read(hn));lo,hi=head['block39.layer0.layer']['data_offsets'];f.seek(8+hn+lo);self.slab39=f.read(hi-lo)
  if len(self.slab39)!=525312:raise ValueError('block39 slab')
  lutdir=Path(lut_dir);self.div_lut=_read_lut(lutdir/'APPROX_PHASE1_INPUT.bin',lutdir/'APPROX_PHASE1_OUTPUT.bin')
  # Isolated live-tail module.
  saved=sys.modules.get('f32bits');oldpath=sys.path[:]
  try:
   sys.modules.pop('f32bits',None);sys.path.insert(0,str(ROOT/'nr-block70-live-tex-main-20260914'));self.live=load('_package_padded_live',ROOT/'nr-block70-live-tex-main-20260914/live_tex.py')
  finally:
   sys.path[:]=oldpath
   if saved is None:sys.modules.pop('f32bits',None)
   else:sys.modules['f32bits']=saved
 def _block39(self,source,skip):
  M=96;OH,OW=16,20;mi=np.arange(M)[:,None];A=D39.axes.a_offset(mi,np.arange(1024)[None,:],1024);B=D39.axes.b_offset(np.arange(1024)[:,None],np.arange(512)[None,:],512);yy,xx=np.indices((OH,OW));gm=16*((yy//4)*(OW//4)+xx//4)+4*(yy%4)+xx%4;C=D39.axes.c_offset(gm.reshape(-1,1),np.arange(512)[None,:],512);aa=D39.num.decode_e4m3(source,A);ww=D39.num.decode_e4m3(self.slab39,B);res=D39.num.decode_e4m3(skip,C);coef=np.frombuffer(self.slab39,'<f2',offset=524288).copy();combined=None
  for z in range(4):
   carry=np.zeros((M,512),np.float16)
   for k0 in range(z*256,(z+1)*256,32):carry=D39.num._rn_half(aa[:,k0:k0+32]@ww[k0:k0+32]+carry.astype(np.float64),'d39 K32')
   combined=carry if combined is None else D39.num._rn_half(combined.astype(np.float64)+carry.astype(np.float64),'d39 zadd')
  up=np.empty((OH,OW,512),np.float16)
  for y in range(OH):
   for x in range(OW):sy,sx=y//2,x//2;lm=16*((sy//4)*3+sx//4)+4*(sy%4)+sx%4;up[y,x]=combined[lm]
  pre=D39.num._rn_half(up.reshape(-1,512).astype(np.float64)+res*coef.astype(np.float64),'d39 fused HFMA');codes=D39.num.fp8_rn_satfinite(pre);phys=np.empty(OH*OW*512,np.uint8);phys[C]=codes;return phys.tobytes()
 def _block48(self,source,skip):
  b48=self.b48;local=self.local;coff=b48.forward.__func__.__globals__['coff'];poff=b48.forward.__func__.__globals__['poff'];out=bytearray(32*36*256);seen=set()
  for cy in range(4):
   for cx in range(5):
    lp=bytearray(8192);sp=bytearray(16384)
    for y in range(4):
     for x in range(4):
      sy,sx=4*cy+y,4*cx+x
      if sy<16 and sx<18:
       for c in range(512):lp[poff(y,x,c,4,4,512)]=source[poff(sy,sx,c,16,20,512)]
    for y in range(8):
     for x in range(8):
      sy,sx=8*cy+y,8*cx+x
      if sy<32 and sx<36:
       for c in range(256):sp[coff(y,x,c,8,8,256)]=skip[coff(sy,sx,c,32,36,256)]
    z=local.forward_packed(bytes(lp),bytes(sp))['output_packed']
    for y in range(8):
     for x in range(8):
      sy,sx=8*cy+y,8*cx+x
      if sy<32 and sx<36:
       for c in range(256):d=coff(sy,sx,c,32,36,256);out[d]=z[coff(y,x,c,8,8,256)];seen.add(d)
  if len(seen)!=len(out):raise ValueError('block48 coverage')
  return bytes(out)
 def forward(self,vit38_2d,retained30,skip22,skip14,skip8,skip4,skip0,color):
  spans=((vit38_2d,98304),(retained30,163840),(skip22,294912),(skip14,589824),(skip8,1179648),(skip4,2359296),(skip0,9437184))
  if any(type(x)is not bytes or len(x)!=n for x,n in spans):raise ValueError('padded decoder input spans')
  if not isinstance(color,np.ndarray) or color.dtype!=np.float32 or color.shape!=(512,512,4) or not np.all(np.isfinite(color)):raise ValueError('color')
  dm=self.dm
  families=(dm.f8,dm.o8,dm.u4,dm.f4,dm.o4,dm.u2,dm.f2,dm.o2,dm.u1,dm.f1,dm.o1,dm.post)
  for f in families:_walk(f.nn)
  # Recursive short-name graphs overlap; apply authoritative direct bindings
  # only after every recursive walk has finished.
  for f in families:
   f.nn.geometry=self._geometry
   if hasattr(f.nn,'extents'):f.nn.extents=self._shape
   if hasattr(f.nn,'engine'):f.nn.engine.geometry=self._geometry
   norm_patch(f.normalized)
  x=self._block39(vit38_2d,retained30);orig={40:(0,0),41:(-4,-4),42:(-4,0),43:(0,-4),44:(0,0),45:(-4,-4),46:(-4,0),47:(0,-4)}
  for b in range(40,48):
   z=dm.split.blocks[b].forward_shifted(x,16,20,orig[b],rsqrt=dm.split.rs,reciprocal=dm.split.rc)
   if b==47:pi=self.sg['pi_index'](512);x=z['layers'][3]['logical'][:,pi].reshape(16,20,32,16).transpose(2,0,1,3).copy().tobytes()
   else:x=z['output']
  x=self._block48(x,skip22)
  for b,(ox,oy) in {49:(-4,-4),50:(-4,0),51:(0,-4),52:(0,0),53:(-4,-4),54:(-4,0),55:(0,-4)}.items():f=dm.f8 if b<55 else dm.o8;x=f.nn.forward(x,dm.slab[b],h=32,w=36,x=ox,y=oy,rsqrt=f.rs,reciprocal=f.rc)['output']
  x=dm.u4.nn.forward(x,skip14,dm.slab[56],h=64,w=72,rsqrt=dm.u4.rs,reciprocal=dm.u4.rc)
  for b,(ox,oy) in {57:(0,-4),58:(0,0),59:(-4,-4),60:(-4,0),61:(0,-4)}.items():f=dm.f4 if b<61 else dm.o4;x=f.nn.forward(x,dm.slab[b],h=64,w=72,x=ox,y=oy,rsqrt=f.rs,reciprocal=f.rc)['output']
  x=dm.u2.nn.forward(x,skip8,dm.slab[62],h=128,w=144,rsqrt=dm.u2.rs,reciprocal=dm.u2.rc)
  for b,(ox,oy) in {63:(-4,-4),64:(-4,0),65:(0,-4)}.items():f=dm.f2 if b<65 else dm.o2;x=f.nn.forward(x,dm.slab[b],h=128,w=144,x=ox,y=oy,rsqrt=f.rs,reciprocal=f.rc)['output']
  x=dm.u1.nn.forward(x,skip4,dm.slab[66],h=256,w=288,x=0,y=0,rsqrt=dm.u1.rs,reciprocal=dm.u1.rc)['output']
  for b,(ox,oy) in {67:(-4,-4),68:(-4,0),69:(0,-4)}.items():f=dm.f1 if b<69 else dm.o1;x=f.nn.forward(x,dm.slab[b],h=256,w=288,x=ox,y=oy,rsqrt=f.rs,reciprocal=f.rc)['output']
  z=dm.post.nn.forward(x,skip0,dm.slab[70],h=512,w=576,ox=-4,oy=-4,rsqrt=dm.post.rs,reciprocal=dm.post.rc);learn=z['learned_RGB_logit_half'][:,:512].copy();live=self.live;lut=self.div_lut
  class Div(live.f.SyntheticRNApprox):
   def __init__(self):super().__init__('SM120 public H512 DIV LUT')
   def div_approx(self,a,b):
    key=(0,live.f.ftz(a),live.f.ftz(b))
    if key not in lut:raise live.f.F32DomainError('DIV domain miss')
    return lut[key]
  texbits=np.ascontiguousarray(color).view(np.uint32).copy()
  class Fetch:
   def fetch(self,slot,mode,u,v):
    if slot!='A' or mode!=2:raise ValueError('unexpected resource')
    xf=struct.unpack('<f',struct.pack('<I',u))[0];yf=struct.unpack('<f',struct.pack('<I',v))[0];xx=min(511,max(0,int(np.floor(xf*512))));yy=min(511,max(0,int(np.floor(yf*512))));return tuple(map(int,texbits[yy,xx]))
  cfg=live.FrameConfig(source_width=512,source_height=512,origin_x=-4,origin_y=-4,rgb_multiplier_bits=0x3d000000,decode_enabled_u32=1,blend_half_bits=None,a_base_x_bits=0,a_base_y_bits=0,a_scale_x_bits=0x44000000,a_scale_y_bits=0x44000000,a_post_x_bits=0x3b000000,a_post_y_bits=0x3b000000);rgba=live.forward_half(learn,config=cfg,textures=live.LiveTextures(a=Fetch()),approx=Div());return {'output_RGBA_f32_bits':rgba,'learned_RGB_logit_half':learn}
