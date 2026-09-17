"""Post70 transparent NN + explicit optional-null-texture RGB branch.
Not an image-input/fullNR/native surface or history implementation.
"""
import numpy as np
import prefix,ffn,qkv,normalized,attention,head,f32_ops

def geometry(h,w,ox,oy):
 if any(type(v)is not int for v in (h,w,ox,oy)) or not(8<=h<=512 and 8<=w<=512 and h%8==w%8==0 and ox in (0,-4) and oy in (0,-4)):raise ValueError('bounded highH/Wmultiples8 and origins0/-4')
 return (w-ox+7)//8,(h-oy+7)//8

def no_texture_rgb(learned):
 if learned.dtype!=np.float16 or learned.shape!=(64,4):raise ValueError('64xRGB/logit half')
 result=np.empty((64,4),dtype=np.uint32)
 for m in range(64):
  for c in range(3):
   v=f32_ops.as_bits(float(learned[m,c]));v=f32_ops.apply('mul.ftz.f32',[v,0x3d000000]);v=f32_ops.apply('fma.rn.ftz.f32',[v,0x41000000,0x3f000000]);v=f32_ops.apply('max.ftz.f32',[v,0]);v=f32_ops.apply('min.ftz.f32',[v,0x3f800000]);result[m,c]=v
  result[m,3]=0x3f800000
 return result

def forward_window(main,skip,slab,*,h,w,ox,oy,cta,rsqrt,reciprocal):
 p=prefix.forward(main,skip,slab,h=h,w=w,ox=ox,oy=oy,cta=cta);f=ffn.forward(p['combined_half'],slab);r=qkv.forward_raw(f['physical'].tobytes(),slab);n=normalized.forward(r['warp_bank_view'],slab,rsqrt=rsqrt);a=attention.forward(n['Q'],n['K'],n['V'],f['prequantization'],slab,reciprocal=reciprocal);hd=head.forward(a['prequantization'],slab);color=no_texture_rgb(hd['live_half'])
 return dict(prefix=p,ffn=f,raw=r,norm=n,attention=a,head=hd,RGBA_f32_bits=color)

def forward(main,skip,slab,*,h=128,w=128,ox=-4,oy=-4,rsqrt,reciprocal,keep_stages=False):
 gx,gy=geometry(h,w,ox,oy);out=np.empty((h,w,4),dtype=np.uint32);learned=np.empty((h,w,4),dtype=np.float16);seen=np.zeros((h,w),dtype=np.bool_);stages={}
 for cy in range(gy):
  for cx in range(gx):
   z=forward_window(main,skip,slab,h=h,w=w,ox=ox,oy=oy,cta=(cx,cy),rsqrt=rsqrt,reciprocal=reciprocal)
   for m in range(64):
    yy=oy+8*cy+4*(m//32)+(m%16)//4;xx=ox+8*cx+4*((m//16)%2)+m%4
    if 0<=yy<h and 0<=xx<w:
     if seen[yy,xx]:raise ValueError('duplicate output pixel')
     seen[yy,xx]=True;out[yy,xx]=z['RGBA_f32_bits'][m];learned[yy,xx]=z['head']['live_half'][m]
   if keep_stages:stages[(cx,cy)]=z
 if not np.all(seen):raise ValueError('missing output pixels')
 return dict(output_RGBA_f32_bits=out,learned_RGB_logit_half=learned,stages=stages)
