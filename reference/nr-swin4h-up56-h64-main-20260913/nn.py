"""Generalized source-equation shifted4h upsample at H64."""
import numpy as np
import prefix,ffn,qkv,normalized,attention
def maps(h,w,cx,cy):
 x0=8*cx-4;y0=8*cy;lh,lw=h//2,w//2;off=np.arange(4096);kg=off//256;rem=off%256;ly=rem//64+y0//2;lx=(rem%64)//16+x0//2;c=rem%16;lv=(ly>=0)&(ly<lh)&(lx>=0)&(lx<lw);lm=np.where(lv,16*((kg*lh+ly)*lw+lx)+c,-1)
 off=np.arange(8192);tile=off//2048;ty=tile//2+y0//4;tx=tile%2+x0//4;hv=(ty>=0)&(ty<h//4)&(tx>=0)&(tx<w//4);hm=np.where(hv,2048*(ty*(w//4)+tx)+off%2048,-1);yy,xx=np.meshgrid(np.arange(8)+y0,np.arange(8)+x0,indexing='ij');mask=(yy>=0)&(yy<h)&(xx>=0)&(xx<w);return lm,hm,mask
def window(source,skip,slab,h,w,cx,cy,rsqrt,reciprocal):
 lm,hm,mask=maps(h,w,cx,cy);lo=np.zeros(4096,np.uint8);sk=np.zeros(8192,np.uint8)
 for a,b,m in ((lo,source,lm),(sk,skip,hm)):
  v=m>=0;a[v]=np.frombuffer(b,np.uint8)[m[v]]
 p=prefix.forward(lo.tobytes(),sk.tobytes(),slab);sel=p['prequantization'].copy();sel[~mask]=np.float16(0);codes=prefix.num.fp8_rn_satfinite(sel);physical=np.empty(8192,np.uint8);physical[prefix.C_MAP]=codes;f=ffn.forward(physical.tobytes(),slab);r=qkv.forward_raw(f['physical'].tobytes(),slab);n=normalized.forward(r['warp_bank_view'],slab,rsqrt=rsqrt);a=attention.forward(n['Q'],n['K'],n['V'],f['physical'].tobytes(),slab,reciprocal=reciprocal);return a['physical'].tobytes(),hm
def forward(source,skip,slab,*,h=64,w=64,rsqrt,reciprocal):
 if type(source)is not bytes or len(source)!=(h//2)*(w//2)*256 or type(skip)is not bytes or len(skip)!=h*w*128:raise ValueError('H64 spans')
 gx=(w+11)//8;gy=(h+7)//8;out=np.empty(h*w*128,np.uint8);seen=np.zeros(len(out),bool)
 for cy in range(gy):
  for cx in range(gx):
   z,mp=window(source,skip,slab,h,w,cx,cy,rsqrt,reciprocal);v=mp>=0
   if np.any(seen[mp[v]]):raise ValueError('duplicate')
   out[mp[v]]=np.frombuffer(z,np.uint8)[v];seen[mp[v]]=True
 if not np.all(seen):raise ValueError('missing')
 return bytes(out)
