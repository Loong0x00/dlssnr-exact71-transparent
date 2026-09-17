import numpy as np
import prefix,ffn,qkv,normalized,attention
def maps(h,w,cx,cy):
 lh,lw=h//2,w//2;off=np.arange(2048);kg=off//256;rem=off%256;yy=rem//64+4*cy;xx=(rem%64)//16+4*cx;c=rem%16;lm=16*((kg*lh+yy)*lw+xx)+c;off=np.arange(4096);tile=off//1024;ty=tile//2+2*cy;tx=tile%2+2*cx;hm=1024*(ty*(w//4)+tx)+off%1024;return lm,hm
def window(low,skip,slab,h,w,cx,cy,rsqrt,reciprocal):
 lm,hm=maps(h,w,cx,cy);lo=np.frombuffer(low,np.uint8)[lm].tobytes();sk=np.frombuffer(skip,np.uint8)[hm].tobytes();p=prefix.forward(lo,sk,slab);physical=np.empty(4096,np.uint8);physical[prefix.C_MAP]=prefix.num.fp8_rn_satfinite(p['prequantization']);f=ffn.forward(physical.tobytes(),slab);r=qkv.forward_raw(f['physical'].tobytes(),slab);n=normalized.forward(r['warp_bank_view'],slab,rsqrt=rsqrt);a=attention.forward(n['Q'],n['K'],n['V'],f['physical'].tobytes(),slab,reciprocal=reciprocal);return a['physical'].tobytes(),hm
def forward(low,skip,slab,*,h=128,w=128,rsqrt,reciprocal):
 if type(low)is not bytes or len(low)!=(h//2)*(w//2)*128 or type(skip)is not bytes or len(skip)!=h*w*64:raise ValueError('H128 spans')
 out=np.empty(h*w*64,np.uint8);seen=np.zeros(len(out),bool)
 for cy in range(h//8):
  for cx in range(w//8):
   z,mp=window(low,skip,slab,h,w,cx,cy,rsqrt,reciprocal);assert not np.any(seen[mp]);out[mp]=np.frombuffer(z,np.uint8);seen[mp]=True
 assert np.all(seen);return bytes(out)
