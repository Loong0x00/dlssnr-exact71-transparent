"""Post70 source-derived A32 low nearest2x with main/skip half scales, no prefixMMA."""
import numpy as np
import ffn
PI=ffn.pi(np.arange(32));MAIN_CHANNEL=np.argsort(PI)
y,x=np.meshgrid(np.arange(8),np.arange(8),indexing='ij');TILE=16*((y//4)*2+x//4)+4*(y%4)+x%4

def crops(main,skip,*,h,w,ox,oy,cta):
 if type(main)is not bytes or len(main)!=h//2*(w//2)*32 or type(skip)is not bytes or len(skip)!=h*w*32:raise ValueError('exact A32low/C32high byte spans')
 cx,cy=cta;yy=oy+8*cy+y;xx=ox+8*cx+x;ly=(oy+8*cy)//2+y//2;lx=(ox+8*cx)//2+x//2;lowvalid=(ly>=0)&(ly<h//2)&(lx>=0)&(lx<w//2);highvalid=(yy>=0)&(yy<h)&(xx>=0)&(xx<w)
 low=np.zeros((8,8,32),dtype=np.uint8);hi=np.zeros_like(low);view=np.frombuffer(main,dtype=np.uint8).reshape(2,h//2,w//2,16).transpose(1,2,0,3).reshape(h//2,w//2,32);low[lowvalid]=view[ly[lowvalid],lx[lowvalid]][:,MAIN_CHANNEL]
 gm=16*((yy//4)*(w//4)+xx//4)+4*(yy%4)+xx%4;mp=ffn.axes.c_offset(gm[:,:,None],np.arange(32)[None,None,:],32);hi[highvalid]=np.frombuffer(skip,dtype=np.uint8)[mp[highvalid]]
 return low,hi

def forward(main,skip,slab,*,h,w,ox,oy,cta):
 lo,sk=crops(main,skip,h=h,w=w,ox=ox,oy=oy,cta=cta);d=lambda a:ffn.num.decode_e4m3(a.tobytes(),np.arange(2048).reshape(8,8,32));a=d(lo);b=d(sk);gm=np.frombuffer(slab,dtype='<f2',count=32,offset=8272).copy();gs=np.frombuffer(slab,dtype='<f2',count=32,offset=8336).copy()
 if not np.all(np.isfinite(gm)) or not np.all(np.isfinite(gs)):raise ValueError('finite input gammas')
 am=ffn.num._rn_half(a*gm.astype(np.float64),'main nearest scale');bs=ffn.num._rn_half(b*gs.astype(np.float64),'highskip scale');p=ffn.num._rn_half(am.astype(np.float64)+bs.astype(np.float64),'main+skip actual half add');logical=np.empty((64,32),dtype=np.float16);logical[TILE.ravel()]=p.reshape(64,32)
 return dict(main_fp8=lo,skip_fp8=sk,main_half=a.astype(np.float16),skip_half=b.astype(np.float16),main_gamma=gm,skip_gamma=gs,scaled_main=am,scaled_skip=bs,combined_grid=p,combined_half=logical)
