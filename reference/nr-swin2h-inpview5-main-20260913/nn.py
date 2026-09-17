"""Block5 2H INPVIEW: exact four-planar16 input adapter plus verified2H core.
No VM/DLL in forward; source80/84 overrides explicit, host default zero.
"""
import numpy as np
import standard_nn,ffn
Y,X=np.indices((8,8));TILE=16*((Y//4)*2+X//4)+4*(Y%4)+X%4

def serialized_rank(c):
 c=np.asarray(c);return 16*(c//16)+4*((c%8)//2)+2*((c//8)%2)+c%2

def source_shape(h,w,source_extent):
 eh,ew=source_extent;eh=eh if eh>0 else h;ew=ew if ew>0 else w
 if not(1<=eh<=32 and 1<=ew<=32):raise ValueError('bounded effective source dimensions')
 return eh,ew

def ingress(features,*,h,w,x,y,cta,source_extent=(0,0)):
 eh,ew=source_shape(h,w,source_extent)
 if type(features)is not bytes or len(features)!=eh*ew*64:raise ValueError('exact four-planar16 C64 source bytes')
 cx,cy=cta;yy=y+8*cy+Y;xx=x+8*cx+X;valid=((eh==1)|((yy>=0)&(yy<eh)))&((ew==1)|((xx>=0)&(xx<ew)));sy=np.zeros_like(yy) if eh==1 else yy;sx=np.zeros_like(xx) if ew==1 else xx;view=np.frombuffer(features,np.uint8).reshape(4,eh,ew,16).transpose(1,2,0,3).reshape(eh,ew,64);grid=np.zeros((8,8,64),np.uint8);grid[valid]=view[sy[valid],sx[valid]][:,serialized_rank(np.arange(64))];rows=np.empty((64,64),np.uint8);rows[TILE.ravel()]=grid.reshape(64,64);packed=np.empty(4096,np.uint8);packed[ffn.C_MAP]=rows;return packed

def forward(features,slab,*,h=32,w=32,x=0,y=0,source_extent=(0,0),rsqrt,reciprocal,keep_stages=False):
 gx,gy=standard_nn.geometry(h,w,x,y);out=np.empty(h*w*64,np.uint8);seen=np.zeros(len(out),bool);stages={}
 for cy in range(gy):
  for cx in range(gx):
   a=ingress(features,h=h,w=w,x=x,y=y,cta=(cx,cy),source_extent=source_extent);z=standard_nn.forward_window(a.tobytes(),slab,rsqrt=rsqrt,reciprocal=reciprocal);mp=standard_nn.tile_sources(h,w,x,y,cx,cy);valid=mp>=0
   if seen[mp[valid]].any():raise ValueError('duplicate output bytes')
   out[mp[valid]]=z['physical'][valid];seen[mp[valid]]=True
   if keep_stages:z['ingress']=a;stages[cx,cy]=z
 if not seen.all():raise ValueError('missing output bytes')
 return dict(output=out.tobytes(),stages=stages)
