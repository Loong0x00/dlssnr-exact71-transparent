"""Base INPVIEW: source planarA/guard/broadcast adapter plus proved ordinary1h core.
NN helper decode/repack is finite254-code identity, NOT a source re-encode claim.
"""
import numpy as np
import standard_nn,ffn
Y,X=np.indices((8,8));TILE=16*((Y//4)*2+X//4)+4*(Y%4)+X%4;INV_PI=np.argsort(ffn.pi(np.arange(32)))

def source_shape(h,w,source_extent):
 hs,ws=source_extent;hs=hs if hs>0 else h;ws=ws if ws>0 else w
 if not(1<=hs<=64 and 1<=ws<=64):raise ValueError('source extent bounds')
 return hs,ws

def ingress(features,*,h,w,x,y,cta,source_extent):
 hs,ws=source_shape(h,w,source_extent)
 if type(features)is not bytes or len(features)!=hs*ws*32:raise ValueError('own exact source planarA bytes')
 cx,cy=cta;yy=y+8*cy+Y;xx=x+8*cx+X;valid=((hs==1)|((yy>=0)&(yy<hs)))&((ws==1)|((xx>=0)&(xx<ws)));sy=np.zeros_like(yy) if hs==1 else yy;sx=np.zeros_like(xx) if ws==1 else xx;view=np.frombuffer(features,np.uint8).reshape(2,hs,ws,16).transpose(1,2,0,3).reshape(hs,ws,32);grid=np.zeros((8,8,32),np.uint8);grid[valid]=view[sy[valid],sx[valid]][:,INV_PI];canonical=np.empty((64,32),np.uint8);canonical[TILE.ravel()]=grid.reshape(64,32);packed=np.empty(2048,np.uint8);packed[ffn.C_MAP]=canonical
 return packed

def forward(features,slab,*,h=64,w=64,x=0,y=0,source_extent=(0,0),rsqrt,reciprocal,keep_stages=False):
 gx,gy=standard_nn.geometry(h,w,x,y);out=np.empty(h*w*32,np.uint8);seen=np.zeros(h*w*32,bool);stages={}
 for cy in range(gy):
  for cx in range(gx):
   a=ingress(features,h=h,w=w,x=x,y=y,cta=(cx,cy),source_extent=source_extent);z=standard_nn.forward_window(a.tobytes(),slab,rsqrt=rsqrt,reciprocal=reciprocal);offset=standard_nn.tile_sources(h,w,x,y,cx,cy);valid=offset>=0;assert not seen[offset[valid]].any();out[offset[valid]]=z['physical'][valid];seen[offset[valid]]=True
   if keep_stages:z['ingress']=a;stages[cx,cy]=z
 assert seen.all();return dict(output=out.tobytes(),stages=stages)
