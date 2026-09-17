"""Transparent first-counter pre0 NN, normal two-output family; no interpreter forward.
Explicit sampler/non-native ingress approximation and measured learned-math policies.
No padded/odd-output extension, live resource, native arithmetic or fullNR claim.
"""
import struct
import numpy as np
import preprocess_candidate as pre
import embedding,ffn,qkv,normalized,attention
Y,X=np.indices((8,8));TILE=16*((Y//4)*2+X//4)+4*(Y%4)+X%4
PI=ffn.pi(np.arange(32))

def geometry(args):
 if type(args)is not bytes or len(args)!=264:raise ValueError('own264ABI')
 h,w=struct.unpack_from('<2i',args,240);ah,aw=struct.unpack_from('<2i',args,256)
 if not (8<=h<=128 and 8<=w<=128 and h%8==w%8==0 and (ah,aw)==(h//2,w//2)):raise ValueError('bounded no-padding positive multiples8, exact half extents')
 if struct.unpack_from('<Q',args,200)[0] or any(struct.unpack_from('<4Q',args,8)):raise ValueError('firstcounter0 and explicit absentT1..T4 family only')
 return h,w

def reduce_half(final):
 """Actual balanced pair order depends on destination lane (low y bit0/x bit1)."""
 grid=final[TILE];out=np.empty((4,4,32),dtype=np.float16);rn=ffn.num._rn_half
 for y in range(4):
  for x in range(4):
   dy=y%2;dx=x//2
   a=grid[2*y+dy,2*x+dx];b=grid[2*y+dy,2*x+1-dx];c=grid[2*y+1-dy,2*x+dx];d=grid[2*y+1-dy,2*x+1-dx]
   ab=rn(a.astype(np.float64)+b.astype(np.float64),'downsample pairAB');cd=rn(c.astype(np.float64)+d.astype(np.float64),'downsample pairCD');total=rn(ab.astype(np.float64)+cd.astype(np.float64),'downsample pair sum');out[y,x]=rn(total.astype(np.float64)*.25,'downsample quarter')
 return out

def forward_window(args,slab,*,cta,sampler,approx,rsqrt,reciprocal):
 features=pre.preprocess_firstcounter0(args,sampler=sampler,approx=approx,cta=cta);grid=np.asarray(features.logical,dtype=np.uint16).transpose(1,0,2).copy().view(np.float16);inputs=np.empty((64,16),dtype=np.float16);inputs[TILE.ravel()]=grid.reshape(64,16);e=embedding.forward(inputs,slab);f=ffn.forward(e['output_half'],slab);r=qkv.forward_raw(f['physical'].tobytes(),slab);n=normalized.forward(r['warp_bank_view'],slab,rsqrt=rsqrt);a=attention.forward(n['Q'],n['K'],n['V'],f['prequantization'],slab,reciprocal=reciprocal)
 high=ffn.num.fp8_rn_satfinite(a['prequantization']);low=reduce_half(a['prequantization']);lc=ffn.num.fp8_rn_satfinite(low);lowA=lc[:,:,PI];packed_low=lowA.reshape(4,4,2,16).transpose(2,0,1,3).copy();packed_high=np.empty(2048,dtype=np.uint8);packed_high[ffn.C_MAP]=high
 return dict(features=features,embedding=e,ffn=f,raw=r,norm=n,attention=a,HIGH=packed_high,LOW_LOCAL_PLANAR=packed_low,high_grid_C=high[TILE],low_grid_A=lowA,low_half_C=low)

def forward(args,slab,*,sampler,approx,rsqrt,reciprocal,keep_stages=False):
 h,w=geometry(args);high=np.empty(h*w*32,dtype=np.uint8);low=np.empty((2,h//2,w//2,16),dtype=np.uint8);high_seen=np.zeros(h*w*32,dtype=bool);low_seen=np.zeros((h//2,w//2),dtype=bool);stages={}
 for cy in range(h//8):
  for cx in range(w//8):
   z=forward_window(args,slab,cta=(cx,cy),sampler=sampler,approx=approx,rsqrt=rsqrt,reciprocal=reciprocal);yy=8*cy+Y;xx=8*cx+X;gm=16*((yy//4)*(w//4)+xx//4)+4*(yy%4)+xx%4;indices=ffn.axes.c_offset(gm[:,:,None],np.arange(32)[None,None,:],32);assert not high_seen[indices].any();high[indices]=z['high_grid_C'];high_seen[indices]=True
   ys=slice(4*cy,4*cy+4);xs=slice(4*cx,4*cx+4);assert not low_seen[ys,xs].any();low[:,ys,xs]=z['LOW_LOCAL_PLANAR'];low_seen[ys,xs]=True
   if keep_stages:stages[cx,cy]=z
 assert high_seen.all() and low_seen.all()
 return dict(skip_C_packed=high,primary_A_planar=low,stages=stages)
