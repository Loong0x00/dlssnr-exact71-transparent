"""Third candidate: source-proven first-matrix row basis and activation E4M3 publish."""
import numpy as np
from candidate import *
def pi_index(n):
 k=np.arange(n);return 16*(k//16)+2*((k%16)//4)+8*((k%4)//2)+k%2
def packed_a_offset(h,w,y,x,c):
 m=4*(y%4)+x%4;k=c%32;lane=4*(m%8)+(k%16)//4;i=8*(k//16)+4*(m//8)+k%4
 return 8192*((y//4)*(w//4)+x//4)+1024*(c//64)+512*((c%64)//32)+16*lane+i
def unpack_a(buf,h,w):
 if type(buf)is not bytes or len(buf)!=h*w*512:raise ValueError('packedA512')
 out=np.empty((h*w,512),np.uint8)
 for y in range(h):
  for x in range(w):
   for c in range(512):out[y*w+x,c]=buf[packed_a_offset(h,w,y,x,c)]
 return out
def unpack_c(buf,h,w):
 if type(buf)is not bytes or len(buf)!=h*w*512:raise ValueError('packedC512')
 out=np.empty((h*w,512),np.uint8)
 for y in range(h):
  for x in range(w):
   for c in range(512):out[y*w+x,c]=buf[packed_c_offset(h,w,y,x,c)]
 return out
class HalfLut:
 def __init__(self,data,kind):
  import hashlib
  pins={'rsqrt':'a43128fa609f3f3a3e2538726fcf6feac5f950b9d5ef8eb523df3fbf43c96011','reciprocal':'1254eed4e932af3deb7894a185a159e4b95f863aa353afbabc603a3ba2abd53b'}
  if type(data)is not bytes or len(data)!=262144 or hashlib.sha256(data).hexdigest()!=pins[kind]:raise ValueError('pinned primitive LUT')
  self.table=np.frombuffer(data,dtype='<u4');self.kind=kind
 @classmethod
 def load(cls,kind):
  path='nr-native-normalization-20260911/runs/20260911-134003/rsqrt-f32.bin' if kind=='rsqrt' else 'nr-native-fp8-mma-20260911/runs/20260911-132533/reciprocal-f32.bin'
  return cls((V/path).read_bytes(),kind)
 def evaluate(self,x):
  if x.dtype!=np.float16:raise ValueError('half LUT domain')
  bits=x.view(np.uint16);raw=self.table[bits].copy();return raw,num._rn_half(raw.view(np.float32).astype(np.float64),self.kind+' f32 to half')
def _norm_tree(x,rsqrt):
 rn=num._rn_half;sq=rn(x.astype(np.float64)**2,'split norm square');pa=rn(sq[...,8:16].astype(np.float64)+sq[...,24:32].astype(np.float64),'split norm pairA');pb=rn(sq[...,0:8].astype(np.float64)+sq[...,16:24].astype(np.float64),'split norm pairB');local=rn(pa.astype(np.float64)+pb.astype(np.float64),'split norm local').reshape(64,16,4,2);b2=rn(local.astype(np.float64)+local[:,:,np.arange(4)^2,:].astype(np.float64),'split norm butterfly2');b1=rn(b2.astype(np.float64)+b2[:,:,np.arange(4)^1,:].astype(np.float64),'split norm butterfly1');den=rn(b1[...,0].astype(np.float64)+b1[...,1].astype(np.float64),'split norm den');eps=np.array([948045311],np.uint32).view(np.float32).astype(np.float16)[0];floor=np.maximum(den,eps).astype(np.float16);raw,factor=rsqrt.evaluate(floor);normalized=rn(x.astype(np.float64)*factor[...,0,None].astype(np.float64),'split norm factor');return dict(square=sq,denominator=den,floor=floor,raw=raw,normalized=normalized)
def _logical_bias(raw):
 flat=raw.reshape(16,-1);idx=[]
 for q in range(64):
  for k in range(64):
   qy,qx=divmod(q,8);ky,kx=divmod(k,8);b=lambda v,p:(v>>p)&1
   idx.append((b(qy,2)<<11)|(b(qx,2)<<10)|(b(ky,2)<<9)|(b(kx,2)<<8)|(b(qy,0)<<7)|(b(qx,1)<<6)|(b(qx,0)<<5)|(b(ky,0)<<4)|(b(kx,1)<<3)|(b(ky,1)<<2)|(b(qy,1)<<1)|b(kx,0))
 return flat[:,idx].reshape(16,64,64)
def _softmax(score,reciprocal):
 rn=num._rn_half;constants=np.array([1027077105,1067877303,1065615360,1070129152],np.uint32).view(np.float32).astype(np.float16);aff=rn(score.astype(np.float64)*float(constants[0])+float(constants[1]),'split softmax affine');clamp=np.minimum(np.maximum(aff,constants[2]),constants[3]).astype(np.float16);pairs=clamp.view(np.uint16).reshape(16,64,32,2).astype(np.uint32);packed=pairs[...,0]|(pairs[...,1]<<16);word=((((packed.astype(np.uint64)<<5)&0xffffffff)+2146992128)&0xffffffff).astype(np.uint32);bits=np.stack([word&65535,word>>16],axis=-1).astype(np.uint16).reshape(16,64,64);mapped=bits.view(np.float16);add=lambda x,y,n:rn(x.astype(np.float64)+y.astype(np.float64),n);base=np.array([0,1,2,3,8,9,10,11,16,17,18,19,24,25,26,27,4,5,6,7,12,13,14,15,20,21,22,23,28,29,30,31]);ordered=mapped[...,np.r_[base,base+32]];pair=[add(ordered[...,i:i+8],ordered[...,i+8:i+16],'split softmax pair') for i in (0,16,32,48)];local=add(add(add(pair[0],pair[1],'split local01'),pair[2],'split local012'),pair[3],'split local0123').reshape(16,64,4,2);lanes=add(add(add(local[...,0,:],local[...,1,:],'split lanes01'),local[...,2,:],'split lanes012'),local[...,3,:],'split lanes0123');den=add(lanes[...,0],lanes[...,1],'split denominator');eps=np.array([948045311],np.uint32).view(np.float32).astype(np.float16)[0];floor=np.maximum(den,eps).astype(np.float16);raw,inv=reciprocal.evaluate(floor);ph=rn(mapped.astype(np.float64)*inv[...,None].astype(np.float64),'split probability mul');codes=num.fp8_rn_satfinite(ph);return codes,dict(mapped=mapped,denominator=den,inverse=inv,probability_half=ph,raw=raw)
class Block23CandidateV3(Block23Candidate):
 def __init__(self,path=P/'block23-logical.npz'):
  super().__init__(path);self.w['first_projection_weight']=self.w['first_projection_weight'][pi_index(512)]
 def layer0(self,source,h,w):
  xcodes=unpack_planar(source,h,w);x=num.decode_e4m3(xcodes.tobytes(),np.arange(xcodes.size).reshape(xcodes.shape));hidden=matmul_k32(x,self.w['first_projection_weight'],label='split23 first K32');hidden_codes,hidden=quant_decode(hidden);groups=[];details=[]
  for g in range(8):
   e=matmul_k32(hidden[:,64*g:64*(g+1)],self.w['group_expand_weight'][g],label='split23 expand K32');act_pre=activation(e);act_codes,act=quant_decode(act_pre);o=matmul_k32(act,self.w['group_project_weight'][g],label='split23 project K32');q,d=quant_decode(o);groups.append(q);details.append(dict(expand=e,activation_prequantization=act_pre,activation_codes=act_codes,prequantization=o))
  logical=np.concatenate(groups,axis=1);return dict(output=pack_c(logical,h,w),logical=logical,first_prequantization=hidden,first_codes=hidden_codes,groups=details)
 def layer1(self,source,skip,h,w):
  codes=unpack_a(source,h,w);x=num.decode_e4m3(codes.tobytes(),np.arange(codes.size).reshape(codes.shape));skip_codes=unpack_planar(skip,h,w);sf=num.decode_e4m3(skip_codes.tobytes(),np.arange(skip_codes.size).reshape(skip_codes.shape));inv=np.argsort(pi_index(512));gamma=self.w['ffn_cos_skip'].astype(np.float64);seed=num._rn_half(sf[:,inv]*gamma[None,:],'split23 layer1 residual mul');pre=matmul_k32(x,self.w['weight3'][pi_index(512)],seed=seed,label='split23 layer1 K32');q,d=quant_decode(pre);return dict(output=pack_c(q,h,w),logical=q,prequantization=pre,seed=seed)
 def layer2(self,source,h,w,*,rsqrt,reciprocal):
  if (h,w) not in ((4,4),(8,8)):raise ValueError('split23 bounded H4/H8')
  compact=unpack_a(source,h,w);window=np.zeros((64,512),np.uint8);token_stride=8 if w==8 else w
  for y in range(h):
   for x in range(w):window[token_stride*y+x]=compact[w*y+x]
  a=num.decode_e4m3(window.tobytes(),np.arange(window.size).reshape(window.shape));raw=matmul_k32(a,self.w['qkv_weight'][pi_index(512)],label='split23 qkv K32').reshape(64,3,16,32).transpose(0,2,1,3);qtree=_norm_tree(raw[:,:,0],rsqrt);ktree=_norm_tree(raw[:,:,1],rsqrt);scale=num._rn_half(self.w['attn_scale'].astype(np.float64),'split23 Q scale f32 to half');qh=num._rn_half(qtree['normalized'].astype(np.float64)*scale[None,:,None].astype(np.float64),'split23 Q scale mul');kh=ktree['normalized'];vh=raw[:,:,2];qc=num.fp8_rn_satfinite(qh);kc=num.fp8_rn_satfinite(kh);vc=num.fp8_rn_satfinite(vh);decode=lambda x:num.decode_e4m3(x.tobytes(),np.arange(x.size).reshape(x.shape)).transpose(1,0,2);q,k,v=map(decode,(qc,kc,vc));idx=pi_index(32);bias=_logical_bias(self.w['attn_bias']);score=np.stack([num._rn_half(q[head][:,idx]@k[head][:,idx].T+bias[head].astype(np.float64),'split23 QK seededK32') for head in range(16)]);prob,soft=_softmax(score,reciprocal);pv_in=num.decode_e4m3(prob.tobytes(),np.arange(prob.size).reshape(prob.shape));pv=np.zeros((16,64,32),np.float16)
  for k0 in (0,32):
   ii=pi_index(64)[k0:k0+32];pv=np.stack([num._rn_half(pv_in[head][:,ii]@v[head][ii]+pv[head].astype(np.float64),'split23 PV K32') for head in range(16)])
  flat=pv.transpose(1,0,2).reshape(64,512);codes=num.fp8_rn_satfinite(flat);compact_out=np.empty((h*w,512),np.uint8)
  for y in range(h):
   for x in range(w):compact_out[y*w+x]=codes[token_stride*y+x]
  return dict(output=pack_c(compact_out,h,w),logical=compact_out,raw_projected=raw,Q=qc,K=kc,V=vc,score=score,probability=prob,PV_half=pv,normalization=(qtree,ktree),softmax=soft)
 def layer3(self,source,skip,h,w):
  codes=unpack_a(source,h,w);x=num.decode_e4m3(codes.tobytes(),np.arange(codes.size).reshape(codes.shape));skip_codes=unpack_c(skip,h,w);sf=num.decode_e4m3(skip_codes.tobytes(),np.arange(skip_codes.size).reshape(skip_codes.shape));gamma=self.w['attn_cos_skip'].astype(np.float64);seed=num._rn_half(sf*gamma[None,:],'split23 layer3 residual mul');pre=matmul_k32(x,self.w['projection_weight'][pi_index(512)],seed=seed,label='split23 layer3 K32');q,d=quant_decode(pre);return dict(output=pack_c(q,h,w),logical=q,prequantization=pre,seed=seed)
 def forward(self,source,h,w,*,rsqrt,reciprocal):
  l0=self.layer0(source,h,w);l1=self.layer1(l0['output'],source,h,w);l2=self.layer2(l1['output'],h,w,rsqrt=rsqrt,reciprocal=reciprocal);l3=self.layer3(l2['output'],l1['output'],h,w);return dict(output=l3['output'],layers={0:l0,1:l1,2:l2,3:l3})
