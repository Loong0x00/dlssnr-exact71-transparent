"""Bounded N64 source-derived attention tensor candidate, CPU only.
NOT standard softmax: half affine/clamp/bit-exp, pre-FP8 half denominator,
FP8 numerator operands, RN-half per K32, measured half-input reciprocal.
"""
from pathlib import Path
import sys,importlib.util,json,hashlib,struct
import numpy as np
HERE=Path(__file__).resolve().parent;V=HERE.parent
sys.path.insert(0,str(V/'nr-vit-expand-tensor-reference-20260912'));import expand_tensor_reference as num
spec=importlib.util.spec_from_file_location('qkv_tensor_maps',HERE/'qkv_tensor.py');qkv=importlib.util.module_from_spec(spec);spec.loader.exec_module(qkv)
SOURCE=V/'nr-vit-attention-main-20260912/original.ptx';SOURCE_SHA='15aa814573504cec154141a6e21e021935f4b5e4cb36176b06ff3fc06acd4de0'
OUTPUT_MAP=qkv.axes.c_offset(np.arange(64)[:,None],np.arange(1024)[None,:],1024)
CONSTANTS_F32=(1035427960,1071303771,1069039616,1073553408)
CONST=np.array([struct.unpack('<f',struct.pack('<I',x))[0] for x in CONSTANTS_F32],dtype=np.float64).astype(np.float16)
EPS=np.array([0x0410],dtype=np.uint16).view(np.float16)[0]
raw=(V/'nr-native-fp8-mma-20260911/runs/20260911-132533/reciprocal-f32.bin').read_bytes()
assert hashlib.sha256(raw).hexdigest()=='1254eed4e932af3deb7894a185a159e4b95f863aa353afbabc603a3ba2abd53b'
RCP_F32=np.frombuffer(raw,dtype='<f4')

def rn(a,label):return num._rn_half(a,label)
def add(a,b,label):return rn(a.astype(np.float64)+b.astype(np.float64),label)
def score_exp(score):
 affine=rn(score.astype(np.float64)*float(CONST[0])+float(CONST[1]),'score affine half FMA')
 clamped=np.minimum(np.maximum(affine,CONST[2]),CONST[3]).astype(np.float16)
 bits=clamped.view(np.uint16).astype(np.uint32)
 if np.any((bits<0x3dc2)|(bits>0x3fe9)):raise ValueError('packed exp carry domain')
 output=((bits<<4)+0x4000).astype(np.uint16).view(np.float16)
 if not np.all(np.isfinite(output)):raise num.UnsupportedNonFinite('bit-exp finite domain')
 return output

def denominator64(p):
 """Literal lane grouping: key pairs(0,8),(16,24),(32,40),(48,56),
 sequential pair accumulations; lane t0+t1+t2+t3, then low+high."""
 if p.shape[-1]!=64 or p.dtype!=np.float16:raise ValueError('exact 64-key half block')
 grouped=p.reshape(*p.shape[:-1],8,4,2)
 a=add(grouped[...,0,:,:],grouped[...,1,:,:],'denom pair0/8')
 b=add(grouped[...,2,:,:],grouped[...,3,:,:],'denom pair16/24')
 c=add(a,b,'denom firsttwo pairs')
 c=add(c,add(grouped[...,4,:,:],grouped[...,5,:,:],'denom pair32/40'),'denom third pair')
 c=add(c,add(grouped[...,6,:,:],grouped[...,7,:,:],'denom pair48/56'),'denom fourth pair')
 d=add(c[...,0,:],c[...,1,:],'denom lanes0/1');d=add(d,c[...,2,:],'denom lane2');d=add(d,c[...,3,:],'denom lane3')
 return add(d[...,0],d[...,1],'denom half pair')

def forward(Q_bytes,K_bytes,V_bytes,hooks=None):
 for data in (Q_bytes,K_bytes,V_bytes):
  if type(data)is not bytes or len(data)!=65536:raise ValueError('exact Q/K/V packed bytes for N64')
 q=num.decode_e4m3(Q_bytes,qkv.QK_MAP).transpose(1,0,2)
 k=num.decode_e4m3(K_bytes,qkv.K_MAP).transpose(1,0,2)
 v=num.decode_e4m3(V_bytes,qkv.V_MAP).transpose(1,0,2)
 for name,value in [('Q',q),('K',k),('V',v)]:num._call_hook(hooks,name,value)
 score=rn(q@k.transpose(0,2,1),'QK dot32');prob=score_exp(score);codes=num.fp8_rn_satfinite(prob);packed_value=num.E4M3_LUT[codes]
 num._call_hook(hooks,'scores',score);num._call_hook(hooks,'unnormalized_half',prob);num._call_hook(hooks,'unnormalized_fp8',codes)
 denom=np.zeros((32,64),dtype=np.float16);numerator=np.zeros((32,64,32),dtype=np.float16);denom_parts=[];numerator_steps=[]
 for start in (0,):
  part=denominator64(prob[...,start:start+64]);denom_parts.append(part);denom=add(denom,part,'running denominator64')
  for k0 in (start,start+32):
   numerator=rn(packed_value[...,k0:k0+32]@v[:,k0:k0+32,:]+numerator.astype(np.float64),'PV dot32 plus C');numerator_steps.append(numerator.copy())
 clamped=np.maximum(denom,EPS).astype(np.float16);bits=clamped.view(np.uint16)
 if np.any((bits<0x0410)|(bits>=0x7c00)):raise num.UnsupportedNonFinite('measured reciprocal half domain')
 inv=rn(RCP_F32[bits].astype(np.float64),'measured RCP tohalf');result=rn(numerator.astype(np.float64)*inv[...,None].astype(np.float64),'final normalization')
 num._call_hook(hooks,'denominator',denom);num._call_hook(hooks,'weighted_sum',numerator);num._call_hook(hooks,'output_half',result)
 output=num.fp8_rn_satfinite(result).transpose(1,0,2).reshape(64,1024)
 physical=np.empty(65536,dtype=np.uint8);physical[OUTPUT_MAP]=output
 return dict(Q=q,K=k,V=v,scores=score,probabilities_half=prob,probabilities_fp8=codes,denominator_parts=denom_parts,denominator=denom,inverse_denominator=inv,numerator=numerator,numerator_steps=numerator_steps,output_half=result,output=output,output_physical=physical)

