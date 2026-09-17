"""Transparent N64 QKV tensor for the original DLL sm_120 CUBIN target.
The only delta from the preserved PTX-source model is the proven ptxas HFMA2
contraction in the Q/K square reduction; no CUBIN/PTX is executed here.
"""
from pathlib import Path
import sys,importlib.util,json,hashlib,struct,math
import numpy as np
HERE=Path(__file__).resolve().parent;V=HERE
sys.path.insert(0,str(V/'nr-vit-expand-tensor-reference-20260912'));import expand_tensor_reference as num
spec=importlib.util.spec_from_file_location('qkv_axes',V/'nr-vit-contract-layout-main-20260912/tensor_reference.py');axes=importlib.util.module_from_spec(spec);spec.loader.exec_module(axes)
SOURCE_SHA='d9c05b923ffcf28e047adb1d9cda1a958191d7811051dda5feb0172cf9ef9917'
M,K,H,D=64,1024,32,32
m=np.arange(M)[:,None];k=np.arange(K)[None,:];n=np.arange(3072)[None,:]
A_MAP=axes.a_offset(m,k,1024);B_MAP=128+axes.b_offset(k.T,n,3072)
mm=np.arange(M)[:,None,None];hh=np.arange(H)[None,:,None];dd=np.arange(D)[None,None,:]
QK_MAP=axes.c_offset(mm,32*hh+dd,1024)
# K's rs57..64 order is d0,d8,d16,d24 at m0, then the same at m8.
# Q instead interleaves m0/m8 after each pair of d8 groups.
K_MAP=(16384*(mm//16)+512*hh+16*(4*(mm%8)+(dd%8)//2)+8*((mm%16)//8)+4*(dd//16)+2*((dd%16)//8)+dd%2)
V_MAP=(32768*(mm//32)+1024*hh+512*(dd//16)+16*(4*(dd%8)+(mm%8)//2)+8*((dd%16)//8)+4*((mm%32)//16)+2*((mm%16)//8)+mm%2)

def rn(v,label):return num._rn_half(v,label)
def add(a,b,label):return rn(a.astype(np.float64)+b.astype(np.float64),label)
def mul(a,b,label):return rn(a.astype(np.float64)*b.astype(np.float64),label)
def fma(a,b,c,label):return rn(a.astype(np.float64)*b.astype(np.float64)+c.astype(np.float64),label)
_lut_bytes=(V/'nr-native-normalization-20260911/runs/20260911-134003/rsqrt-f32.bin').read_bytes()
assert hashlib.sha256(_lut_bytes).hexdigest()=='a43128fa609f3f3a3e2538726fcf6feac5f950b9d5ef8eb523df3fbf43c96011'
RSQRT_F32=np.frombuffer(_lut_bytes,dtype='<f4')
EPSILON=np.array([0x0410],dtype='<u2').view('<f2')[0]
SQRT32_HALF=np.array([struct.unpack('<f',struct.pack('<f',math.sqrt(32.)))[0]],dtype=np.float64).astype(np.float16)[0]

def normalize(x):
 """CUBIN SASS: x8*x8+RN(x24*x24), x0*x0+RN(x16*x16), then HADD2 tree."""
 square24=mul(x[...,24:32],x[...,24:32],'square24')
 square16=mul(x[...,16:24],x[...,16:24],'square16')
 pair8_24=fma(x[...,8:16],x[...,8:16],square24,'HFMA square pair8+24')
 pair0_16=fma(x[...,:8],x[...,:8],square16,'HFMA square pair0+16')
 g=add(pair8_24,pair0_16,'four groups')
 pair=add(add(g[...,:2],g[...,4:6],'xor2 lane0+2'),add(g[...,2:4],g[...,6:8],'xor2 lane1+3'),'xor1')
 norm2=add(pair[...,0],pair[...,1],'half swap/add')
 clamped=np.maximum(norm2,EPSILON).astype(np.float16)
 bits=clamped.view(np.uint16)
 if np.any((bits==0)|(bits>=0x7c00)):raise num.UnsupportedNonFinite('positive finite measured RSQRT half required')
 inv=rn(RSQRT_F32[bits].astype(np.float64),'measured RSQRT tohalf')
 result=mul(x,inv[...,None],'normalized')
 return result,dict(square24=square24,square16=square16,pair8_24=pair8_24,pair0_16=pair0_16,norm2=norm2,clamped_norm2=clamped,inverse_norm=inv)

def forward(input_fp8,slab,hooks=None):
 if type(input_fp8)is not bytes or len(input_fp8)!=M*K or type(slab)is not bytes or len(slab)!=3145856:raise ValueError('exact original QKV input/slab spans required')
 a=num.decode_e4m3(input_fp8,A_MAP);b=num.decode_e4m3(slab,B_MAP)
 scales=rn(np.frombuffer(slab,dtype='<f4',count=32).astype(np.float64),'head scales')
 num._call_hook(hooks,'input_mma_axes',a);num._call_hook(hooks,'weight_mma_axes',b);num._call_hook(hooks,'head_scales',scales)
 partials=[]
 for z in range(2):
  c=np.zeros((M,3072),dtype=np.float16)
  for k0 in range(512*z,512*(z+1),32):c=rn(a[:,k0:k0+32]@b[k0:k0+32]+c.astype(np.float64),'QKV perK32')
  partials.append(c);num._call_hook(hooks,f'partial_z{z}',c.reshape(M,32,3,32))
 merged=add(partials[0],partials[1],'two zpartials').reshape(M,32,3,32)
 q,qn=normalize(merged[:,:,0,:]);key,kn=normalize(merged[:,:,1,:]);val=merged[:,:,2,:]
 q=mul(mul(q,np.asarray(SQRT32_HALF),'sqrt32 half queryscale'),scales[None,:,None],'learned queryscale')
 result=dict(input_mma_axes=a,weight_mma_axes=b,head_scales=scales,partials=partials,merged=merged,Q_half=q,K_half=key,V_half=val,Q_norm=qn,K_norm=kn)
 for name,tensor,mapping in [('Q',q,QK_MAP),('K',key,K_MAP),('V',val,V_MAP)]:
  num._call_hook(hooks,name,tensor);codes=num.fp8_rn_satfinite(tensor);physical=np.empty(M*K,dtype=np.uint8);physical[mapping]=codes;result[name]=codes;result[name+'_physical']=physical
 return result

