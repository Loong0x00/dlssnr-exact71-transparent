"""Source-equation candidate, own consumer A versus producer C axes explicit.
N64 candidate, finite operands/carry/reduction domain. CPU NumPy, no GPU or DLL.
"""
from pathlib import Path
import sys,hashlib,json
import numpy as np
HERE=Path(__file__).resolve().parent;V=HERE.parent
sys.path.insert(0,str(V/'nr-vit-expand-tensor-reference-20260912'));import expand_tensor_reference as numeric
M,K,N=64,4096,1024
SOURCE=V/'nr-vit-contract-main-20260912/original.ptx'
SOURCE_SHA='241e1ef9d146d34ce54b36abed24bbe41e6a04f2a3c4a9c5af699fa63f2adf96'

def a_offset(m,k,width):
 return 16*width*(m//16)+512*(k//32)+16*(4*(m%8)+(k%16)//4)+8*((k%32)//16)+4*((m%16)//8)+k%4

def b_offset(k,n,width):
 return 32*width*(k//32)+512*(n//16)+16*(4*(n%8)+(k%16)//4)+8*((n%16)//8)+4*((k%32)//16)+k%4

def c_offset(m,n,width):
 return 16*width*(m//16)+512*(n//32)+16*(4*(m%8)+(n%8)//2)+8*((n%32)//16)+4*((m%16)//8)+2*((n%16)//8)+n%2

def pi(k):
 """Consumer MMA-A channel k names producer C channel pi(k), same token."""
 return 16*(k//16)+2*((k%16)//4)+8*((k%4)//2)+k%2

def maps():
 m=np.arange(M,dtype=np.int64)[:,None];k=np.arange(K,dtype=np.int64)[None,:];n=np.arange(N,dtype=np.int64)[None,:]
 return a_offset(m,k,K),b_offset(k.T,n,N),c_offset(m,n,N)

A_MAP,B_MAP,C_MAP=maps()

def forward(p0,p8,slab,hooks=None):
 for b,size in ((p0,262144),(p8,65536),(slab,4196352)):
  if type(b)is not bytes or len(b)!=size:raise ValueError('exact input/slab bytes required')
 a=numeric.decode_e4m3(p0,A_MAP);b=numeric.decode_e4m3(slab,B_MAP);skip=numeric.decode_e4m3(p8,C_MAP)
 coefficient=np.frombuffer(slab,dtype='<f2',offset=4194304).copy()
 if not np.all(np.isfinite(coefficient)):raise numeric.UnsupportedNonFinite('finite residual coefficient required')
 for name,array in [('input_mma_axes',a),('weight_mma_axes',b),('skip_output_axes',skip),('residual_coefficient',coefficient)]:numeric._call_hook(hooks,name,array)
 carry=numeric._rn_half(skip*coefficient.astype(np.float64),'learned skip C')
 partials=[];combined=None
 for z in range(4):
  c=carry.copy() if z==0 else np.zeros((M,N),dtype=np.float16)
  for k0 in range(z*1024,(z+1)*1024,32):
   c=numeric._rn_half(a[:,k0:k0+32]@b[k0:k0+32,:]+c.astype(np.float64),f'z{z} K{k0}')
  partials.append(c);numeric._call_hook(hooks,f'partial_z{z}',c)
  combined=c.copy() if z==0 else numeric._rn_half(combined.astype(np.float64)+c.astype(np.float64),f'ordered z{z} half add noftz')
 numeric._call_hook(hooks,'prequantization',combined)
 output=numeric.fp8_rn_satfinite(combined);physical=np.empty(M*N,dtype=np.uint8);physical[C_MAP]=output
 numeric._call_hook(hooks,'output',output)
 return dict(input_mma_axes=a,weight_mma_axes=b,skip_output_axes=skip,residual_coefficient=coefficient,initial_c=carry,partials=partials,prequantization=combined,output=output,output_physical=physical)

def main():
 root=V/'nr-vit-expand-tensor-main-20260912';p0=(root/'output_fp8_physical.bin').read_bytes();p8=(root/'input_fp8_physical.bin').read_bytes()
 ck=HERE.parent/'weights/dlssnr-310.8-weights.safetensors'
 with ck.open('rb') as f:f.seek(39812674);slab=f.read(4196352)
 assert hashlib.sha256(slab).hexdigest()=='5c0a940e75c7df37d50098a53796a5014019844d347dabe8c34d73dff4ca6696'
 result=forward(p0,p8,slab);data=result['output_physical'].tobytes()
 (HERE/'prediction_fp8_physical.bin').write_bytes(data)
 pred=dict(classification='source-equation CPU prediction; comparison and address proof separate',source_sha256=SOURCE_SHA,implementation_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),p0_sha256=hashlib.sha256(p0).hexdigest(),p8_sha256=hashlib.sha256(p8).hexdigest(),slab_sha256=hashlib.sha256(slab).hexdigest(),output_sha256=hashlib.sha256(data).hexdigest(),output_bytes=len(data))
 (HERE/'PREDICTION.json').write_text(json.dumps(pred,indent=2)+'\n');print(pred['output_sha256'])
if __name__=='__main__':main()
