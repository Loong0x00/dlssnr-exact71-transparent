"""Source-equation FFN candidate: expand, cubic, grouped projection, seeded mixing.
Consumes/produces canonical-C packed64x256 FP8. Separate original-operand acceptance.
"""
from pathlib import Path
import sys,importlib.util
import numpy as np
HERE=Path(__file__).resolve().parent;V=HERE.parent
sys.path.insert(0,str(V/'nr-vit-expand-tensor-reference-20260912'));import expand_tensor_reference as num
spec=importlib.util.spec_from_file_location('_swin8_ffn_axes',V/'nr-vit-contract-layout-main-20260912/tensor_reference.py');axes=importlib.util.module_from_spec(spec);spec.loader.exec_module(axes)
M=64;C=64;E=256
m=np.arange(M)[:,None];k=np.arange(C)[None,:];h=np.arange(E)[None,:]
A_MAP=axes.a_offset(m,k,C);C_MAP=axes.c_offset(m,k,C)
EXP_MAP=8192*(h//128)+axes.b_offset(k.T,h%128,128)
GROUP_MAP=16384+4096*np.arange(2)[:,None,None]+axes.b_offset(np.arange(128)[None,:,None],np.arange(32)[None,None,:],32)
MIX_MAP=24576+axes.b_offset(k.T,k,C)
def pi(k):return 16*(k//16)+2*((k%16)//4)+8*((k%4)//2)+k%2

def activate(x):
 if x.dtype!=np.float16 or not np.all(np.isfinite(x)):raise num.UnsupportedNonFinite('finite half activation')
 q=np.minimum(x.astype(np.float64),4);q=np.maximum(q,-4)
 a,b,c=np.array([0xab28,0x3728,0x3b28],dtype=np.uint16).view(np.float16).astype(np.float64)
 t=num._rn_half(a*np.abs(q)+b,'cubic first FMA')
 z=num._rn_half(q*t.astype(np.float64)+c,'cubic second FMA')
 return num._rn_half(x.astype(np.float64)*z.astype(np.float64),'cubic final multiply')

def forward(features,slab,hooks=None):
 if type(features)is not bytes or len(features)!=4096 or type(slab)is not bytes or len(slab)!=70048:raise ValueError('own64x256/slab byte contract')
 a=num.decode_e4m3(features,A_MAP);x=num.decode_e4m3(features,C_MAP)
 we=num.decode_e4m3(slab,EXP_MAP);wg=num.decode_e4m3(slab,GROUP_MAP);wm=num.decode_e4m3(slab,MIX_MAP)
 gamma=np.frombuffer(slab,dtype='<f2',count=64,offset=36864).copy()
 if not np.all(np.isfinite(gamma)):raise num.UnsupportedNonFinite('finite residual gamma')
 e=np.zeros((M,E),dtype=np.float16);eb={};ea={}
 for k0 in range(0,C,32):
  eb[k0]=e.copy();e=num._rn_half(a[:,k0:k0+32]@we[k0:k0+32]+e.astype(np.float64),'expand K32');ea[k0]=e.copy()
 activation=activate(e);codes=num.fp8_rn_satfinite(activation);act=num.decode_e4m3(codes.tobytes(),np.arange(M*E).reshape(M,E))
 grouped=np.zeros((M,C),dtype=np.float16);gb={};ga={}
 for q in range(4):
  gb[q]=grouped.copy()
  for g in range(2):
   inp=act[:,128*g+pi(np.arange(q*32,(q+1)*32))]
   grouped[:,32*g:32*(g+1)]=num._rn_half(inp@wg[g,q*32:(q+1)*32]+grouped[:,32*g:32*(g+1)].astype(np.float64),'grouped K32')
  ga[q]=grouped.copy()
 gcode=num.fp8_rn_satfinite(grouped);gact=num.decode_e4m3(gcode.tobytes(),np.arange(M*C).reshape(M,C));ma=gact[:,pi(np.arange(C))]
 seed=num._rn_half(x*gamma.astype(np.float64),'early residual seed multiply');out=seed.copy();mb={};md={}
 for k0 in range(0,C,32):
  mb[k0]=out.copy();out=num._rn_half(ma[:,k0:k0+32]@wm[k0:k0+32]+out.astype(np.float64),'mix seeded K32');md[k0]=out.copy()
 oc=num.fp8_rn_satfinite(out);physical=np.empty(4096,dtype=np.uint8);physical[C_MAP]=oc
 gp=np.empty(4096,dtype=np.uint8);gp[C_MAP]=gcode
 result=dict(input_mma_axes=a,input_C_axes=x,weight_expand=we,weight_group=wg,weight_mix=wm,gamma=gamma,expanded_half=e,activation_half=activation,activation_fp8=codes,grouped_half=grouped,grouped_fp8=gcode,grouped_physical=gp,mix_input_axes=ma,residual_seed_half=seed,prequantization=out,output=oc,physical=physical,expand_before=eb,expand_after=ea,group_before=gb,group_after=ga,mix_before=mb,mix_after=md)
 for name,v in result.items():
  if isinstance(v,np.ndarray):num._call_hook(hooks,name,v)
 return result
