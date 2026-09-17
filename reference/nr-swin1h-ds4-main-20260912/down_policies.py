"""Explicit finite CPU NN policies; eight genuine learned one-K32 matrix products.
No claim about native internal TensorCore accumulation/FTZ outside prior evidence.
"""
import numpy as np
import ffn,downsample

def half_from_bits(bits):return np.array([bits],np.uint16).view(np.float16)[0]
def half_add(a,b):return ffn.num._rn_half(np.float64(a)+np.float64(b),'DS4 ordered pool pair/add')[()]
def half_mul(a,b):return ffn.num._rn_half(np.float64(a)*np.float64(b),'DS4 pool quarter')[()]
def quantize(a):return int(ffn.num.fp8_rn_satfinite(np.asarray(a,dtype=np.float16))[()])
def mma(a,b,c):
 aa=np.asarray(a,dtype=np.uint8);bb=np.asarray(b,dtype=np.uint8);cc=np.asarray(c,dtype=np.float16)
 if aa.shape!=(16,32) or bb.shape!=(8,32) or cc.shape!=(16,8):raise ValueError('one m16n8k32 matrix shapes')
 av=ffn.num.decode_e4m3(aa.tobytes(),np.arange(512).reshape(16,32));bv=ffn.num.decode_e4m3(bb.tobytes(),np.arange(256).reshape(8,32));return ffn.num._rn_half(av@bv.T+cc.astype(np.float64),'DS4 learned oneK32 projection')
def policies():return downsample.Policies(half_from_bits,half_add,half_mul,quantize,mma,'finite binary64 exact pair/product thenRNhalf perop','explicit existing E4M3 RN-satfinite','genuine FP8 matrix32-product plusC thenRNhalf perK32')
