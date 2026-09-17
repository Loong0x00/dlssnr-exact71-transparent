"""Transparent bit model of Blackwell SM120 HMMA.16816.F16 with F16 C/D.

Global-alignment parameters are the B200 FP16->FP16 model (fma=16,
neab=2, alignment truncation, min product-alignment exponent -21, final RNE)
from north-numerical-computing/MATLAB-tensor-core commit
79143bd29a4a71dc7c2ed47741b63572498b17f6. This file is an independent
integer implementation and executes no GPU code, CUBIN, PTX, VM, or DLL.
"""
import numpy as np
MIN_ALIGN_EXP=-21
EXTRA_ALIGN_BITS=2

def finite_units(h):
 h=int(h)
 if not 0<=h<65536 or h&0x7c00==0x7c00:raise ValueError('finite binary16 bits required')
 e=(h>>10)&31;f=h&1023;n=f if e==0 else (1024+f)<<(e-1)
 return -n if h&0x8000 else n

def round_half(n,scale=48):
 if type(n) is not int or type(scale) is not int or scale<24:raise ValueError('integer fixed-point half rounding domain')
 sign=0x8000 if n<0 else 0;n=abs(n)
 if not n:return 0
 floor=n.bit_length()-1-scale;shift=scale+max(floor-10,-24);q,r=divmod(n,1<<shift)
 if r>(1<<shift)//2 or (shift and r==(1<<(shift-1)) and q&1):q+=1
 if floor<-14:return sign|q
 e=floor+15
 if q==2048:q=1024;e+=1
 if e>=31:raise OverflowError('SM120 HMMA result overflow outside recovered finite domain')
 return sign|(e<<10)|(q-1024)

def _input_sig_exp(bits):
 e=(bits>>10)&31;f=bits&1023
 return (f,-14) if e==0 else (1024+f,e-15)

def _c_sig_exp(bits):
 e=(bits>>10)&31;f=bits&1023
 if e:return (1024+f)<<13,e-15
 top=f.bit_length()-1
 return f<<(23-top),top-24

def dot16(a_bits,b_bits,c_bits=0):
 if len(a_bits)!=16 or len(b_bits)!=16:raise ValueError('one K16 dot required')
 c_bits=int(c_bits)
 if not 0<=c_bits<65536 or c_bits&0x7c00==0x7c00:raise ValueError('finite F16 accumulator required')
 terms=[];maximum=-32768
 for ah,bh in zip(a_bits,b_bits):
  ah=int(ah);bh=int(bh)
  if not 0<=ah<65536 or not 0<=bh<65536 or ah&0x7c00==0x7c00 or bh&0x7c00==0x7c00:raise ValueError('finite F16 multiplicands required')
  if not(ah&0x7fff) or not(bh&0x7fff):continue
  sa,ea=_input_sig_exp(ah);sb,eb=_input_sig_exp(bh);ex=ea+eb;terms.append((-1 if (ah^bh)&0x8000 else 1,(sa*sb)<<(3+EXTRA_ALIGN_BITS),ex));maximum=max(maximum,ex)
 if c_bits&0x7fff:
  sig,ex=_c_sig_exp(c_bits);terms.append((-1 if c_bits&0x8000 else 1,sig<<EXTRA_ALIGN_BITS,ex));maximum=max(maximum,ex)
 if not terms:return 0
 maximum=max(maximum,MIN_ALIGN_EXP);total=0
 for sign,sig,ex in terms:
  shift=maximum-ex
  if shift<=31:total+=sign*(sig>>shift)
 if not total:return 0
 answer=round_half(int(total)<<(maximum+23))
 return 0 if not(answer&0x7fff) else answer

def mma_bits(a_bits,b_bits,c_bits=None):
 a=np.asarray(a_bits);b=np.asarray(b_bits)
 if a.dtype!=np.uint16 or b.dtype!=np.uint16 or a.ndim!=2 or b.ndim!=2 or a.shape[1]!=16 or b.shape[0]!=16:raise ValueError('uint16 A[M,16], B[16,N] required')
 if c_bits is None:c=np.zeros((a.shape[0],b.shape[1]),dtype=np.uint16)
 else:
  c=np.asarray(c_bits)
  if c.dtype!=np.uint16 or c.shape!=(a.shape[0],b.shape[1]):raise ValueError('uint16 C[M,N] required')
 if np.any((a&0x7c00)==0x7c00) or np.any((b&0x7c00)==0x7c00) or np.any((c&0x7c00)==0x7c00):raise ValueError('finite F16 matrix domain')
 M,N=a.shape[0],b.shape[1];out=np.empty((M,N),dtype=np.uint16)
 for m in range(M):
  av=a[m].tolist()
  for n in range(N):out[m,n]=dot16(av,b[:,n].tolist(),int(c[m,n]))
 return out
