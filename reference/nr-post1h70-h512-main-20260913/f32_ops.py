"""Explicit finite RN-F32/FTZ integer-rational CPU policy."""
import struct,math
from fractions import Fraction
class UnsupportedDomain(ValueError):pass
SIGN=0x80000000;EXP=0x7f800000;MANT=0x7fffff
def as_float(bits):
 v=struct.unpack('<f',struct.pack('<I',bits&0xffffffff))[0]
 if not math.isfinite(v):raise UnsupportedDomain('finiteF32 only')
 return v
def as_bits(value):
 try:b=struct.pack('<f',value)
 except (OverflowError,struct.error) as e:raise UnsupportedDomain('finiteF32 overflow') from e
 v=struct.unpack('<f',b)[0]
 if not math.isfinite(v):raise UnsupportedDomain('finiteF32 only')
 return struct.unpack('<I',b)[0]
def ftz_bits(bits):
 bits&=0xffffffff;as_float(bits);return bits&SIGN if bits&EXP==0 else bits
def frac(bits):
 b=ftz_bits(bits);e=(b>>23)&255;m=b&MANT
 if e==0:return Fraction(0)
 n=(1<<23)+m;p=e-127-23;q=Fraction(n<<p) if p>=0 else Fraction(n,1<<-p);return -q if b&SIGN else q
def rr(n,d):
 q,r=divmod(n,d);return q+int(2*r>d or (2*r==d and q&1))
def enc(value,zero_sign=0):
 q=Fraction(value)
 if q==0:return (zero_sign&1)<<31
 sign=int(q<0);n,d=abs(q.numerator),q.denominator;e=n.bit_length()-d.bit_length()
 if (n<(d<<e)) if e>=0 else ((n<<-e)<d):e-=1
 quantum=max(e,-126)-23;m=rr(n,d<<quantum) if quantum>=0 else rr(n<<-quantum,d)
 if m>=1<<24:m>>=1;quantum+=1
 if m<1<<23:return (sign<<31)|m
 ef=quantum+150
 if ef>=255:raise UnsupportedDomain('finiteF32 overflow')
 return (sign<<31)|(ef<<23)|(m-(1<<23))
def apply(op,b):
 b=[x&0xffffffff for x in b]
 if op.startswith('cvt.rn.f32.'):
  n=b[0]
  if op.endswith('s32') and n&SIGN:n-=1<<32
  return enc(Fraction(n))
 if op=='selp.f32':return b[0] if b[2] else b[1]
 x=[ftz_bits(q) for q in b];v=[frac(q) for q in x]
 if op=='add.ftz.f32':out=enc(v[0]+v[1],int(v[0]==0 and v[1]==0 and x[0]&SIGN and x[1]&SIGN))
 elif op=='sub.ftz.f32':
  ybits=x[1]^SIGN;out=enc(v[0]-v[1],int(v[0]==0 and v[1]==0 and x[0]&SIGN and ybits&SIGN))
 elif op=='mul.ftz.f32':out=enc(v[0]*v[1],((x[0]^x[1])>>31)&1)
 elif op=='fma.rn.ftz.f32':out=enc(v[0]*v[1]+v[2],int(v[0]*v[1]==0 and v[2]==0 and ((x[0]^x[1])>>31)==(x[2]>>31)==1))
 elif op=='min.ftz.f32':
  if v[0]==v[1]==0:return (x[0]|x[1])&SIGN
  out=enc(min(v[0],v[1]))
 elif op=='max.ftz.f32':
  if v[0]==v[1]==0:return (x[0]&x[1])&SIGN
  out=enc(max(v[0],v[1]))
 elif op=='abs.ftz.f32':out=x[0]&~SIGN
 elif op=='setp.equ.ftz.f32':return v[0]==v[1]
 elif op=='setp.gt.ftz.f32':return v[0]>v[1]
 elif op=='cvt.rmi.ftz.f32.f32':out=x[0] if v[0]==0 else enc(Fraction(math.floor(v[0])))
 else:raise UnsupportedDomain('no supported finite policy for '+op)
 return ftz_bits(out)
