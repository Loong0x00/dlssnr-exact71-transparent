"""Source-derived FP8 1D->2D raw b32 repack; no FP8 arithmetic/conversion.
Admits positive H,W multiples4; host FP8 CTA256/gridH*W. Not inferred as inverse.
"""
from pathlib import Path
import struct,hashlib,json
HERE=Path(__file__).resolve().parent
NAME='cc_vit_1d_repack_1d_to_2d_fp8';ABI_BYTES=24;CTA=(256,1,1)

def dimensions(h,w):
 if any(type(x)is not int or not 4<=x<=256 or x%4 for x in (h,w)):raise ValueError('bounded positive four-aligned geometry required')
 return h*w

def word_in_tile(m,q):
 # Original s16 intermediates reduce to these nonnegative values in domain:
 # rs1=q,rs7=4*(q%8),rs13=4*(q%4),rs15=(q%8)//4,
 # rs22=m%8,rs24=(m%16)//8,rs28=4*(m%8)+q%4,rs32=q//8.
 return 128*(q//8)+16*(m%8)+4*(q%4)+2*((q%8)//4)+(m%16)//8

def source_word(work):
 m,q=divmod(work,256)
 return 4096*(m//16)+word_in_tile(m,q)

def destination_word(work,w):
 m,q=divmod(work,256);y,x=divmod(m,w)
 return 4096*((y//4)*(w//4)+x//4)+word_in_tile(4*(y%4)+x%4,q)

def forward(source,h,w):
 n=dimensions(h,w);size=n*1024
 if type(source)is not bytes or len(source)<size:raise ValueError('complete original source read footprint required')
 out=bytearray(size)
 for work in range(n*256):
  a=4*source_word(work);b=4*destination_word(work,w);out[b:b+4]=source[a:a+4]
 return bytes(out)

def record(src,dst,h,w):
 dimensions(h,w)
 if any(type(x)is not int or x<=0 or x%4 or x>=1<<64 for x in (src,dst)):raise ValueError('aligned nonzero symbolic pointers')
 return struct.pack('<QQii',src,dst,h,w)
