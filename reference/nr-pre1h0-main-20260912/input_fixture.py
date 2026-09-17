"""Explicit synthetic procedural raw-F32 sampler, not real/native texture semantics."""
import struct
import preprocess_candidate as pre
from approx_policies import reference_nonnative
SOURCE_HANDLE=0x1100000011

def args(h=128,w=128):
 a=bytearray(264);struct.pack_into('<Q',a,0,SOURCE_HANDLE);struct.pack_into('<6f',a,136,0,0,1,1,1,1);struct.pack_into('<5fIf',a,172,.25,.75,2/128,-1,-1,1,1/16);struct.pack_into('<Q',a,200,0);struct.pack_into('<2i',a,208,h,w);struct.pack_into('<2Q',a,216,0x40000000,0x20000000);struct.pack_into('<2i',a,240,h,w);struct.pack_into('<Q',a,248,0x50000000);struct.pack_into('<2i',a,256,h//2,w//2);return bytes(a)

def sample(handle,xbits,ybits):
 if handle!=SOURCE_HANDLE:raise ValueError('explicit source resource only')
 x=struct.unpack('<f',struct.pack('<I',xbits))[0];y=struct.unpack('<f',struct.pack('<I',ybits))[0]
 return tuple(struct.unpack('<I',struct.pack('<f',v))[0] for v in (.2+.25*x,.3+.2*y,.4+.1*x*y,1.0))

def sampler():return pre.SamplerPolicy('SYNTHETIC_procedural_F32_source_not_native_sampler',False,sample)
def approx():return reference_nonnative()
