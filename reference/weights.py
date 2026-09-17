"""Read-only logical 649-tensor interface for the pinned transparent model."""
from pathlib import Path
from types import MappingProxyType
import struct,json,hashlib,numpy as np
SHA256='399d900614665918b4f4d51cea0eebc4d8a3a3752c3ab19847e9206f50f2d043';DT={'F16':'<f2','F32':'<f4'}
class LogicalWeights649:
 def __init__(self,path):
  self.path=Path(path);h=hashlib.sha256()
  with self.path.open('rb') as f:
   for b in iter(lambda:f.read(8<<20),b''):h.update(b)
  if h.hexdigest()!=SHA256:raise ValueError('logical weights identity')
  with self.path.open('rb') as f:n=struct.unpack('<Q',f.read(8))[0];header=json.loads(f.read(n))
  meta=header.pop('__metadata__',{});self._base=8+n
  if len(header)!=649 or set(v['dtype'] for v in header.values())-set(DT):raise ValueError('649 logical F16/F32 tensors required')
  self._header=MappingProxyType(header);self.metadata=MappingProxyType(meta)
 def __len__(self):return 649
 def __iter__(self):return iter(self._header)
 def keys(self):return self._header.keys()
 def descriptor(self,name):return MappingProxyType(dict(self._header[name]))
 def __getitem__(self,name):
  d=self._header[name];lo,hi=d['data_offsets'];shape=tuple(d['shape']);a=np.memmap(self.path,mode='r',dtype=DT[d['dtype']],offset=self._base+lo,shape=shape,order='C');a.flags.writeable=False
  if a.nbytes!=hi-lo:raise ValueError('tensor extent')
  return a
