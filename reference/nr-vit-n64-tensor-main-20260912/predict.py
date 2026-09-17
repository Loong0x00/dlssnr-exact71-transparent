"""First N64 tensor prediction, saved before comparing sealed literal results."""
from pathlib import Path
import importlib.util,json,struct,hashlib
import numpy as np
HERE=Path(__file__).resolve().parent;V=HERE.parent;RAW=V/'nr-vit-n64-main-20260912'
sha=lambda b:hashlib.sha256(b).hexdigest()
def load(name):
 spec=importlib.util.spec_from_file_location(name,HERE/f'{name}_tensor.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
expand,contract,qkv,attention,projection=[load(s) for s in ('expand','contract','qkv','attention','projection')]

def weights():
 ck=V/'weights/dlssnr-310.8-weights.safetensors';slabs=[]
 with ck.open('rb') as f:
  n=struct.unpack('<Q',f.read(8))[0];header=json.loads(f.read(n))
  for i in range(5):
   a,b=header[f'block31.layer{i}.layer']['data_offsets'];f.seek(8+n+a);slabs.append(f.read(b-a))
 return slabs

def forward(inp,slabs):
 e=expand.forward(inp,slabs[0]);F=e['output_physical'].tobytes()
 c=contract.forward(F,inp,slabs[1]);P=c['output_physical'].tobytes()
 q=qkv.forward(P,slabs[2]);qbytes=[q[n+'_physical'].tobytes() for n in ('Q','K','V')]
 a=attention.forward(*qbytes);A=a['output_physical'].tobytes()
 p=projection.forward(A,P,slabs[4]);Y=p['output_physical'].tobytes()
 return dict(expand=F,contract=P,Q=qbytes[0],K=qbytes[1],V=qbytes[2],attention=A,projection=Y),dict(expand=e,contract=c,qkv=q,attention=a,projection=p)

def main():
 inp=(V/'nr-split16h-pool-main-20260912/repacked_N64_fp8.bin').read_bytes();slabs=weights();outputs,traces=forward(inp,slabs)
 meta=dict(scope='first N64 valid-row tensor prediction, before output comparison',input_sha256=sha(inp),slab_sha256=[sha(b) for b in slabs],implementation_sha256={name:sha((HERE/f'{name}_tensor.py').read_bytes()) for name in ('expand','contract','qkv','attention','projection')},outputs={})
 for name,data in outputs.items():
  (HERE/f'{name}_prediction.bin').write_bytes(data);meta['outputs'][name]=dict(bytes=len(data),sha256=sha(data))
 (HERE/'FIRST_PREDICTION.json').write_text(json.dumps(meta,indent=2)+'\n')
 comparison={}
 for name,data in outputs.items():
  fn=f'{name}_fp8_physical.bin' if name in ('Q','K','V') else f'{name}_output_fp8_physical.bin';expected=(RAW/fn).read_bytes()
  comparison[name]=dict(bytes=len(data),length_match=len(data)==len(expected),different_bytes=sum(a!=b for a,b in zip(data,expected)))
 (HERE/'COMPARISON.json').write_text(json.dumps(comparison,indent=2)+'\n');print(json.dumps(comparison,indent=2))
 if any(not r['length_match'] or r['different_bytes'] for r in comparison.values()):raise ValueError('first N64 tensor prediction mismatch; preserve before corrections')
if __name__=='__main__':main()
