#!/usr/bin/env python3
from pathlib import Path
import ast,hashlib,json,struct
R=Path(__file__).resolve().parent;exclude={'PUBLIC_PACKAGE_MANIFEST.json','PUBLIC_PACKAGE_MANIFEST.log'};files={};symlinks=[];nonregular=[]
for p in sorted(R.rglob('*')):
 rel=p.relative_to(R).as_posix()
 if p.is_symlink():symlinks.append(rel);continue
 if p.is_dir():continue
 if not p.is_file():nonregular.append(rel);continue
 if rel in exclude:continue
 b=p.read_bytes();files[rel]={'bytes':len(b),'sha256':hashlib.sha256(b).hexdigest()}
entry=('model_public_sm120.py','public_pre_stage.py','public_encoder_stage.py','public_split_stage.py','public_vit_stage.py','public_decoder_stage.py','public_temporal_stage.py','public_temporal_post_stage.py','sm120_hmma.py','qkv_sm120_cubin.py');forbidden_imports=[]
for rel in entry:
 t=ast.parse((R/rel).read_text(),rel)
 for n in ast.walk(t):
  if isinstance(n,ast.Import):names=[x.name.split('.')[0] for x in n.names]
  elif isinstance(n,ast.ImportFrom):names=[(n.module or '').split('.')[0]]
  else:continue
  for x in names:
   if x in {'ctypes','subprocess','socket','torch','cupy'}:forbidden_imports.append([rel,x])
exts={p.suffix.lower() for p in R.rglob('*') if p.is_file()};forbidden_files=[rel for rel in files if Path(rel).suffix.lower() in {'.dll','.exe','.cubin','.ptx','.so'}]
with (R/'weights/dlssnr-logical-v18.safetensors').open('rb') as f:n=struct.unpack('<Q',f.read(8))[0];head=json.loads(f.read(n));head.pop('__metadata__',None)
agg=hashlib.sha256()
for rel,row in sorted(files.items()):agg.update(rel.encode()+b'\0'+bytes.fromhex(row['sha256']))
z={'status':'PASS_INDEPENDENT_TRANSPARENT_PUBLIC_SM120_PACKAGE_MANIFEST','entrypoint':'model_public_sm120.ExactNR71PublicSM120','file_count':len(files),'total_bytes':sum(x['bytes'] for x in files.values()),'aggregate_name_hash_sha256':agg.hexdigest(),'symlinks':symlinks,'nonregular':nonregular,'forbidden_binary_or_ptx_files':forbidden_files,'forbidden_entry_imports':forbidden_imports,'absolute_home_paths_in_entry_sources':sum('/home/user' in (R/x).read_text() for x in entry),'logical_tensor_count':len(head),'logical_weights_sha256':files['weights/dlssnr-logical-v18.safetensors']['sha256'],'checkpoint_sha256':files['weights/dlssnr-310.8-weights.safetensors']['sha256'],'fixture_input_sha256':files['fixtures/public-input.rgba32f']['sha256'],'fixture_output_sha256':files['fixtures/public-output.rgba32f']['sha256'],'fixture_frame1_output_sha256':files['fixtures/public-frame1-output.rgba32f']['sha256'],'fixture_frame1_history_sha256':files['fixtures/public-frame1-history.rgba16f']['sha256'],'two_frame_no_io_result':json.loads((R/'PUBLIC_TWO_FRAME_ACCEPTANCE.json').read_text()),'isolated_two_frame_result':json.loads((R/'ISOLATED_PUBLIC_TWO_FRAME_RESULT.json').read_text()),'repeat_no_io_result':json.loads((R/'PUBLIC_PACKAGE_RESULT.json').read_text()),'isolated_copy_result':json.loads((R/'ISOLATED_PUBLIC_PACKAGE_RESULT.json').read_text()),'files':files};assert not symlinks and not nonregular and not forbidden_files and not forbidden_imports and z['absolute_home_paths_in_entry_sources']==0 and len(head)==649 and z['checkpoint_sha256']=='890a53f05473dd6fd0980d1ca8501cfd84c1f6d7330b9531ca0981b6a6778d0e' and z['logical_weights_sha256']=='399d900614665918b4f4d51cea0eebc4d8a3a3752c3ab19847e9206f50f2d043' and z['two_frame_no_io_result']['status'].startswith('PASS_') and z['isolated_two_frame_result']['status'].startswith('PASS_') and z['repeat_no_io_result']['status'].startswith('PASS_') and z['isolated_copy_result']['status'].startswith('PASS_');(R/'PUBLIC_PACKAGE_MANIFEST.json').write_text(json.dumps(z,indent=2)+'\n');print(json.dumps({k:v for k,v in z.items() if k!='files'},indent=2))
