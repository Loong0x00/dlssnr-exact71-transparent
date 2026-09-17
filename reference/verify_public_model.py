#!/usr/bin/env python3
from pathlib import Path
import builtins,ctypes,hashlib,json,os,socket,subprocess,sys,numpy as np
ROOT=Path(__file__).resolve().parent;FIXTURES=ROOT/'fixtures';EXPECTED_SHA='571fa2100c1842deff8769510a78f90cbe84810d64bb3ea82eaba41cf1cd793e'
def sha(b):return hashlib.sha256(b).hexdigest()
# All allowed constructor/input I/O precedes arming.
color=np.fromfile(FIXTURES/'public-input.rgba32f','<f4').reshape(512,512,4);expected=(FIXTURES/'public-output.rgba32f').read_bytes()
from model_public_sm120 import ExactNR71PublicSM120
model=ExactNR71PublicSM120();events=[];armed=False
def audit(event,args):
 if armed and (event=='open' or event.startswith('socket.') or event.startswith('subprocess.') or event.startswith('ctypes.')):events.append((event,repr(args)[:300]));raise RuntimeError('forbidden forward audit event '+event)
sys.addaudithook(audit)
def deny(*a,**k):raise RuntimeError('forbidden forward I/O/process/device loader')
old=(builtins.open,Path.open,Path.read_bytes,Path.write_bytes,np.fromfile,subprocess.Popen,subprocess.run,subprocess.check_output,socket.socket,ctypes.CDLL,ctypes.PyDLL,os.system)
builtins.open=deny;Path.open=deny;Path.read_bytes=deny;Path.write_bytes=deny;np.fromfile=deny;subprocess.Popen=deny;subprocess.run=deny;subprocess.check_output=deny;socket.socket=deny;ctypes.CDLL=deny;ctypes.PyDLL=deny;os.system=deny
armed=True
try:
 a=model.forward(color)['output_RGBA_f32_bits'].tobytes();b=model.forward(color)['output_RGBA_f32_bits'].tobytes()
finally:
 armed=False;(builtins.open,Path.open,Path.read_bytes,Path.write_bytes,np.fromfile,subprocess.Popen,subprocess.run,subprocess.check_output,socket.socket,ctypes.CDLL,ctypes.PyDLL,os.system)=old
result={'status':'PASS_SINGLE_INDEPENDENT_TRANSPARENT_PUBLIC_NR71_SM120_FIRST_FRAME_BITWISE','output_bytes':len(a),'run1_sha256':sha(a),'run2_sha256':sha(b),'expected_public_dll_sha256':sha(expected),'run1_run2_byte_equal':a==b,'run1_public_dll_byte_equal':a==expected,'differing_bytes':sum(x!=y for x,y in zip(a,expected)),'forward_audit_events':events,'forward_no_file_network_process_dlopen':not events,'numeric_target':model.numeric_target,'scope':model.supported_state,'geometry':{'visible':[512,512],'internal_high':[512,576],'vit_N':96},'public_acceptance':'../nr-native-equivalence-matrix-main-20260914/PUBLIC_FULL71_ACCEPTANCE.json'}
assert result['run1_sha256']==result['run2_sha256']==result['expected_public_dll_sha256']==EXPECTED_SHA and result['run1_run2_byte_equal'] and result['run1_public_dll_byte_equal'] and not events
(ROOT/'PUBLIC_PACKAGE_RESULT.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
