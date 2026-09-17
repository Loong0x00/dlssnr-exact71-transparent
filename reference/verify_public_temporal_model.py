#!/usr/bin/env python3
from pathlib import Path
import builtins,subprocess,socket,ctypes,hashlib,json,numpy as np
from model_public_sm120 import ExactNR71PublicSM120
ROOT=Path(__file__).resolve().parent;color=np.fromfile(ROOT/'fixtures/public-input.rgba32f','<f4').reshape(512,512,4);model=ExactNR71PublicSM120();events=[]
def deny(kind):
 def f(*a,**k):events.append((kind,str(a[:2])));raise AssertionError('forward I/O '+kind)
 return f
old=(builtins.open,Path.open,Path.read_bytes,Path.read_text,subprocess.Popen,subprocess.run,socket.socket,ctypes.CDLL,ctypes.WinDLL if hasattr(ctypes,'WinDLL') else None);builtins.open=deny('open');Path.open=deny('Path.open');Path.read_bytes=deny('Path.read_bytes');Path.read_text=deny('Path.read_text');subprocess.Popen=deny('Popen');subprocess.run=deny('run');socket.socket=deny('socket');ctypes.CDLL=deny('CDLL')
if hasattr(ctypes,'WinDLL'):ctypes.WinDLL=deny('WinDLL')
try:z=model.forward_two_frames(color)
finally:
 builtins.open,Path.open,Path.read_bytes,Path.read_text,subprocess.Popen,subprocess.run,socket.socket,ctypes.CDLL=old[:8]
 if hasattr(ctypes,'WinDLL'):ctypes.WinDLL=old[8]
h=lambda b:hashlib.sha256(b).hexdigest();b0=z['frame0']['output_RGBA_f32_bits'].tobytes();b1=z['frame1']['output_RGBA_f32_bits'].tobytes();bh=z['history_RGBA16F_bits'].tobytes();s0=h(b0);s1=h(b1);sh=h(bh);fixture_equal=(b0==(ROOT/'fixtures/public-output.rgba32f').read_bytes() and b1==(ROOT/'fixtures/public-frame1-output.rgba32f').read_bytes() and bh==(ROOT/'fixtures/public-frame1-history.rgba16f').read_bytes());report={'status':'PASS_PURE_PUBLIC_TWO_FRAME_SM120_DLL_BITWISE' if fixture_equal and not events else 'FAIL','frame0_sha256':s0,'frame1_sha256':s1,'history_rgba16f_rtz_sha256':sh,'fixture_byte_equal':fixture_equal,'forward_io_events':events,'counter_after':z['frame1']['counter_after'],'scope':'512x512 deterministic RGBA32F, reset then counter=1, zero RG32F motion, no mask'};(ROOT/'PUBLIC_TWO_FRAME_ACCEPTANCE.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2));assert report['status'].startswith('PASS')
