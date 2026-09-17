#!/usr/bin/env python3
"""Bytewise and float32 comparison for two raw RGBA buffers."""
from __future__ import annotations
import argparse, hashlib
from pathlib import Path
import numpy as np

def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()

def main() -> int:
    ap=argparse.ArgumentParser()
    ap.add_argument("expected",type=Path)
    ap.add_argument("actual",type=Path)
    args=ap.parse_args()
    a=args.expected.read_bytes(); b=args.actual.read_bytes()
    print(f"expected_sha256={digest(a)}")
    print(f"actual_sha256={digest(b)}")
    print(f"expected_bytes={len(a)} actual_bytes={len(b)}")
    n=min(len(a),len(b)); print(f"differing_bytes={sum(x!=y for x,y in zip(a[:n],b[:n])) + abs(len(a)-len(b))}")
    if len(a)%4==0 and len(b)%4==0 and len(a)==len(b):
        af=np.frombuffer(a,dtype='<f4'); bf=np.frombuffer(b,dtype='<f4')
        finite=np.isfinite(af)&np.isfinite(bf)
        d=np.abs(af[finite]-bf[finite])
        print(f"differing_f32_bits={int(np.count_nonzero(af.view('<u4')!=bf.view('<u4')))}")
        print(f"max_abs_finite={float(d.max()) if d.size else 0.0}")
    ok=a==b
    print("PASS_BYTE_EQUAL" if ok else "FAIL_BYTE_DIFFERENT")
    return 0 if ok else 1
if __name__=='__main__': raise SystemExit(main())
