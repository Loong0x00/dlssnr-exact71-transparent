#!/usr/bin/env python3
"""Verify an extracted ExactNR71 release against its embedded frozen manifest."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import sys

EXCLUDED = {"PUBLIC_PACKAGE_MANIFEST.json", "PUBLIC_PACKAGE_MANIFEST.log"}

def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("package", type=Path)
    args = ap.parse_args()
    root = args.package.resolve()
    manifest_path = root / "PUBLIC_PACKAGE_MANIFEST.json"
    manifest = json.loads(manifest_path.read_text())
    expected = manifest["files"]
    actual = {
        p.relative_to(root).as_posix()
        for p in root.rglob("*")
        if p.is_file() and p.relative_to(root).as_posix() not in EXCLUDED
    }
    wanted = set(expected)
    failures: list[str] = []
    if actual != wanted:
        for name in sorted(wanted - actual): failures.append(f"MISSING {name}")
        for name in sorted(actual - wanted): failures.append(f"UNEXPECTED {name}")
    aggregate = hashlib.sha256()
    for name in sorted(wanted):
        path = root / name
        if not path.is_file():
            continue
        got_size = path.stat().st_size
        got_sha = sha256(path)
        row = expected[name]
        if got_size != row["bytes"]:
            failures.append(f"SIZE {name}: {got_size} != {row['bytes']}")
        if got_sha != row["sha256"]:
            failures.append(f"SHA256 {name}: {got_sha} != {row['sha256']}")
        aggregate.update(name.encode() + b"\0" + bytes.fromhex(got_sha))
    got_aggregate = aggregate.hexdigest()
    expected_aggregate = manifest["aggregate_name_hash_sha256"]
    if got_aggregate != expected_aggregate:
        failures.append(f"AGGREGATE {got_aggregate} != {expected_aggregate}")
    if failures:
        print("FAIL_RELEASE_MANIFEST")
        print("\n".join(failures))
        return 1
    print("PASS_RELEASE_MANIFEST")
    print(f"files={len(wanted)}")
    print(f"bytes={manifest['total_bytes']}")
    print(f"aggregate_name_hash_sha256={got_aggregate}")
    print(f"checkpoint_sha256={manifest['checkpoint_sha256']}")
    print(f"logical_weights_sha256={manifest['logical_weights_sha256']}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
