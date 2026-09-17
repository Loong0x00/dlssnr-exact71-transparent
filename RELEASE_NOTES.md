# v0.1.0 research preview

**AI-assisted and untrusted. No warranty. Independently verify all claims.**

This release contains the complete frozen runnable package for the narrow
ExactNR71 SM120 public-CUDA contract described in the repository README.

## Assets

- `exactnr71-sm120-public-20260915.tar.zst` — complete 591,757,959-byte
  uncompressed package, including recovered numerical assets and frozen fixtures;
- `PUBLIC_PACKAGE_MANIFEST.json` — per-file sizes and SHA-256 values;
- `SHA256SUMS` — release-asset hashes.

Archive SHA-256:

```text
b0ce8f73d75d6189dd17f62f25433b84f8e2ce3c760fdf0a08982b688ee9c375
```

## Accepted narrow results

```text
reset output:
571fa2100c1842deff8769510a78f90cbe84810d64bb3ea82eaba41cf1cd793e

immediate second-frame output:
e451fcf93663340a587888fa244e6cb62bad3030a0e4b157ff42a9bc6ec2530f
```

The claimed scope is only 512×512 RGBA32F, reset then immediate counter=1,
zero motion, no control mask, against the pinned DLL hash documented in README.
Nothing broader is guaranteed.

## Rights warning

The archive contains numerical material recovered from a locally held vendor
binary. No rights or license to vendor-derived material are granted. The DLL,
original CUBINs, and original PTX are not included. See
`docs/THIRD_PARTY_AND_RIGHTS.md` before downloading or redistributing.
