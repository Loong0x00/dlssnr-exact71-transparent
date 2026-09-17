# Verification guide

Verification is layered. Do not collapse an integrity check, a transparent
self-test, and an original-DLL oracle into one claim.

## 1. Release archive integrity

After downloading the release files, the expected archive SHA-256 is:

```text
b0ce8f73d75d6189dd17f62f25433b84f8e2ce3c760fdf0a08982b688ee9c375
```

Verify and extract:

```bash
sha256sum -c SHA256SUMS
unzstd exactnr71-sm120-public-20260915.tar.zst
# or: tar --zstd -xf exactnr71-sm120-public-20260915.tar.zst
python3 tools/verify_release_manifest.py \
  nr-exact71-sm120-public-package-20260915
```

Expected package identity:

```text
files (manifest domain): 302
total bytes: 591757959
aggregate_name_hash_sha256:
59a1fbc2b23dc0fd3b3b1b7a8e0ac5e297bc1d79aa732f3c95243f73dc9cbba7
checkpoint_sha256:
890a53f05473dd6fd0980d1ca8501cfd84c1f6d7330b9531ca0981b6a6778d0e
logical_weights_sha256:
399d900614665918b4f4d51cea0eebc4d8a3a3752c3ab19847e9206f50f2d043
```

Run this before the model verifiers: those verifiers intentionally write result
JSON files back into the extracted directory.

## 2. Independent transparent forward

Requirements:

- Python 3 (the release was exercised with Python 3.14);
- NumPy;
- roughly 1.2 GiB free disk space for archive plus extraction;
- patience: the scalar/exact CPU route is intentionally slow.

```bash
cd nr-exact71-sm120-public-package-20260915
python3 -B verify_public_model.py
```

The script:

1. loads frozen assets before arming its audit gate;
2. disables file, process, network, socket, and dynamic-loader APIs;
3. executes the complete transparent forward twice;
4. requires both outputs and the recorded original-DLL fixture to be byte-equal.

Expected SHA-256:

```text
571fa2100c1842deff8769510a78f90cbe84810d64bb3ea82eaba41cf1cd793e
```

Expected status:

```text
PASS_SINGLE_INDEPENDENT_TRANSPARENT_PUBLIC_NR71_SM120_FIRST_FRAME_BITWISE
```

## 3. Immediate second frame

```bash
python3 -B verify_public_temporal_model.py
```

Expected values:

```text
frame0_sha256 = 571fa2100c1842deff8769510a78f90cbe84810d64bb3ea82eaba41cf1cd793e
history_rgba16f_rtz_sha256 = 47c5e92a43d021e32860756d2395d908a484ab7d9ca425756168c91d5e3f41a4
frame1_sha256 = e451fcf93663340a587888fa244e6cb62bad3030a0e4b157ff42a9bc6ec2530f
```

Expected status:

```text
PASS_PURE_PUBLIC_TWO_FRAME_SM120_DLL_BITWISE
```

## 4. Optional original-DLL oracle

The repository does not distribute `nvngx_dlssnr.dll`. If you independently and
lawfully possess the exact target DLL, first verify:

```bash
sha256sum /path/to/nvngx_dlssnr.dll
# e16bcf15e16e13f527491cdf7845b2fe6521a738d8f7c9c721866a8496e1fc8e
```

On Linux this optional helper requires an NVIDIA CUDA-capable system, Wine, and
a MinGW cross compiler. Use a clean Wine prefix and stop games first:

```bash
export WINEPREFIX="$PWD/oracle-wineprefix"
tools/run_native_oracle.sh \
  ./nr-exact71-sm120-public-package-20260915 \
  /path/to/nvngx_dlssnr.dll
```

The helper builds `tools/cuda_feature_probe.cpp`, evaluates Feature 18 once and
then twice in one persistent feature instance, and performs strict bytewise
comparisons against the frozen reset- and second-frame fixtures.

Expected final status:

```text
PASS_NATIVE_DLL_ORACLE_FIRST_AND_SECOND_FRAME
```

A successful API return alone is not parity. Require exact DLL identity, complete
readback size, output SHA, and zero differing bytes.

## 5. Evidence levels

- **Manifest pass:** files were not changed relative to this package.
- **Transparent pass:** the published Python/NumPy implementation reproduces the
  frozen expected bytes while its forward audit blocks external execution.
- **Native oracle pass:** your supplied target DLL reproduces those same bytes on
  your system.

None of these establish behavior outside the frozen 512×512 first/two-frame
contract.
