# ExactNR71 — transparent DLSS NR 310.8 research reconstruction

> [!CAUTION]
> **AI-assisted, untrusted research artifact.** A large part of the implementation,
> reverse-engineering notes, and verification code was generated or edited by AI
> under human direction. Human review is incomplete. The repository may contain
> incorrect names, wrong interpretations, latent bugs, or conclusions that do not
> generalize. **Do not treat it as authoritative, secure, production-ready, or a
> drop-in DLSS replacement.** Reproduce every claim you rely on.

This repository publishes a transparent Python/NumPy reconstruction of one
narrow path through NVIDIA DLSS Neural Rendering 310.8. It exposes a 71-block
forward graph and 649 logical tensors. The transparent `forward` does not call
or wrap a DLL, CUBIN, PTX, VM/interpreter, CUDA, subprocess, network, or file API.
Constructor-time code loads frozen numerical assets.

This is **not a complete reconstruction of the DLL**. It is a bounded research
result for one pinned binary and contract.

## What is actually proven

Target DLL SHA-256:

```text
e16bcf15e16e13f527491cdf7845b2fe6521a738d8f7c9c721866a8496e1fc8e
```

Pinned contract:

- public `NVSDK_NGX_CUDA_*`, Feature 18;
- 512×512 RGBA32F input/output;
- first/reset frame;
- immediately following `counter=1` frame with zero RG32F motion;
- no control mask;
- internal padded high extent 512×576 and ViT token count 96.

Frozen byte-exact results:

| Artifact | SHA-256 |
|---|---|
| input fixture | `78523ae1814f60c64b3f54819e67964385edecc49c22b290c4b0bac293ef6e09` |
| reset-frame output | `571fa2100c1842deff8769510a78f90cbe84810d64bb3ea82eaba41cf1cd793e` |
| RGBA16F RTZ history | `47c5e92a43d021e32860756d2395d908a484ab7d9ca425756168c91d5e3f41a4` |
| immediate second-frame output | `e451fcf93663340a587888fa244e6cb62bad3030a0e4b157ff42a9bc6ec2530f` |

The original DLL and transparent implementation produced identical 4,194,304
output bytes for those frozen cases. That does **not** establish parity for
other versions, resolutions, formats, counters ≥2, nonzero motion, masks, or
real Vulkan/D3D game resource lifecycles.

## Repository layout

- `reference/` — browseable Python source snapshot. Large numerical assets are
  intentionally not committed to Git.
- `evidence/` — frozen manifests and acceptance receipts.
- `tools/verify_release_manifest.py` — verifies every file in the runnable
  release archive.
- `tools/run_native_oracle.sh` — optional comparison against a DLL supplied by
  the user; the DLL is not included.
- GitHub **Releases** — complete runnable research package.

## Quick verification

Download and extract the release archive, then run:

```bash
python3 tools/verify_release_manifest.py ./nr-exact71-sm120-public-package-20260915

cd nr-exact71-sm120-public-package-20260915
python3 -B verify_public_model.py
python3 -B verify_public_temporal_model.py
```

Expected terminal statuses:

```text
PASS_RELEASE_MANIFEST
PASS_SINGLE_INDEPENDENT_TRANSPARENT_PUBLIC_NR71_SM120_FIRST_FRAME_BITWISE
PASS_PURE_PUBLIC_TWO_FRAME_SM120_DLL_BITWISE
```

The first transparent verifier executes the full model twice and can take many
minutes on a CPU. The temporal verifier executes two frames. See
[`docs/VERIFY.md`](docs/VERIFY.md) for archive hashes, exact commands, output
hashes, forward-I/O auditing, and the optional original-DLL oracle.

## Minimal API

```python
import numpy as np
from model_public_sm120 import ExactNR71PublicSM120

model = ExactNR71PublicSM120()
color = np.fromfile("fixtures/public-input.rgba32f", "<f4").reshape(512, 512, 4)
result = model.forward(color, tone=.25, structure=.75, style=0,
                       skin=-1., use_auto_mask=False)
rgba_u32 = result["output_RGBA_f32_bits"]
```

## Rights and provenance

No NVIDIA DLL is included. Numerical assets in the runnable research package
were recovered from a pinned locally held binary and may be subject to rights
or contractual restrictions not granted by this repository. No license is
asserted for vendor-derived material, and no permission from NVIDIA is implied.
See [`docs/THIRD_PARTY_AND_RIGHTS.md`](docs/THIRD_PARTY_AND_RIGHTS.md).

No warranty is provided. Use only where permitted by applicable law and the
terms governing software you possess.
