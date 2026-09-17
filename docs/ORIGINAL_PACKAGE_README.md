# Transparent NR71 SM120 public-path package

## API

```python
import numpy as np
from model_public_sm120 import ExactNR71PublicSM120

model = ExactNR71PublicSM120()
color = np.asarray(..., dtype=np.float32)  # exact shape (512, 512, 4)
result = model.forward(
    color,
    tone=0.25,
    structure=0.75,
    style=0,
    skin=-1.0,
    use_auto_mask=False,
)
rgba_f32_bits = result["output_RGBA_f32_bits"]  # uint32 (512,512,4)

# Reset frame followed by the immediate counter=1 frame, with zero motion:
two = model.forward_two_frames(color)  # optional distinct color1 is accepted
t0_bits = two["frame0"]["output_RGBA_f32_bits"]
t1_bits = two["frame1"]["output_RGBA_f32_bits"]
history_half_bits = two["history_RGBA16F_bits"]
```

`forward` is transparent Python/NumPy arithmetic. It does not call a DLL,
CUBIN, PTX, VM, CUDA, GPU, subprocess, network, or file API. Constructor I/O
loads the pinned checkpoint and recovered SM120 approximation tables.

All model and temporal forwards are transparent and fail closed outside their
stated resource/numeric domains. The public temporal transition is explicit:
RGBA32F output is converted by formatted-surface **RTZ** to RGBA16F, then read
by normalized linear history sampling. Under the proven zero-motion route all
history coordinates are exact texel centers.

Additional interfaces:

- `ExactNR71PublicSM120.logical_weights()` — read-only 649 logical tensors.
- `ExactNR71PublicSM120.execution_contract()` — synchronization/state contract.
- `keep_boundaries=True` — retained logical stage/boundary probes.

## Proven public route

Pinned DLL SHA-256:
`e16bcf15e16e13f527491cdf7845b2fe6521a738d8f7c9c721866a8496e1fc8e`

For the frozen deterministic RGBA32F input and first/reset invocation:

- public API: `NVSDK_NGX_CUDA_*`, feature 18;
- source/visible output: 512×512;
- internal high extent: 512×576;
- bottleneck allocation: 16×20 (logical downsample width 18 plus two clear-padding columns);
- pool/head: 8×12;
- ViT: N96;
- Color is live at pre0 T0 and post70 A;
- reset gates pre0 temporal resources and post70 B/C null.

The original DLL, an instrumented observer run, and this transparent model all
produce the same 4,194,304 bytes:

`571fa2100c1842deff8769510a78f90cbe84810d64bb3ea82eaba41cf1cd793e`

Across pre0, encoder, split, N96 ViT, decoder and final public surface, 177
complete first-frame boundaries compare byte-equal. See the sibling evidence
file `../nr-native-equivalence-matrix-main-20260914/PUBLIC_FULL71_ACCEPTANCE.json`.

The immediate public second frame (`reset=0`, counter 1, live RGBA16F history,
live zero RG32F motion) is also byte-equal:

- second pre0 primary: `597bd892d630ce4d098b4b6e81165b5172ce4d67908e07aa1df9afb10c7fbf81`;
- second pre0 skip: `cad1bbd53c6c7508eae4b503127a7467f57c1e84bd393b4010c257d72051fe93`;
- formatted RGBA16F-RTZ history: `47c5e92a43d021e32860756d2395d908a484ab7d9ca425756168c91d5e3f41a4`;
- final 4,194,304-byte surface: `e451fcf93663340a587888fa244e6cb62bad3030a0e4b157ff42a9bc6ec2530f`.

`PUBLIC_TWO_FRAME_ACCEPTANCE.json` records the single package API run and its
zero forward-I/O audit. The sibling `PUBLIC_FRAME2_PRE_SM120_COMPARISON.json`
and `PUBLIC_FRAME2_SM120_COMPARISON.json` retain native-DLL comparisons.

## Scope boundary

Proven public-DLL scope is the pinned 512×512 RGBA32F fixture: reset frame and
its immediate second frame, zero RG32F motion, and no control mask. This does
not claim parity for frame counters beyond 1, nonzero motion, arbitrary
resolutions/formats, masks, or Vulkan/D3D game resource lifecycles.
