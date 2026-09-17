# Third-party references, provenance, and rights boundary

This is a research publication, not an NVIDIA product. NVIDIA, DLSS, NGX, CUDA,
and related marks belong to their respective owners. No endorsement or permission
is implied.

## Target identity

The numerical target was a locally held `nvngx_dlssnr.dll` with SHA-256:

```text
e16bcf15e16e13f527491cdf7845b2fe6521a738d8f7c9c721866a8496e1fc8e
```

The DLL itself, original CUBINs, and original PTX are not included in this Git
repository or release archive.

## Numerical assets

The runnable release contains recovered numerical weights, layouts, fixtures,
and approximation-domain tables needed to reproduce the frozen transparent
forward. These artifacts were derived from the target binary. This repository
does not claim ownership of underlying vendor model material and does not grant
a license to it. Users are responsible for determining whether possession,
download, analysis, or use is permitted in their jurisdiction and under the
terms applying to software they possess.

## Community research references

Topology and extraction research was compared against public community work,
including pinned snapshots of:

- `iamwavecut/MLX-DLSS`, commit
  `0ca2deab092fe6f3e331bf4f616271dbc64521d0` (repository code states
  Apache-2.0; that license does not cover NVIDIA binaries or extracted weights);
- `taowen/dlss5-as-inpainting`, commit
  `59a414cfd09676f3a09ff042330b5dc8407ebc79` (no license was identified in
  the audited snapshot, so no license is asserted here).

Community graphs were treated as research scaffolding, not as proof of the final
skip edges or numerical behavior. Corrected skip routing and native-arithmetic
checks were established separately for the frozen target.

## Repository licensing

No blanket open-source license is attached to the repository because the tree
contains a mixture of newly generated research code, potentially adapted
research structure, and separately controlled vendor-derived numerical material.
Absence of a license is intentional and grants no permission beyond rights that
applicable law independently provides.

If you need redistribution or commercial-use rights, obtain qualified legal
advice and permissions from the relevant rightsholders rather than relying on
this file.
