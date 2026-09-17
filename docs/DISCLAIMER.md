# AI and reliability disclaimer

This project is an AI-assisted reverse-engineering research artifact.

- A large fraction of source, documentation, test code, naming, and analysis was
  produced or modified by AI under human direction.
- Human review is incomplete.
- Passing included tests only proves the exact frozen assertions they check.
- Labels assigned to unknown structures may be wrong even when arithmetic bytes
  happen to match.
- The implementation has not been audited for security, robustness, portability,
  numerical behavior outside the admitted domain, or production use.
- No claim is made that this reconstructs every path or feature of the original
  DLL.
- Do not use it for safety-critical, competitive, commercial, or production
  decisions without independent review and reproduction.

The strongest accepted statement is narrow: for one pinned DLL and two frozen
512×512 public-CUDA frames, the recorded transparent outputs match the recorded
original outputs byte-for-byte. All broader claims remain unproven.
