#!/usr/bin/env bash
# Optional original-DLL oracle. The user must supply their own DLL.
set -euo pipefail
if [[ $# -ne 2 ]]; then
  echo "usage: $0 EXTRACTED_PACKAGE_DIR /path/to/nvngx_dlssnr.dll" >&2
  exit 2
fi
ROOT=$(cd "$(dirname "$0")/.." && pwd)
PACKAGE=$(cd "$1" && pwd)
DLL=$(readlink -f "$2")
EXPECTED_DLL=e16bcf15e16e13f527491cdf7845b2fe6521a738d8f7c9c721866a8496e1fc8e
[[ $(sha256sum "$DLL" | awk '{print $1}') == "$EXPECTED_DLL" ]] || {
  echo "DLL hash mismatch; this verification is pinned to $EXPECTED_DLL" >&2; exit 3;
}
for x in wine x86_64-w64-mingw32-g++ python3; do command -v "$x" >/dev/null || { echo "missing $x" >&2; exit 4; }; done
# Fail closed if common game processes are active.
if ps -eo comm,args | grep -Eiv 'grep|run_native_oracle' | grep -Ei 'Cyberpunk2077|WutheringWaves|Client-Win64-Shipping|StarRail|YuanShen|ZenlessZoneZero' >/dev/null; then
  echo "game process detected; refusing GPU oracle run" >&2; exit 5
fi
if command -v nvidia-smi >/dev/null; then
  free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1 | tr -d ' ')
  (( free >= 4096 )) || { echo "less than 4 GiB VRAM free" >&2; exit 6; }
fi
BUILD="$ROOT/.oracle-build"; mkdir -p "$BUILD"
EXE="$BUILD/cuda_feature_probe.exe"
x86_64-w64-mingw32-g++ -std=c++20 -O2 -static -DWIN32_LEAN_AND_MEAN \
  "$ROOT/tools/cuda_feature_probe.cpp" -o "$EXE"
WORK=$(mktemp -d); trap 'rm -rf "$WORK"' EXIT
cp "$PACKAGE/fixtures/public-input.rgba32f" "$WORK/input.rgba32f"
w(){ printf 'Z:%s' "${1//\//\\}"; }
export WINEDEBUG=-all WINEDLLOVERRIDES='winedbg.exe=d' DISPLAY="${DISPLAY:-:0}"
run_case(){
  local iterations=$1 output=$2
  (cd "$WORK" && timeout --signal=TERM --kill-after=3 30s wine "$(w "$EXE")" "$(w "$DLL")" \
    "$(w "$WORK/input.rgba32f")" "$(w "$WORK/$output")" \
    512 512 1 1 .25 .75 -1 0 0 .5 -1 "$iterations")
}
run_case 1 frame0.rgba32f
python3 "$ROOT/tools/compare_raw.py" "$PACKAGE/fixtures/public-output.rgba32f" "$WORK/frame0.rgba32f"
run_case 2 frame1.rgba32f
python3 "$ROOT/tools/compare_raw.py" "$PACKAGE/fixtures/public-frame1-output.rgba32f" "$WORK/frame1.rgba32f"
echo PASS_NATIVE_DLL_ORACLE_FIRST_AND_SECOND_FRAME
