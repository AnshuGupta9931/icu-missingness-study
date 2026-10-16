#!/usr/bin/env bash
# Install the OpenMP runtime that XGBoost and LightGBM need, into this
# project's virtual environment.
#
#     bash scripts/setup_libomp_macos.sh
#
# Why this exists
# ---------------
# XGBoost's and LightGBM's macOS wheels do not ship libomp.dylib. They look for
# it at Homebrew's prefix, and without it `import xgboost` fails with
# "Library not loaded: @rpath/libomp.dylib".
#
# The normal fix is `brew install libomp`. Use that if you can. On a machine
# where the Homebrew prefix is owned by a different user account, brew refuses
# to run and taking ownership of it would affect that other account - so this
# script installs libomp into .venv/ instead, leaving the rest of the system
# untouched. Everything it changes lives inside .venv and disappears with it.
#
# What it does, in order:
#   1. downloads the libomp bottle from Homebrew's registry
#   2. verifies the SHA-256 against the digest Homebrew publishes
#   3. copies libomp.dylib into .venv/lib and sets its install name
#   4. adds .venv/lib to the library search path of the two wheels
#   5. re-signs all three (patching a dylib invalidates its signature, and
#      unsigned dylibs are killed on Apple Silicon)
#
# Re-run it after reinstalling or upgrading xgboost or lightgbm, which restores
# the unpatched wheel libraries.
set -euo pipefail

if [[ "$(uname -s)" != "Darwin" ]]; then
  echo "This script is macOS-only. On Linux install libgomp with your package manager."
  exit 1
fi

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV="${PROJECT_ROOT}/.venv"
VENV_LIB="${VENV}/lib"

if [[ ! -d "${VENV}" ]]; then
  echo "No virtual environment at ${VENV}. Create it first:"
  echo "  python3 -m venv .venv && .venv/bin/pip install -r requirements.txt"
  exit 1
fi

if [[ -n "${1:-}" ]]; then
  ARCH_KEY="$1"
else
  case "$(sw_vers -productVersion | cut -d. -f1)" in
    26) ARCH_KEY="arm64_tahoe" ;;
    15) ARCH_KEY="arm64_sequoia" ;;
    14) ARCH_KEY="arm64_sonoma" ;;
    13) ARCH_KEY="arm64_ventura" ;;
    *)  ARCH_KEY="arm64_sequoia" ;;
  esac
  if [[ "$(uname -m)" != "arm64" ]]; then
    ARCH_KEY="${ARCH_KEY/arm64/x86_64}"
  fi
fi
echo "macOS $(sw_vers -productVersion) on $(uname -m) -> bottle '${ARCH_KEY}'"

WORK="$(mktemp -d)"
trap 'rm -rf "${WORK}"' EXIT

echo "resolving the current libomp bottle..."
curl -fsSL --max-time 120 -o "${WORK}/libomp.json" https://formulae.brew.sh/api/formula/libomp.json

read -r VERSION URL EXPECTED_SHA < <(
  "${VENV}/bin/python" - "${WORK}/libomp.json" "${ARCH_KEY}" <<'PY'
import json, sys
meta = json.load(open(sys.argv[1]))
key = sys.argv[2]
files = meta["bottle"]["stable"]["files"]
if key not in files:
    sys.exit(f"no bottle '{key}'; available: {', '.join(sorted(files))}")
print(meta["versions"]["stable"], files[key]["url"], files[key]["sha256"])
PY
)
echo "  libomp ${VERSION}"

echo "downloading..."
curl -fsSL --max-time 600 -H "Authorization: Bearer QQ==" -o "${WORK}/bottle.tar.gz" "${URL}"

ACTUAL_SHA="$(shasum -a 256 "${WORK}/bottle.tar.gz" | cut -d' ' -f1)"
if [[ "${ACTUAL_SHA}" != "${EXPECTED_SHA}" ]]; then
  echo "CHECKSUM MISMATCH - refusing to install"
  echo "  expected ${EXPECTED_SHA}"
  echo "  got      ${ACTUAL_SHA}"
  exit 1
fi
echo "  sha256 verified against Homebrew's published digest"

mkdir -p "${WORK}/extract"
tar -xzf "${WORK}/bottle.tar.gz" -C "${WORK}/extract"
DYLIB="$(find "${WORK}/extract" -name libomp.dylib -type f | head -1)"
[[ -n "${DYLIB}" ]] || { echo "libomp.dylib not found inside the bottle"; exit 1; }

mkdir -p "${VENV_LIB}"
cp -f "${DYLIB}" "${VENV_LIB}/libomp.dylib"
chmod u+w "${VENV_LIB}/libomp.dylib"
# The bottle ships an unrelocated install name; the wheels ask for
# @rpath/libomp.dylib, so make the name they ask for the name it has.
install_name_tool -id @rpath/libomp.dylib "${VENV_LIB}/libomp.dylib" 2>/dev/null
echo "installed ${VENV_LIB}/libomp.dylib"

PATCHED=("${VENV_LIB}/libomp.dylib")
for REL in \
  "python3*/site-packages/xgboost/lib/libxgboost.dylib" \
  "python3*/site-packages/lightgbm/lib/lib_lightgbm.dylib"
do
  for LIB in ${VENV_LIB}/${REL}; do
    [[ -f "${LIB}" ]] || continue
    if otool -l "${LIB}" | grep -q "path ${VENV_LIB}$"; then
      echo "  already patched: $(basename "${LIB}")"
    else
      install_name_tool -add_rpath "${VENV_LIB}" "${LIB}" 2>/dev/null
      echo "  added search path to $(basename "${LIB}")"
    fi
    PATCHED+=("${LIB}")
  done
done

for LIB in "${PATCHED[@]}"; do
  codesign --force --sign - "${LIB}" >/dev/null 2>&1 || true
done
echo "re-signed ${#PATCHED[@]} libraries"

echo
echo "verifying..."
"${VENV}/bin/python" - <<'PY'
import sys
failed = []
for name in ("xgboost", "lightgbm", "catboost"):
    try:
        mod = __import__(name)
        print(f"  {name:10s} {mod.__version__}")
    except Exception as exc:
        failed.append(name)
        print(f"  {name:10s} FAILED: {type(exc).__name__}: {exc}")
sys.exit(1 if failed else 0)
PY

echo
echo "OpenMP runtime is in place."
