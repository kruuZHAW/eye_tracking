#!/usr/bin/env bash
set -euo pipefail

# Resolve paths
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PARENT_DIR="$(dirname "$SCRIPT_DIR")"

# --- Find and activate an existing venv in the parent directory ---
# You can override with: VENV_PATH=/path/to/venv ./setup.sh
CANDIDATES=()
if [[ "${VENV_PATH:-}" != "" ]]; then
  CANDIDATES+=("$VENV_PATH")
fi
CANDIDATES+=("$PARENT_DIR/.venv" "$PARENT_DIR/env" "$PARENT_DIR/venv")

VENV_ACTIVATE=""
for v in "${CANDIDATES[@]}"; do
  if [[ -f "$v/bin/activate" ]]; then
    VENV_ACTIVATE="$v/bin/activate"
    break
  fi
done

if [[ -z "$VENV_ACTIVATE" ]]; then
  echo "Error: couldn't find a Python venv in the parent directory."
  echo "Looked for: .venv, env, venv under: $PARENT_DIR"
  echo "Tip: run with VENV_PATH=/absolute/path/to/venv ./setup.sh"
  exit 1
fi

# shellcheck disable=SC1090
source "$VENV_ACTIVATE"
echo "Using venv: $(dirname "$VENV_ACTIVATE")"

# --- Ensure protoc is available ---
if ! command -v protoc >/dev/null 2>&1; then
  echo "Error: 'protoc' not found on PATH. Please install the Protocol Buffers compiler."
  echo "  • On Ubuntu/Debian: sudo apt-get install -y protobuf-compiler"
  echo "  • Or use a prebuilt binary from https://github.com/protocolbuffers/protobuf/releases"
  exit 1
fi

# --- Generate Python classes from .proto files ---
PROTO_DIR="$SCRIPT_DIR/proto"
GEN_DIR="$SCRIPT_DIR/gen"
mkdir -p "$GEN_DIR"

shopt -s nullglob
PROTO_FILES=("$PROTO_DIR"/*.proto)
if (( ${#PROTO_FILES[@]} == 0 )); then
  echo "Warning: no .proto files found in: $PROTO_DIR"
else
  echo "Generating Python code to: $GEN_DIR"
  protoc -I="$PROTO_DIR" -I="$HOME/protoc-include/include" --python_out="$GEN_DIR" "${PROTO_FILES[@]}"
  echo "Done."
fi

# ensure gen is a package
touch "$GEN_DIR/__init__.py"

# PAtch absolute imports to relative instide gen
export GEN_DIR  # visible to the Python snippet below
python - <<'PY'
import os, re, pathlib
gen = pathlib.Path(os.environ["GEN_DIR"])
pat = re.compile(r'^(import (?!google\.)(([A-Za-z0-9_]+)_pb2) as )', re.M)
for p in gen.glob("*_pb2.py"):
    s = p.read_text()
    s2 = pat.sub(lambda m: f'from . import {m.group(2)} as ', s)
    if s2 != s:
        p.write_text(s2)
        print("patched", p.name)
PY

# Add "events" dir to venv's site-package via .pth 
SITEPKG="$("$VIRTUAL_ENV/bin/python" -c 'import site; print([p for p in site.getsitepackages() if "site-packages" in p][0])')"
echo "$SCRIPT_DIR" > "$SITEPKG/gen_path.pth"

# ---------- Smoke test ----------
"$VIRTUAL_ENV/bin/python" - <<'PY'
import gen.messages_pb2 as m
print("Import OK:", m.__name__)
PY

echo "Done."