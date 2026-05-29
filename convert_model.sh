#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 2 ]]; then
    echo "Usage: $0 <input.gguf> <output.gguf>" >&2
    exit 1
fi

ROOT_DIR=$(cd -- "$(dirname -- "$0")" && pwd)
TOOL_PATH="$ROOT_DIR/build-host/bin/llama-encrypt"

if [[ ! -x "$TOOL_PATH" ]]; then
    echo "error: $TOOL_PATH not found or not executable" >&2
    echo "build it first with:" >&2
    echo "  cmake -B build-host -S . -DGGML_RKNPU_RE=ON -DLLAMA_CURL=OFF" >&2
    echo "  cmake --build build-host -j10 --target llama-encrypt" >&2
    exit 1
fi

exec "$TOOL_PATH" "$1" "$2"
