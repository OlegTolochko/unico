#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

if ! command -v uv >/dev/null 2>&1; then
    echo "uv is required. Install it from https://docs.astral.sh/uv/ first." >&2
    exit 1
fi

echo ">>> Syncing Python environment with uv..."
uv sync

echo ">>> Building chamfer_dist extension..."
uv pip install --project "$SCRIPT_DIR" --reinstall --no-build-isolation --no-deps -e "$SCRIPT_DIR/extensions/chamfer_dist"
echo ">>> chamfer_dist built successfully!"

echo ">>> Building pointnet2_ops_lib extension..."
uv pip install --project "$SCRIPT_DIR" --reinstall --no-build-isolation --no-deps -e "$SCRIPT_DIR/extensions/pointnet2_ops_lib"
echo ">>> pointnet2_ops_lib built successfully!"

echo ">>> Environment ready. Use: uv run python main.py ..."
