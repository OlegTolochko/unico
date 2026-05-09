#!/usr/bin/env bash

set -x

UNICO_USE_HELPER_SCRIPT=1 uv run python inference.py "$@"
