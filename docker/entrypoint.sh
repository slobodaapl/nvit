#!/bin/sh

set -xeuo pipefail

# Install the mounted project in editable mode
uv sync

# Execute passed command
exec "$@"
