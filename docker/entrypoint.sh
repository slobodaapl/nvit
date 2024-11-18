#!/bin/sh

set -xeuo pipefail

# Install the mounted project in editable mode, for both development and running
pip install -e .

# Execute passed command
exec "$@"
