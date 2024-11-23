#!/bin/sh

set -xeuo pipefail

# Install the mounted project in editable mode, for both development and running
pip install --no-deps --user --root-user-action ignore -e .

# Execute passed command
exec "$@"
