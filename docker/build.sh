#!/bin/bash

# Build the Docker image from parent directory context
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
docker build -t nvit:latest -f "$SCRIPT_DIR/Dockerfile" "$SCRIPT_DIR/.."
