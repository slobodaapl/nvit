# Build the Docker image from parent directory context
$SCRIPT_DIR = Split-Path -Parent $MyInvocation.MyCommand.Path

# Enable Docker BuildKit
$env:DOCKER_BUILDKIT = 1

# Build the Docker image with GPU support using buildx
docker buildx build --build-arg BUILDKIT_INLINE_CACHE=1 `
    --platform=linux/amd64 `
    --output=type=docker `
    --build-context gpu=docker-image://nvcr.io/nvidia/pytorch:24.10-py3 `
    -t nvit:latest `
    -f "$SCRIPT_DIR/Dockerfile" "$SCRIPT_DIR/.."
