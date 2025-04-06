# Build the Podman image from parent directory context
$SCRIPT_DIR = Split-Path -Parent $MyInvocation.MyCommand.Path

# Build the Podman image with GPU support
podman build `
    --platform=linux/amd64 `
    -t nvit:latest `
    -f "$SCRIPT_DIR/Dockerfile" "$SCRIPT_DIR/.."
