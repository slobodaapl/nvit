# Build the Docker image from parent directory context
$SCRIPT_DIR = Split-Path -Parent $MyInvocation.MyCommand.Path
docker build -t nvit:latest -f "$SCRIPT_DIR/Dockerfile" "$SCRIPT_DIR/.."
