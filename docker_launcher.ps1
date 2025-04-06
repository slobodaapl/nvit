# Default values
param(
    [int]$num_gpus = 1,
    [string]$visible_gpus = "0",
    [switch]$detached = $false
)

# Load environment variables from .env file if it exists
if (Test-Path .env) {
    Get-Content .env | ForEach-Object {
        if ($_ -match '^([^=]+)=(.*)$') {
            [Environment]::SetEnvironmentVariable($matches[1], $matches[2])
        }
    }
}

New-Item -ItemType Directory -Force -Path "out" | Out-Null
icacls "out" /grant Everyone:F /T

# Build podman run command
$podman_cmd = "podman run --rm"
if ($detached) {
    $podman_cmd = "$podman_cmd -d"
}

# Run podman container with local directory mounted and execute training command
Invoke-Expression "$podman_cmd ``
    --device nvidia.com/gpu=$visible_gpus ``
    --shm-size=16gb ``
    -v ${PWD}:/app ``
    -w /app ``
    -e HOME=/app ``
    -e TORCHINDUCTOR_CACHE_DIR=/app/.cache ``
    -e NCCL_TIMEOUT=1200 ``
    -e NCCL_DEBUG=INFO ``
    --env-file .env ``
    nvit:latest ``
    torchrun --nnodes 1 --nproc_per_node $num_gpus --rdzv_endpoint=localhost:29501 nvit/train.py"
