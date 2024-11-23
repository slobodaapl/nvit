# Default values
param(
    [int]$num_gpus = 1,
    [string]$visible_gpus = "all",
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

# Get current user's UID and GID
$USER_ID = (Get-WmiObject -Class Win32_UserAccount -Filter "Name = '$env:USERNAME'").SID
$GROUP_ID = (Get-WmiObject -Class Win32_Group -Filter "Name = 'Users'").SID

# Build docker run command
$docker_cmd = "docker run --rm"
if ($detached) {
    $docker_cmd = "$docker_cmd -d"
}

# Run docker container with local directory mounted and execute training command
Invoke-Expression "$docker_cmd ``
    --gpus `"device=$visible_gpus`" ``
    -v ${PWD}:/app ``
    -w /app ``
    -e HOME=/app ``
    -e TORCHINDUCTOR_CACHE_DIR=/app/.cache ``
    -e NCCL_TIMEOUT=1200 ``
    -e NCCL_DEBUG=INFO ``
    --env-file .env ``
    --user ${USER_ID}:${GROUP_ID} ``
    nvit:latest ``
    torchrun --nnodes 1 --nproc_per_node $num_gpus --rdzv_endpoint=localhost:29501 nvit/train.py"
