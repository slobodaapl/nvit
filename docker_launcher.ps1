# Default values
param(
    [int]$num_gpus = 1,
    [string]$wandb_token = ""
)

# Load environment variables from .env file if it exists
if (Test-Path .env) {
    Get-Content .env | ForEach-Object {
        if ($_ -match '^([^=]+)=(.*)$') {
            [Environment]::SetEnvironmentVariable($matches[1], $matches[2])
        }
    }
}

# Run docker container with local directory mounted and execute training command
docker run --rm `
    --gpus all `
    -v ${PWD}:/app `
    -w /app `
    -e WANDB_API_KEY=$wandb_token `
    --env-file .env `
    nvit:latest `
    torchrun --nnodes 1 --nproc_per_node $num_gpus --rdzv_endpoint=localhost:29501 nvit/train.py
