# Read number of GPUs from args on command line, if present, otherwise default to 1
param(
    [int]$num_gpus = 1
)

# Run docker container with local directory mounted and execute training command
docker run --rm `
    --gpus all `
    -v ${PWD}:/app `
    -w /app `
    nvit:latest `
    torchrun --nnodes 1 --nproc_per_node $num_gpus --rdzv_endpoint=localhost:29501 nvit/train.py
