echo "Warning: Environment file $env_file does not exist or is not accessible"
fi

# Run podman container with local directory mounted and execute training command
podman run \
    ${remove_container:+--rm} \
    --gpus "\"device=$visible_gpus\"" \
    --shm-size=16gb \
    -v $(pwd):/app \
    -w /app \
    -e HOME=/app \
    -e TORCHINDUCTOR_CACHE_DIR=/app/.cache \
    -e NCCL_TIMEOUT=1200 \
    -e NCCL_DEBUG=INFO \
    --env-file "$env_file" \
    --user ${USER_ID}:${GROUP_ID} \
    ${detached:+-d} \
    nvit:latest \
    torchrun --nnodes 1 --nproc_per_node $num_gpus --rdzv_endpoint=localhost:29501 nvit/train.py 