#!/bin/bash

# Run docker container with local directory mounted and open shell
docker run --rm -it \
    --gpus all \
    --shm-size=16gb \
    -v "$(pwd)":/app \
    -w /app \
    nvit:latest \
    /bin/bash
