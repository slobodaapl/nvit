# Run podman container with local directory mounted and open shell
podman run --rm -it `
    --gpus all `
    --shm-size=16gb `
    -v "${PWD}:/app" `
    -w /app `
    nvit:latest `
    /bin/bash
