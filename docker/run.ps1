# Run docker container with local directory mounted and open shell
docker run --rm -it `
    --gpus all `
    --shm-size=16gb `
    -v "${PWD}:/app" `
    -w /app `
    nvit:latest `
    /bin/bash
