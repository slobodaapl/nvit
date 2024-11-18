# Run docker container with local directory mounted and open shell
docker run --rm -it `
    --gpus all `
    -v "${PWD}:/app" `
    -w /app `
    nvit:latest `
    /bin/bash
