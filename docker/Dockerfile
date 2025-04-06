# syntax=docker/dockerfile:1.4
FROM nvcr.io/nvidia/pytorch:24.10-py3

ENV DOCKER_BUILDKIT=1

# Install system dependencies
RUN apt-get update -y --no-install-recommends && \
    apt-get upgrade -y --no-install-recommends && \
    apt-get install -y --no-install-recommends \
    git \
    curl \
    nano && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Install uv
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="$HOME/.local/bin:$PATH"

WORKDIR /app

# Copy and set entrypoint
COPY docker/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh
ENTRYPOINT ["/entrypoint.sh"]

# Default command
CMD ["uv", "run", "python", "-m", "nvit.train"]