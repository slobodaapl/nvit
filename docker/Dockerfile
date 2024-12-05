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

# Install poetry
ENV PATH="/root/.local/bin:$PATH"
RUN curl -sSL https://install.python-poetry.org | python3 -
RUN poetry self add poetry-plugin-export
RUN poetry config virtualenvs.create false


WORKDIR /setup

# Export requirements without torch packages
COPY pyproject.toml poetry.lock /setup/
RUN poetry export --without-hashes -f requirements.txt --output requirements.txt && \
    sed -i '/^torch/d' requirements.txt && \
    sed -i '/^torchvision/d' requirements.txt

# Install Python dependencies
RUN pip install -r requirements.txt --no-cache-dir --no-build-isolation

# Set working directory
WORKDIR /app

# Copy and set entrypoint
COPY docker/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh
ENTRYPOINT ["/entrypoint.sh"]

# Default command
CMD ["python", "-m", "nvit.train"]