# Use full Debian base to avoid apt repo issues
FROM python:3.10-bullseye

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# Install build dependencies for faiss, transformers, etc.
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    wget \
    curl \
    ca-certificates \
    libopenblas-dev \
    libblas-dev \
    libatlas-base-dev \
    && apt-get clean && rm -rf /var/lib/apt/lists/*
