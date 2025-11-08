# Use Bookworm (Debian 12) base for stability and latest mirrors
FROM python:3.10-bookworm

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# Update mirrors, install core packages, and clean up
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        build-essential \
        git \
        wget \
        curl \
        ca-certificates \
        libopenblas-dev \
        libblas-dev \
        libatlas-base-dev && \
    apt-get clean && rm -rf /var/lib/apt/lists/*
