# Dockerfile - CPU FAISS + sentence-transformers + Flask (Gunicorn)
# Place at repository root

FROM python:3.10-slim

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# Install system-level dependencies required by faiss, torch, and builds
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    wget \
    curl \
    ca-certificates \
    libopenblas-dev \
    libblas-dev \
    libatlas-base-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy only requirements first to leverage Docker cache
COPY requirements.txt /app/requirements.txt

# Upgrade pip and install Python dependencies
RUN pip install --upgrade pip setuptools wheel \
 && pip install --no-cache-dir -r /app/requirements.txt

# Copy application code
COPY . /app

# Make sure data folder exists (if you plan to mount or download at runtime)
RUN mkdir -p /app/data

# Expose port. Render provides $PORT at runtime; default to 5000 locally.
EXPOSE 5000

# Use Gunicorn to serve the Flask app; bind to $PORT when provided by host.
# backend.app:app refers to Flask app instance in backend/app.py
CMD ["bash", "-lc", "gunicorn -w 4 -b 0.0.0.0:${PORT:-5000} backend.app:app"]
