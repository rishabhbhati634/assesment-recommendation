# Dockerfile — Render-ready, stable build using Ubuntu 22.04

FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# ---- System setup ----
WORKDIR /app

# Install Python, build tools, and FAISS dependencies
RUN apt-get update --allow-releaseinfo-change && \
    apt-get install -y --no-install-recommends \
        python3 python3-pip python3-venv \
        build-essential \
        git \
        wget \
        curl \
        ca-certificates \
        libopenblas-dev \
        libatlas-base-dev \
        libssl-dev \
        && apt-get clean && rm -rf /var/lib/apt/lists/*

# Make "python" & "pip" commands point to python3
RUN ln -sf /usr/bin/python3 /usr/bin/python && \
    ln -sf /usr/bin/pip3 /usr/bin/pip

# ---- Python setup ----
COPY requirements.txt /app/requirements.txt
RUN pip install --upgrade pip setuptools wheel && \
    pip install --no-cache-dir -r /app/requirements.txt

# ---- App code ----
COPY . /app

# Expose the port Render will bind to
EXPOSE 5000

# ---- Start command ----
CMD ["bash", "-lc", "gunicorn -w 4 -b 0.0.0.0:${PORT:-5000} backend.app:app"]
