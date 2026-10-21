# AdLens end-to-end pipeline image (crawler + OCR + detection).
#
# Reconstructed from the layer history of the images published at
# https://hub.docker.com/r/racro/adlens (tags: cpu_v1/latest, gpu_v1/gpu) —
# the original Dockerfile used to build those images was never committed to
# this repo. Building this file reproduces the same image contents; it is
# not byte-identical to the published images.
#
# Build:
#   docker build --build-arg DEVICE=cpu -t adlens:cpu .
#   docker build --build-arg DEVICE=gpu -t adlens:gpu .

# google-chrome-stable has no arm64 build, so this image is amd64-only —
# pinned here so it also builds (via emulation) on Apple Silicon / arm64 hosts.
FROM --platform=linux/amd64 python:3.12

ARG DEVICE=cpu

WORKDIR /app

# Chrome/Puppeteer runtime deps + Node.js (crawler) + Xvfb (headless Chrome display)
RUN apt-get update && apt-get install -y --no-install-recommends \
        curl gnupg build-essential git xvfb zstd \
        ca-certificates fonts-liberation \
        libasound2 libatk-bridge2.0-0 libatk1.0-0 libatspi2.0-0 \
        libcups2 libdbus-1-3 libdrm2 libgbm1 libgtk-3-0 \
        libnspr4 libnss3 libx11-xcb1 libxcomposite1 libxdamage1 \
        libxfixes3 libxkbcommon0 libxrandr2 libxss1 libxtst6 xdg-utils \
    && curl -fsSL https://deb.nodesource.com/setup_18.x | bash - \
    && apt-get install -y --no-install-recommends nodejs \
    && rm -rf /var/lib/apt/lists/*

# Google Chrome (stable) — used by the Puppeteer crawler
RUN curl -fsSL https://dl-ssl.google.com/linux/linux_signing_key.pub \
        | gpg --dearmor -o /usr/share/keyrings/google-chrome.gpg \
    && echo "deb [arch=amd64 signed-by=/usr/share/keyrings/google-chrome.gpg] http://dl.google.com/linux/chrome/deb/ stable main" \
        > /etc/apt/sources.list.d/google-chrome.list \
    && apt-get update && apt-get install -y --no-install-recommends google-chrome-stable \
    && rm -rf /var/lib/apt/lists/*

ENV PUPPETEER_SKIP_DOWNLOAD=true

# Ollama CLI/daemon — lets the container run its own local model server when
# no host Ollama is reachable at $OLLAMA_URL (see docker-entrypoint.sh)
RUN curl -fsSL https://ollama.com/install.sh | sh

COPY requirements.txt requirements.txt
COPY ocr/requirements.txt ocr/requirements.txt
# torch goes in first, from the wheel index matching DEVICE, so pip never resolves a
# CUDA build newer than paddle/the driver expects (symptom of a mismatch:
# "undefined symbol: ncclCommResume" on `import torch`).
RUN pip install --no-cache-dir --upgrade pip \
    && if [ "$DEVICE" = "gpu" ]; then \
           pip install --no-cache-dir "torch>=2.4,<2.6" --index-url https://download.pytorch.org/whl/cu121 ; \
       else \
           pip install --no-cache-dir "torch>=2.4,<2.6" --index-url https://download.pytorch.org/whl/cpu ; \
       fi \
    && pip install --no-cache-dir -r requirements.txt \
    && pip install --no-cache-dir -r ocr/requirements.txt

# GPU builds swap in paddlepaddle-gpu (see ocr/requirements.txt for the CPU/GPU note)
RUN if [ "$DEVICE" = "gpu" ]; then \
        pip uninstall -y paddlepaddle && \
        pip install --no-cache-dir paddlepaddle-gpu==3.1.0 \
            --extra-index-url https://www.paddlepaddle.org.cn/packages/stable/cu123/ ; \
    fi

# Fail the build if core imports are broken. paddle/paddleocr require libcuda.so.1 at
# import time, which is only available at runtime (--gpus all), not during docker build.
RUN python -c "import torch, ollama, sentence_transformers; print('torch', torch.__version__)" \
    && if [ "$DEVICE" != "gpu" ]; then python -c "import paddle, paddleocr; print('paddle ok')"; fi

COPY crawler/ crawler/
RUN cd crawler && npm ci

COPY . .

ENV PYTHONUNBUFFERED=1 \
    DEVICE=${DEVICE} \
    OLLAMA_URL=http://host.docker.internal:11434

ENTRYPOINT ["/app/docker-entrypoint.sh"]
CMD ["--help"]
