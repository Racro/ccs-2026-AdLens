# AdLens artifact image.
#
# CPU (default):
#   docker build -t adlens .
#
# GPU (requires host NVIDIA driver + nvidia-container-toolkit, and
# `docker run --gpus all ...`):
#   docker build --build-arg DEVICE=gpu -t adlens:gpu .
#
# The `ollama` CLI/server binary is installed below, but model weights
# (qwen3.5:9b, gemma3:12b, gemma4:26b, translategemma:4b — tens of GB) are
# NOT baked into the image. Run scripts/setup_ollama.sh (in this container
# or on the host) before the translate/detect stages; point --ollama-url /
# OLLAMA_URL at wherever it ends up serving. See README.md.

FROM python:3.12-slim

ARG DEVICE=cpu

WORKDIR /app

# System deps: build tools, Node.js 18 (crawler), and the shared-lib set
# Chromium/puppeteer needs to launch under Xvfb on a headless host.
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

# Real Chrome for the crawler (puppeteer's own Chromium download is skipped
# below). Wherever this package places the binary, docker-entrypoint.sh
# resolves it at container start via `command -v` rather than a fixed path.
RUN curl -fsSL https://dl-ssl.google.com/linux/linux_signing_key.pub \
        | gpg --dearmor -o /usr/share/keyrings/google-chrome.gpg \
    && echo "deb [arch=amd64 signed-by=/usr/share/keyrings/google-chrome.gpg] http://dl.google.com/linux/chrome/deb/ stable main" \
        > /etc/apt/sources.list.d/google-chrome.list \
    && apt-get update && apt-get install -y --no-install-recommends google-chrome-stable \
    && rm -rf /var/lib/apt/lists/*

# Skip puppeteer's own Chromium download — we already have real Chrome above.
ENV PUPPETEER_SKIP_DOWNLOAD=true

# Ollama binary (server + CLI). Model weights are pulled at runtime, not here.
RUN curl -fsSL https://ollama.com/install.sh | sh

# Python dependencies (parent pipeline + OCR).
# torch is installed first from the wheel index matching DEVICE so that pip
# never resolves a CUDA build newer than the one paddle/the driver expects
# (a mismatch shows up as "undefined symbol: ncclCommResume" on import).
COPY requirements.txt requirements.txt
COPY ocr/requirements.txt ocr/requirements.txt
RUN pip install --no-cache-dir --upgrade pip \
    && if [ "$DEVICE" = "gpu" ]; then \
           pip install --no-cache-dir "torch>=2.4,<2.6" --index-url https://download.pytorch.org/whl/cu121 ; \
       else \
           pip install --no-cache-dir "torch>=2.4,<2.6" --index-url https://download.pytorch.org/whl/cpu ; \
       fi \
    && pip install --no-cache-dir -r requirements.txt \
    && pip install --no-cache-dir -r ocr/requirements.txt

# Swap in the GPU build of PaddlePaddle when requested. The CPU wheel
# installed above must be removed first — the two cannot coexist.
RUN if [ "$DEVICE" = "gpu" ]; then \
        pip uninstall -y paddlepaddle && \
        pip install --no-cache-dir paddlepaddle-gpu==3.1.0 \
            --extra-index-url https://www.paddlepaddle.org.cn/packages/stable/cu123/ ; \
    fi

# Fail the build (not the user's first run) if the core imports are broken.
RUN python -c "import torch, ollama, sentence_transformers, paddle, paddleocr; print('torch', torch.__version__)"

# Node dependencies (crawler).
COPY crawler/package.json crawler/package-lock.json crawler/
RUN cd crawler && npm ci

# Application code.
COPY . .

ENV PYTHONUNBUFFERED=1 \
    DEVICE=${DEVICE} \
    OLLAMA_URL=http://host.docker.internal:11434

ENTRYPOINT ["/app/docker-entrypoint.sh"]
CMD ["--help"]
