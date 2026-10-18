#!/bin/sh
# Starts a local Ollama server (if one isn't already running) and pulls the
# models AdLens' detection pipeline needs. Run this once before Step 3/4 of
# the pipeline (translate/detect), on the host or inside the Docker container.
#
# This targets Ollama's own bind address (localhost:11434, or $OLLAMA_HOST
# if set) — not the pipeline-level $OLLAMA_URL, which may point elsewhere
# (e.g. host.docker.internal) when the pipeline talks to a server started
# by this script on a different machine/container.
#
# Usage:
#   ./scripts/setup_ollama.sh
#   OLLAMA_HOST=localhost:11500 ./scripts/setup_ollama.sh
set -e

HOST_URL="http://${OLLAMA_HOST:-localhost:11434}"
MODELS="qwen3.5:9b gemma3:12b gemma4:26b translategemma:4b"

if ! command -v ollama >/dev/null 2>&1; then
    echo "ollama not found on PATH — install it first: https://ollama.com/download" >&2
    exit 1
fi

if ! curl -fsS "$HOST_URL" >/dev/null 2>&1; then
    echo "Starting ollama serve on $HOST_URL..."
    ollama serve >/tmp/ollama-serve.log 2>&1 &
    for i in $(seq 1 30); do
        curl -fsS "$HOST_URL" >/dev/null 2>&1 && break
        sleep 1
    done
    curl -fsS "$HOST_URL" >/dev/null 2>&1 || { echo "ollama serve did not come up — see /tmp/ollama-serve.log" >&2; exit 1; }
else
    echo "ollama server already reachable at $HOST_URL"
fi

for model in $MODELS; do
    echo "Pulling $model..."
    ollama pull "$model"
done

echo "Ready. Point --ollama-url / OLLAMA_URL at $HOST_URL (or wherever this ends up reachable from the pipeline)."
