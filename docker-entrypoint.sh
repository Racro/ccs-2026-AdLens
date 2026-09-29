#!/bin/sh
# Entrypoint for the racro/adlens image (see Dockerfile / README "Docker" section).
#
# Reconstructed for this repo: the exact script baked into the published
# racro/adlens:cpu_v1 / :gpu_v1 images was never committed, so this is a
# best-effort recreation of its observable behavior (its args and env vars
# line up 1:1 with run_all.py's --device / --ollama-url flags), not a
# byte-identical recovery.
set -e

# If $OLLAMA_URL points at this container itself, no host Ollama is being
# used — start the bundled Ollama daemon and wait for it to come up.
case "$OLLAMA_URL" in
    http://localhost*|http://127.0.0.1*)
        ollama serve >/var/log/ollama.log 2>&1 &
        for i in $(seq 1 30); do
            curl -sf "$OLLAMA_URL" >/dev/null 2>&1 && break
            sleep 1
        done
        ;;
esac

# The crawler needs Xvfb inside a container (no local X display available).
export XVFB_SWITCH=1

exec python run_all.py --device "$DEVICE" --ollama-url "$OLLAMA_URL" "$@"
