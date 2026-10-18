#!/bin/sh
# Resolves the Chrome binary at container start instead of trusting a fixed
# path baked in at build time — matches whatever the base image/packaging
# actually produced, and BROWSER_EXECUTABLE_PATH still overrides it if set.
set -e

if [ -z "$BROWSER_EXECUTABLE_PATH" ]; then
    for candidate in google-chrome google-chrome-stable chromium chromium-browser; do
        path=$(command -v "$candidate" 2>/dev/null) && { export BROWSER_EXECUTABLE_PATH="$path"; break; }
    done
fi

exec python3 run_all.py "$@"
