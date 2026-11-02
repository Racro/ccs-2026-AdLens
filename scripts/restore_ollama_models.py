#!/usr/bin/env python3
"""Restore the exact Ollama model builds used for the paper results (see run.md digests).

Tags such as qwen3.5:9b are mutable: `ollama pull` today may give a different build.
This downloads the recorded layer blobs from registry.ollama.ai by sha256, verifies them,
and installs the recorded manifests from model_manifests/ into the Ollama store.

Usage: OLLAMA_MODELS=/path/to/store python scripts/restore_ollama_models.py [model ...]
       (default store: ~/.ollama/models; default models: all four)
Then run `ollama list` and compare the manifest sha256 below (also checked here).
"""
import hashlib, json, os, sys, urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent / "model_manifests"
EXPECTED = {  # sha256 of the manifest file (= "Digest" column in run.md)
    "qwen3.5/9b": "6488c96fa5faab64bb65cbd30d4289e20e6130ef535a93ef9a49f42eda893ea7",
    "gemma3/12b": "f4031aab637d1ffa37b42570452ae0e4fad0314754d17ded67322e4b95836f8a",
    "gemma4/26b": "08ae7ec1744bd7f451c4a530afb39d2673ad9d07a8369b8a33a3613b41212a68",
    "translategemma/4b": "c49d986b0764f5881c476eb21435bb62b7abc62347aab3d4a6071e811be510a1",
}
store = Path(os.environ.get("OLLAMA_MODELS", Path.home() / ".ollama" / "models"))


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 24), b""):
            h.update(chunk)
    return h.hexdigest()


def fetch(name, digest, dest):
    if dest.exists() and sha256_file(dest) == digest[7:]:
        return
    url = f"https://registry.ollama.ai/v2/library/{name}/blobs/{digest}"
    tmp = dest.with_suffix(".part")
    print(f"  downloading {digest[:19]} ...", flush=True)
    with urllib.request.urlopen(url) as r, open(tmp, "wb") as f:
        while chunk := r.read(1 << 24):
            f.write(chunk)
    if sha256_file(tmp) != digest[7:]:
        tmp.unlink()
        sys.exit(f"checksum mismatch for {digest}")
    tmp.rename(dest)


for key in (sys.argv[1:] or EXPECTED):
    name, tag = key.split("/") if "/" in key else (key.split(":")[0], key.split(":")[1])
    key = f"{name}/{tag}"
    src = ROOT / key
    if sha256_file(src) != EXPECTED[key]:
        sys.exit(f"{src} does not match the recorded digest")
    print(f"{name}:{tag}")
    (store / "blobs").mkdir(parents=True, exist_ok=True)
    m = json.loads(src.read_text())
    for layer in [m["config"]] + m["layers"]:
        fetch(name, layer["digest"], store / "blobs" / layer["digest"].replace(":", "-"))
    dst = store / "manifests" / "registry.ollama.ai" / "library" / name / tag
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_bytes(src.read_bytes())
    print(f"  installed, manifest sha256 {EXPECTED[key]}")
