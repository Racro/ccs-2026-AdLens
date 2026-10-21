# TODO — what is left on `artifact-eval-fixes` (do not merge yet)

Status: code, data and docs fixes for the AEC review are on this branch. Evaluation numbers and the Docker work are in
[`run.md`](run.md). Everything below needs a machine with docker and/or a few GPU hours. Delete `todo.md` and (optionally) `run.md`
before the final release.

## 1. Docker (needs docker, GPU host for the gpu image)

The Dockerfile itself is unchanged. Its Python layers were replayed without docker (apptainer sandbox of `python:3.12`, L40S): the
`ncclCommResume` error did not reproduce, and the import check only failed because a fresh install pulled `transformers` 5.x
(needs torch>=2.6); `requirements.txt` now has `transformers>=4.57,<5`. The apt/Chrome/Node/Ollama layers and the CPU image are untested.

**Done 2026-10-21 (CPU image):** `docker build --build-arg DEVICE=cpu -t adlens:cpu .` succeeds from the current tree; `--help` runs and the
import check passes (torch 2.5.1+cpu, transformers 4.57.6). `.dockerignore` now excludes `crawler/browser_profile/` (unreadable files broke the
build context). GPU image check was skipped on this host (GTX 970, no nvidia-container-toolkit); CPU working is considered enough. Still open: the push.

**Done 2026-10-21 (GPU image):** `docker build --build-arg DEVICE=gpu -t adlens:gpu_v1 .` succeeds on Tesla V100-SXM2-16GB (CUDA 12.4 driver,
cu121 torch wheels). Import check prints `torch 2.5.1+cu121`; `docker run --gpus all adlens:gpu_v1 --help` passes; `torch.cuda.is_available()`
returns `True`. Dockerfile fix: the build-time import check now skips `paddle`/`paddleocr` for GPU builds (they require `libcuda.so.1` at import
time, which is only available at runtime). Pushed to Docker Hub as `racro/adlens:gpu_v1` (sha256:d3a4e737...).

```bash
docker build --build-arg DEVICE=cpu -t adlens:cpu .
docker build --build-arg DEVICE=gpu -t adlens:gpu .       # build fails if `import torch, ollama, sentence_transformers, paddle, paddleocr` breaks
docker run --rm --gpus all adlens:gpu --help
docker run --rm --gpus all --entrypoint python adlens:gpu -c "import torch; print(torch.cuda.is_available())"   # expect True
```

- If `undefined symbol: ncclCommResume` still appears: pin `nvidia-nccl-cu12` to the version torch declares (2.21.5 for torch 2.5.1), or run Ollama in
  its own GPU container and use the CPU image for the rest.
- The container needs an Ollama server >= 0.40 (0.21.0 cannot load gemma4): `OLLAMA_URL` defaults to `http://host.docker.internal:11434`.
- When the images are verified: push `racro/adlens:gpu_v1` (and `:cpu_v1` / `:latest` if rebuilt), then check the Docker section of the README still matches.

## 2. Reproduce from a clean checkout on the other system

```bash
git fetch && git checkout artifact-eval-fixes
ollama pull qwen3.5:9b && ollama pull gemma3:12b && ollama pull gemma4:26b      # + translategemma:4b only to re-run translation
pip install "torch>=2.4,<2.6" --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
python detection/adlens_pipeline.py --eval --judge-model gemma4:26b                # Table 2(a)/(b), 1,283 ads, a few hours
python scripts/eval_pipeline_results.py
python detection/adlens_pipeline.py --metadata datasets/golden_test_data/metadata.json \
    --images-dir datasets/golden_test_data/images --out-dir detection/results/test_set_results     # Table 2(c), 253 ads
python scripts/eval_pipeline_results.py --metadata datasets/golden_test_data/metadata.json --results-dir detection/results/test_set_results
```

Expected (see `run.md` for the full tables): 2(a)/(b) within a few points of the paper (judge macro F1 .894 with thinking off, .923 with it on, vs .922); 2(c) full pipeline macro F1 **.969**
with Gemma4 thinking off (erratum/paper .997; .882 with thinking on).

## 3. Open evaluation items

- **Golden set with thinking off:** done 2026-10-21 (`run.md`): judge macro F1 .894 (paper .922; .923 with thinking on), full pipeline .946.
### Results vs erratum (P / R / F1; Gemma4 judge)

| Table | Run | Deceptive Claims | Scareware | Misleading Design | Macro |
|---|---|---|---|---|---|
| 2(b) judge, golden set | erratum | .897 / .929 / .912 | .967 / .935 / .951 | .861 / .946 / .902 | F1 .922 |
| | thinking on | .846 / 1.000 / .917 | .955 / .955 / .955 | .880 / .917 / .898 | F1 .923 |
| | thinking off | .750 / .955 / .840 | .955 / .955 / .955 | .870 / .906 / .888 | F1 .894 |
| 2(c) full pipeline, test set | erratum | 1.000 / 1.000 / 1.000 | .980 / 1.000 / .990 | 1.000 / 1.000 / 1.000 | P .993 / R 1.000 / F1 .997 |
| | thinking on | .979 / .904 / .940 | 1.000 / .917 / .957 | .600 / 1.000 / .750 | P .860 / R .940 / F1 .882 |
| | thinking off | 1.000 / .904 / .949 | 1.000 / .917 / .957 | 1.000 / 1.000 / 1.000 | P 1.000 / R .940 / F1 .969 |

2(a) (classifiers, no judge) macro F1, mine vs erratum: Qwen3.5:9b .929 vs .917, Gemma3:12b .847 vs .821 (unaffected by thinking).

- **2(c) gap to .997:** the remainder is classifier recall (5 Deceptive Claims and 4 Scareware positives missed or mis-judged, 8 of them never reach the judge
  because both classifiers said SAFE). The paper's exact 2(c) sampling code is not in the repo; the shipped 253-ad set is a seed-42 recreation (`run.md`).
- **Table 2(c) erratum:** the paper PDF's DC/Scareware columns (and the Qwen/Gemma3 rows) are transposed relative to `scripts/f1_score_table2.py`; the
  README/`scripts/README.md` erratum is restored. Fix the camera-ready table. Note the erratum's numbers could not be reproduced exactly from the shipped data.
- `scripts/f1_score_table2.py` (c) still carries the erratum's hard-coded counts; they do not equal the shipped 253-ad set's class sizes
  (DC 52/50, SW 48/50, MD 3/50). Decide whether to regenerate it from `scripts/eval_pipeline_results.py` output.

## 4. Merge and release (after 1 and 2 pass)

- Merge `artifact-eval-fixes` into `master` (fast-forward; `master` has nothing the branch lacks as of the last push).
- Publish a new Zenodo version (new flat `datasets/*/images`, 253-ad test set) and tell the reviewer.
- Delete `todo.md` (and `run.md` if you do not want it released).

## 5. Local files deliberately NOT committed (not part of the review fixes)

- `crawler/creative.js`: opt-in `SCREENSHOT_FULL_PAGE=1` extra screenshot for a live-run UI.
- `run_all.py`: refactor that batches classifier and judge calls per model (fewer Ollama model swaps).
- `.ipynb_checkpoints/`, `Untitled*.ipynb`: scratch notebooks.
- Decide whether to commit or discard each (the old notes also asked about a `--force-auth` change in `crawler/run.py`; it is not in the working tree now).
