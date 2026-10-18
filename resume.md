# Resume notes — AEC review fixes (GPU machine)

Read this first when picking the work up on the GPU box. Branch: `artifact-eval-fixes`
(`git fetch && git checkout artifact-eval-fixes`). Delete this file before the final release.

## Context

CCS 2026 artifact-evaluation reviewer: *Available* and *Evaluated (Functional)* badges recommended,
*Results Reproduced* **declined**. The code/doc fixes are already on this branch. What remains needs a GPU
(or the paper), and is listed below in order.

| Reviewer issue | State |
|---|---|
| `ollama` not in requirements / images | fixed (`requirements.txt`) |
| `gpu_v1` image: `undefined symbol: ncclCommResume` | Dockerfile changed (torch from CUDA-matched index + build-time import check) — **untested, needs rebuild** |
| only lower-bound pins | bounded ranges; **`requirements.lock` still to generate** |
| embeddinggemma needs HF token | fixed (`HF_TOKEN` / `--hf-token`, clear error, README) |
| `sample_data/metadata.json` has no labels | fixed: `adlens_pipeline.py --eval` uses `datasets/golden_main_data/metadata.json` |
| Table 2 only from hard-coded counts | fixed: `scripts/eval_pipeline_results.py` scores pipeline output |
| Table 2(c) DC/Scareware swapped in `scripts/f1_score_table2.py` | **open** — needs the paper's Table 2(c) |
| golden test images were `/scratch` symlinks | fixed: 301 real images + `datasets/golden_test_data/metadata.json` |
| Zenodo outdated | later, after the evaluation settles |

## 0. Environment

```bash
ollama serve &                     # if not already running
ollama pull qwen3.5:9b
ollama pull gemma3:12b
ollama pull gemma4:26b
ollama pull translategemma:4b      # only if re-running translation

# torch first, from the index matching the driver (CUDA 12.x shown)
pip install "torch>=2.4,<2.6" --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
nvidia-smi && ollama list          # sanity check
```

## 1. Table 2(a)/(b) — golden set (1,283 labeled ads)

```bash
python detection/adlens_pipeline.py --eval --judge-model gemma4:26b
python scripts/eval_pipeline_results.py --json-out detection/results/table2_ab.json
```

Results land in `detection/results/pipeline_results/` (cached and resumable — rerun the same command after an interruption).
Smoke test first with `--limit 20`. Expected (paper):

| | DC P/R/F1 | SW P/R/F1 | MD P/R/F1 | macro F1 |
|---|---|---|---|---|
| 2(a) Qwen3.5:9b | .973/.787/.870 | .989/.949/.969 | .859/.970/.911 | .917 |
| 2(a) Gemma3:12b | .873/.822/.847 | .994/.803/.888 | .966/.586/.729 | .821 |
| 2(b) Gemma4:26b judge | .897/.929/.912 | .967/.935/.951 | .861/.946/.902 | .922 |

(The stored runs in `../paper_tables/golden_set/` already reproduce 2(b) through the new script, so a
mismatch means the live models/prompts drifted, not the scorer.)

## 2. Table 2(c) — look-forward test set (301 ads)

```bash
python detection/adlens_pipeline.py \
    --metadata datasets/golden_test_data/metadata.json \
    --images-dir datasets/golden_test_data \
    --out-dir detection/results/test_set_results
python scripts/eval_pipeline_results.py \
    --metadata datasets/golden_test_data/metadata.json \
    --results-dir detection/results/test_set_results
```

Caveats:
- Test-set OCR was **regenerated** (PaddleOCR on CPU) — no stored OCR existed. It may differ slightly from the paper's run.
- The paper sampled the classifiers (seed 42, ≤50 ads/folder); the command above scores every ad, so the classifier rows
  won't match exactly. The full-pipeline row should be close to P 0.993 / R 1.000 / F1 0.997.
- 2 of 301 records have no `translated_ocr_text` (one of them has empty OCR). Check how the pipeline handles them.

## 3. Close the Table 2(c) swap (needs the paper)

Compare the per-class DC/Scareware numbers from step 2 with the paper's Table 2(c), then fix
`scripts/f1_score_table2.py` accordingly (labels swapped vs. counts wrong). Known oddity: the judge row uses
51 pos / 62 neg for Deceptive Claims, but the folders hold 52 / 64.

## 4. Docker

```bash
docker build --build-arg DEVICE=gpu -t adlens:gpu .
docker run --rm --gpus all adlens:gpu --help
docker run --rm --gpus all --entrypoint python adlens:gpu -c "import torch; print(torch.cuda.is_available())"
```

The build now fails if `import torch, ollama, sentence_transformers, paddle, paddleocr` breaks. If the nccl
error persists, the cu121 torch and `paddlepaddle-gpu==3.1.0` (cu123) wheels are still pulling conflicting
NVIDIA libs — pin `nvidia-nccl-cu12` or run Ollama in its own GPU container and use the CPU image for the rest.
After it works, push `racro/adlens:gpu_v1`.

## 5. Finish

- `pip freeze > requirements.lock` from the working GPU env and commit it.
- Commit the numbers or logs worth keeping (not `detection/results/`, it is git-ignored).
- Merge `artifact-eval-fixes` into `master`, publish a new Zenodo version, tell the reviewer.
- Stray local files, deliberately **not** committed: `crawler/flow_check.mjs`, `home_review.mjs`, `review2.mjs`,
  `crawler/browser_profile/`, and the `--force-auth` change in `crawler/run.py` (check whether to keep it).
