# Live run — Table 2 reproduction (GPU box, 2026-10-18/19)

Branch `artifact-eval-fixes`. Everything below was produced by the pipeline in this checkout; raw outputs are in
`detection/results/` (git-ignored): `pipeline_results/` (golden set), `test_set_results/` (look-forward set),
`table2_ab.json`, `table2_c.json`.

## Environment and caveats

- GPU: one NVIDIA L40S (46 GB), **shared** with another process holding ~20 GB at ~95% utilisation for most of the run.
- Python/env: `detection.ext3` overlay on `cuda12.3.2-cudnn9.0.0-ubuntu-22.04.4.sif` (Python 3.12, torch 2.11+cu128,
  paddlepaddle-gpu 3.1.0, paddleocr 3.4.1, ollama-py 0.6.1). `import cv2` needed the container's own libGL ahead of the
  host's (`--nv` injects a libGL that needs a newer glibc) — a cluster quirk, not a repo bug.
- Ollama **0.40.6** (`/scratch/rr3953/ollama_new/bin/ollama`). The `.sif` ships 0.21.0, which cannot load `gemma4`
  (`unknown model architecture: 'gemma4'`, HTTP 500 on every judge call, which the pipeline cached as `judge=ERROR`).
  README/requirements should say a recent Ollama is required.
- Models: `qwen3.5:9b`, `gemma3:12b` (classifiers), `gemma4:26b` (judge), all default Q4 builds pulled 2026-09-29.
- Thinking: the golden-set run and the first test-set run had **Gemma4 thinking on** (Ollama default; the pipeline only disabled it for `qwen3`). The pipeline now disables it for `gemma4` too; see the thinking-off rerun below.
- Judge latency: 30–60 s per call (reasoning trace), ~220 golden-set cases ≈ 3 h. The paper reports ~1–2 s. Not changed, to keep the run faithful.
- Runtime: golden set (1,283 ads) ≈ 5.5 h end to end; test set (301 images) ≈ 1 h. 0 `judge=ERROR` in either run.
- `scripts/eval_pipeline_results.py` fixes made during this run (uncommitted):
  1. **Result-file names:** it looked for `misleading_results.json` / `ad_design_results.json`, but the pipeline writes
     `deceptive_claim_results.json` / `misleading_design_results.json`, so tables (b) and (c) silently showed Scareware only.
     It now accepts both names.
  2. **Test-set dedup:** when `--metadata` is in `datasets/golden_test_data/`, it keeps the first record per
     (`crid`, violation type) — 301 records → **253 ads**, as in the paper. The golden main set is *not* deduped
     (1,283 records but only 1,140 unique `crid`s; the paper's 1,283 and its per-class splits count by `ad_id`).
- `requirements.lock`: **not generated.** `pip freeze` of this env (`/tmp/claude-3639252/requirements.lock.candidate`, 259 lines) has
  torch 2.11+cu128 and both `nvidia-nccl-cu12` and `-cu13`, which contradicts `requirements.txt` (`torch>=2.4,<2.6`, cu121). A lock
  should come from an environment built from `requirements.txt` as the Docker image will build it.
- Docker (resume.md step 4): **not tested** — no docker on this node (apptainer only). Needs another machine.

## Table 2(a) — ensemble classifiers, golden set (1,283 ads)

P / R / F1. Paper in brackets.

| Model | Deceptive Claims | Scareware | Misleading Design | Macro F1 |
|---|---|---|---|---|
| Qwen3.5:9b | .973 / .800 / .878 [.973 / .787 / .870] | .995 / .975 / .985 [.989 / .949 / .969] | .862 / .996 / .924 [.859 / .970 / .911] | **.929** [.917] |
| Gemma3:12b | .900 / .844 / .872 [.873 / .822 / .847] | .983 / .894 / .937 [.994 / .803 / .888] | .959 / .594 / .734 [.966 / .586 / .729] | **.847** [.821] |
| Majority-vote ensemble | .989 / .773 / .868 | .994 / .879 / .933 | .959 / .594 / .734 | .845 |

Live counts (tp/fp/fn/tn): Qwen DC 180/5/45/219, SW 193/1/5/201, MD 238/38/1/157. Differences from the paper are 1–5 points on single
classes (largest: Gemma3 scareware recall .894 vs .803), consistent with model/prompt drift; no class changes ordering.

## Table 2(b) — Gemma4:26b judge on classifier disagreements, golden set

| | Deceptive Claims | Scareware | Misleading Design | Macro F1 |
|---|---|---|---|---|
| live (n = 44 / 24 / 128) | .846 / 1.000 / .917 | .955 / .955 / .955 | .880 / .917 / .898 | **.923** |
| paper | .897 / .929 / .912 | .967 / .935 / .951 | .861 / .946 / .902 | .922 |

Full pipeline on the golden set (ensemble + judge): DC .970/.871/.918, SW .990/.985/.987, MD .927/.962/.945, macro F1 **.950**.
Paper disagreement counts were 50 / 34 / 124 (scareware lower, DC lower in the live run).

## Table 2(c) — look-forward test set

Ground truth in `datasets/golden_test_data/metadata.json` (301 records, 253 unique creatives):

| | raw 301 (tp/tn) | deduped 253 (tp/tn) |
|---|---|---|
| Deceptive Claims (`misleading`) | 52 / 64 | 50 / 63 |
| Scareware | 48 / 67 | 27 / 66 |
| Misleading Design (`ad_design`) | 3 / 67 | 2 / 45 |

Live full pipeline (P / R / F1; counts tp/fp/fn/tn):

| | Deceptive Claims | Scareware | Misleading Design | Macro P / R / F1 |
|---|---|---|---|---|
| deduped 253 | .978 / .900 / .938 (45/1/5/62) | 1.000 / .889 / .941 (24/0/3/66) | .400 / 1.000 / .571 (2/3/0/42) | .793 / .930 / **.817** |
| raw 301 | .979 / .904 / .940 (47/1/5/63) | 1.000 / .917 / .957 (44/0/4/67) | .500 / 1.000 / .667 (3/3/0/64) | .826 / .940 / **.854** |
| paper (macro F1) | | | | **.997** |

Live classifiers on the deduped set: Qwen DC 1.000/.800/.889, SW 1.000/.815/.898, MD .333/1.000/.500; Gemma3 DC .917/.880/.898,
SW 1.000/.852/.920, MD 0/0/0. Judge-only: DC 7/1/1/3 (12 cases), SW 3/0/0/0 (3), MD 2/3/0/1 (6).

**With thinking on, the live run does not reproduce Table 2(c)** (macro F1 .817 deduped / .854 raw vs .997). The gaps: recall of ~.90 on DC and SW
(paper ~1.00) and, mainly, Misleading Design precision (3 false positives against 2–3 positives). Caveats: the paper sampled
(seed 42, ≤ 50 per folder) while this scores everything; test-set OCR was regenerated; Gemma4 thinking left on.

### Table 2(c) rerun with Gemma4 thinking OFF (test set, 253 deduped ads)

`detection/adlens_pipeline.py` now disables thinking for `gemma4` as well as `qwen3` (`_needs_think_disabled`). Classifier outputs are the
same cached ones; only the judge stage was rerun (`detection/results/test_set_results_nothink/`, repeated in `..._nothink2/`).

| Full pipeline (ensemble + judge on disagreements) | Deceptive Claims | Scareware | Misleading Design | Macro P / R / F1 |
|---|---|---|---|---|
| thinking on (above) | .978 / .900 / .938 | 1.000 / .889 / .941 | .400 / 1.000 / .571 | .793 / .930 / .817 |
| **thinking off** | 1.000 / .900 / .947 (45/0/5/63) | 1.000 / .889 / .941 (24/0/3/66) | 1.000 / 1.000 / 1.000 (2/0/0/45) | 1.000 / .930 / **.963** |
| erratum / script | 1.000 / 1.000 / 1.000 | .980 / 1.000 / .990 | 1.000 / 1.000 / 1.000 | .993 / 1.000 / **.997** |

Judge only (thinking off): DC 1.000/.875/.933 (n=12), SW 1.000/1.000/1.000 (n=3), MD 1.000/1.000/1.000 (n=6). 0 `judge=ERROR`.
The second thinking-off run produced **identical** judge labels and metrics (temperature 0, deterministic).
The remaining gap is 5 DC and 3 SW positives where *both* classifiers said SAFE; the judge never sees them. The golden-set
2(a)/(b) numbers above were produced with thinking **on** and were not rerun.

## Table 2(c) column order (DC vs Scareware)

- Script counts (`scripts/f1_score_table2.py`, DC tp+fn = 50, SW tp+fn = 48) agree with the data: the `misleading` folder
  (Deceptive Claims) has 52 positives (→ 50 sampled) and `scareware` has 48. The live run also finds 50 DC positives.
- The paper PDF's Table 2(c) has recall steps of 1/48 in the **Deceptive Claims** column and 1/50 in the **Scareware** column,
  i.e. the opposite sizes; its Qwen and Gemma3 rows also correspond to the script's rows exchanged. So the PDF's DC/SW columns
  (and the two classifier rows) are transposed relative to the data; the script's order is the one that matches the folders.
  This matches the erratum in commit `11da0cb`.
- Not resolvable from the data: the paper's 48 / 50 positives never occur after dedup (scareware has 27); the exact sampling that
  produced the paper's counts is unknown. The script's judge row (51 pos / 62 neg for DC) also differs from the folders (52 / 64) and
  from the deduped set (50 / 63) by one.

## Open

- Thinking off brings live 2(c) to macro F1 .963 (erratum .997); the rest is classifier recall (both-SAFE misses). A seed-42 ≤50-per-folder
  subsample has not been tried, and the golden set has not been rerun with thinking off.
- `requirements.lock`, Docker build (nccl fix, push `racro/adlens:gpu_v1`), merge to `master`, Zenodo — see `resume.md` steps 4–5.
- Uncommitted changes: `scripts/eval_pipeline_results.py` (this run), plus pre-existing `crawler/creative.js`, `run_all.py`. READMEs untouched.
