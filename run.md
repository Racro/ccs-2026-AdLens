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
- Runtime: golden set (1,283 ads) ≈ 5.5 h end to end; test set (301 images, scored on the 253-ad sample) ≈ 1 h. 0 `judge=ERROR` in either run.
- `scripts/eval_pipeline_results.py` fixes made during this run (uncommitted):
  1. **Result-file names:** it looked for `misleading_results.json` / `ad_design_results.json`, but the pipeline writes
     `deceptive_claim_results.json` / `misleading_design_results.json`, so tables (b) and (c) silently showed Scareware only.
     It now accepts both names.
  2. (An earlier creative-ID dedupe of the test set was dropped: versions of a creative are different images, so it removed ads that are not duplicates.)
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

## Table 2(c) — look-forward test set (253 ads)

The test set is the balanced sample of the 301-file look-forward crawl: up to 50 negatives per violation type, 50 positives for
scareware / deceptive claims where available, all misleading-design positives. Available positives: scareware 48, deceptive claims 52,
misleading design 3, so the 253 ads are:

| | positives | negatives |
|---|---|---|
| Deceptive Claims (`misleading`) | 52 | 50 |
| Scareware | 48 | 50 |
| Misleading Design (`ad_design`) | 3 | 50 |

(Drawn with `random.Random(42)` over `ad_id`-sorted groups; the authors' original sampling code is not in the repo, so individual ads may differ
from the paper's draw. Classification and judging were run on all 301 files; scoring uses the 253.)

Live full pipeline (ensemble + judge on disagreements), P / R / F1 and counts tp/fp/fn/tn:

| Gemma4 judge | Deceptive Claims | Scareware | Misleading Design | Macro P / R / F1 |
|---|---|---|---|---|
| thinking on | .979 / .904 / .940 (47/1/5/49) | 1.000 / .917 / .957 (44/0/4/50) | .600 / 1.000 / .750 (3/2/0/48) | .860 / .940 / **.882** |
| **thinking off** | 1.000 / .904 / .949 (47/0/5/50) | 1.000 / .917 / .957 (44/0/4/50) | 1.000 / 1.000 / 1.000 (3/0/0/50) | 1.000 / .940 / **.969** |
| erratum / script | 1.000 / 1.000 / 1.000 | .980 / 1.000 / .990 | 1.000 / 1.000 / 1.000 | .993 / 1.000 / **.997** |

Classifiers (identical in both runs): Qwen3.5:9b DC 1.000/.808/.894 (42/0/10/50), SW 1.000/.875/.933 (42/0/6/50), MD .500/1.000/.667 (3/3/0/47);
Gemma3:12b DC .938/.865/.900 (45/3/7/47), SW 1.000/.875/.933 (42/0/6/50), MD 0/0/0 (0/0/3/50); majority-vote ensemble DC 1.000/.750/.857,
SW 1.000/.833/.909, MD 0/0/0. Judge only, thinking off: DC 1.000/.889/.941 (n=12), SW 1.000/1.000/1.000 (n=4), MD 1.000/1.000/1.000 (n=6).

Thinking: `detection/adlens_pipeline.py` now disables thinking for `gemma4` as well as `qwen3` (`_needs_think_disabled`). The thinking-off run reran only
the judge stage (`detection/results/test_set_results_nothink/`; a second run in `..._nothink2/` gave identical judge labels, temperature 0, 0 `judge=ERROR`).
With thinking on the judge produced 2 false positives on Misleading Design; with it off the macro F1 rises from .882 to **.969**.
The remaining gap to .997 is 5 DC and 4 SW positives where *both* classifiers said SAFE, which the judge never sees.
The golden-set 2(a)/(b) numbers above were produced with thinking **on** and were not rerun.

## Table 2(c) column order (DC vs Scareware)

- The script's class sizes match the data: Deceptive Claims has 52 positives (50 in the script's seed-42 version) and Scareware 48.
- The paper PDF's Table 2(c) has recall steps of 1/48 in the **Deceptive Claims** column and 1/50 in the **Scareware** column, i.e. the
  opposite sizes, and its Qwen and Gemma3 rows correspond to the script's rows exchanged. This matches the erratum in commit `11da0cb`
  (columns transposed); the erratum numbers in `scripts/f1_score_table2.py` are used as the reference.
- The erratum numbers could not be reproduced from the shipped data; see the table above.

## Open

- Thinking off brings live 2(c) to macro F1 .969 (erratum .997); the rest is classifier recall (both-SAFE misses). The golden set has not been rerun with thinking off.
- `requirements.lock`, Docker build (nccl fix, push `racro/adlens:gpu_v1`), merge to `master`, Zenodo — see `resume.md` steps 4–5.
- Uncommitted changes: `scripts/eval_pipeline_results.py` (this run), plus pre-existing `crawler/creative.js`, `run_all.py`. READMEs untouched.
