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
- Models (`ollama show` / `ollama list` on 0.40.6; digest = full sha256 of the model manifest, same for every run in this file):

  | Model | Role | Architecture / params / quant | Digest |
  |---|---|---|---|
  | `qwen3.5:9b` | classifier | qwen35, 9.7B, Q4_K_M | `6488c96fa5faab64bb65cbd30d4289e20e6130ef535a93ef9a49f42eda893ea7` |
  | `gemma3:12b` | classifier | gemma3, 12.2B, Q4_K_M | `f4031aab637d1ffa37b42570452ae0e4fad0314754d17ded67322e4b95836f8a` |
  | `gemma4:26b` | judge | gemma4, 25.2B, Q4_K_M | `08ae7ec1744bd7f451c4a530afb39d2673ad9d07a8369b8a33a3613b41212a68` |
  | `translategemma:4b` | translation | gemma3, 4.3B, Q4_K_M | `c49d986b0764f5881c476eb21435bb62b7abc62347aab3d4a6071e811be510a1` |
- Thinking: the golden-set run and the first test-set run had **Gemma4 thinking on** (Ollama default; the pipeline only disabled it for `qwen3`). The pipeline now disables it for `gemma4` too; see the thinking-off rerun below.
- Judge latency: 30–60 s per call (reasoning trace), ~220 golden-set cases ≈ 3 h. The paper reports ~1–2 s. Not changed, to keep the run faithful.
- Runtime: golden set (1,283 ads) ≈ 5.5 h end to end; test set (301 images, scored on the 253-ad sample) ≈ 1 h. 0 `judge=ERROR` in either run.
- `scripts/eval_pipeline_results.py` fixes made during this run (uncommitted):
  1. **Result-file names:** it looked for `misleading_results.json` / `ad_design_results.json`, but the pipeline writes
     `deceptive_claim_results.json` / `misleading_design_results.json`, so tables (b) and (c) silently showed Scareware only.
     It now accepts both names.
  2. (An earlier creative-ID dedupe of the test set was dropped: versions of a creative are different images, so it removed ads that are not duplicates.)
- Docker (GPU) fix, tested without docker: no docker on this node, so the Dockerfile's Python layers (`DEVICE=gpu`: torch cu121, `requirements.txt`, `ocr/requirements.txt`,
  `paddlepaddle-gpu==3.1.0` cu123, import check) were replayed in an apptainer sandbox of `python:3.12` on the L40S. Findings: (1) the `ncclCommResume` error did not
  reproduce, torch 2.5.1+cu121 sees the GPU and reports NCCL 2.21.5, Paddle passes `run_check` on the GPU, pip only warns about nvidia-* version mismatches;
  (2) the build **still failed the import check** because a fresh install pulls `transformers` 5.x, which refuses torch<2.6 — fixed by `transformers>=4.57,<5` in `requirements.txt`;
  with it, `import torch, ollama, sentence_transformers, paddle, paddleocr` passes. Not covered: the apt/Chrome/Node/Ollama layers and the CPU image; `racro/adlens:gpu_v1` has not been rebuilt or pushed.

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
| live, thinking off (n = 44 / 24 / 128) | .750 / .955 / .840 | .955 / .955 / .955 | .870 / .906 / .888 | **.894** |
| paper | .897 / .929 / .912 | .967 / .935 / .951 | .861 / .946 / .902 | .922 |

The first live row is Gemma4 thinking **on**, the second thinking **off** (the pipeline default now). Full pipeline on the golden set (ensemble + judge): DC .970/.871/.918, SW .990/.985/.987, MD .927/.962/.945, macro F1 **.950** (thinking off: DC .956/.867/.909, SW .990/.985/.987, MD .923/.958/.940, macro F1 **.946**).
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
| thinking on, fresh rerun (2026-10-21, classifiers rerun, no cache) | 1.000 / .885 / .939 | 1.000 / .917 / .957 | 1.000 / 1.000 / 1.000 | 1.000 / .934 / **.965** |
| **thinking off** | 1.000 / .904 / .949 (47/0/5/50) | 1.000 / .917 / .957 (44/0/4/50) | 1.000 / 1.000 / 1.000 (3/0/0/50) | 1.000 / .940 / **.969** |
| erratum / script | 1.000 / 1.000 / 1.000 | .980 / 1.000 / .990 | 1.000 / 1.000 / 1.000 | .993 / 1.000 / **.997** |

Classifiers (identical in both runs): Qwen3.5:9b DC 1.000/.808/.894 (42/0/10/50), SW 1.000/.875/.933 (42/0/6/50), MD .500/1.000/.667 (3/3/0/47);
Gemma3:12b DC .938/.865/.900 (45/3/7/47), SW 1.000/.875/.933 (42/0/6/50), MD 0/0/0 (0/0/3/50); majority-vote ensemble DC 1.000/.750/.857,
SW 1.000/.833/.909, MD 0/0/0. Judge only, thinking off: DC 1.000/.889/.941 (n=12), SW 1.000/1.000/1.000 (n=4), MD 1.000/1.000/1.000 (n=6).

Thinking: `detection/adlens_pipeline.py` now disables thinking for `gemma4` as well as `qwen3` (`_needs_think_disabled`). The thinking-off run reran only
the judge stage (`detection/results/test_set_results_nothink/`; a second run in `..._nothink2/` gave identical judge labels, temperature 0, 0 `judge=ERROR`).
With thinking on the judge produced 2 false positives on Misleading Design in the first run, but not in a fresh rerun with all classifiers rerun (macro F1 .965, `detection/results/test_set_results_think2`), so the thinking-on/off difference on this set is within run-to-run variation; with it off the macro F1 rises from .882 to **.969**.
The remaining gap to .997 is 5 DC and 4 SW positives where *both* classifiers said SAFE, which the judge never sees.
The golden-set 2(a)/(b) numbers above were produced with thinking **on**. The judge stage was rerun with thinking off (2026-10-21, ~20 min, same classifier cache, 0 judge errors); 2(a) is unaffected and 2(b) is in the extra row. Results: `detection/results/pipeline_results_nothink`. Thinking off lowers DC judge precision (more false positives), so the 2(b) macro F1 is .894 vs the paper's .922. A second thinking-off run with the classifier cache also cleared (`pipeline_results_nothink2`, Qwen/Gemma3 reclassified from scratch, 0 judge errors) reproduced it: judge macro F1 .894 (DC .750/.955/.840, SW .955/.955/.955, MD .869/.905/.887), full pipeline .946; only one Misleading Design disagreement differed (127 vs 128).

## Table 2(c) column order (DC vs Scareware)

- The script's class sizes match the data: Deceptive Claims has 52 positives (50 in the script's seed-42 version) and Scareware 48.
- The paper PDF's Table 2(c) has recall steps of 1/48 in the **Deceptive Claims** column and 1/50 in the **Scareware** column, i.e. the
  opposite sizes, and its Qwen and Gemma3 rows correspond to the script's rows exchanged. This matches the erratum in commit `11da0cb`
  (columns transposed); the erratum numbers in `scripts/f1_score_table2.py` are used as the reference.
- The erratum numbers could not be reproduced from the shipped data; see the table above.

## The 253 ads (look-forward test set)

Shipped in `datasets/golden_test_data/` (`images/<ad_id>.png` + `metadata.json`; 251 PNGs because 2 ads carry two violation-type labels and share one image).
The golden main set was flattened the same way: `datasets/golden_main_data/images/` (1,271 PNGs for the 1,283 labeled records; 12 ads carry two labels and share one
image). `adlens_pipeline.py --eval` now reads its images from there. Ad IDs per class:

<details><summary>Deceptive Claims — positives (52)</summary>

`CR00332366868834680833-v0`, `CR01038988606498144257-v0`, `CR01525350746046857217-v0`, `CR01667567996664545281-v0`, `CR01927279403423760385-v0`, `CR02032272816732110849-v0`, `CR02270743033232228353-v1`, `CR02323378415790456833-v2`, `CR02492581779152568321-v0`, `CR02967867902419533825-v2`, `CR03052136818624430081-v0`, `CR03741991708338946049-v0`, `CR03741991708338946049-v2`, `CR04321080989217456129-v0`, `CR04383411195294515201-v0`, `CR04671037398396698625-v0`, `CR04901681643317624833-v0`, `CR05838375553438056449-v0`, `CR07126578262897065985-v0`, `CR07267924287988695041-v0`, `CR07399076371412025345-v2`, `CR07520108317885071361-v0`, `CR07964177381894127617-v0`, `CR07964177381894127617-v1`, `CR08914599914458054657-v0`, `CR09519464616929787905-v1`, `CR11205788898984722433-v0`, `CR11317044494869725185-v0`, `CR11387399160144592897-v0`, `CR11813071161237962753-v2`, `CR11848072507590443009-v0`, `CR12003101104486219777-v0`, `CR12072516881320247297-v0`, `CR12468590647869177857-v0`, `CR12551950671684829185-v0`, `CR12606714743114170369-v0`, `CR13140008342493593601-v0`, `CR13348562788927143937-v1`, `CR13370315466841522177-v0`, `CR13387018972612788225-v0`, `CR13965385925891457025-v0`, `CR14647740067540369409-v0`, `CR14899796536908054529-v0`, `CR15739815698177720321-v0`, `CR15917618170181976065-v0`, `CR15974644546915008513-v0`, `CR16710952359794049025-v0`, `CR16735882007837409281-v0`, `CR17219349494047440897-v0`, `CR17431513465356288001-v0`, `CR17673672174000406529-v0`, `CR17696496076887949313-v0`

</details>

<details><summary>Deceptive Claims — negatives (50)</summary>

`CR00279030221196230657-v1`, `CR00334282853745426433-v2`, `CR00832592844987301889-v0`, `CR01019122630407487489-v0`, `CR01291818006770352129-v3`, `CR01426548313347325953-v0`, `CR01619755458652798977-v0`, `CR01801189419995103233-v0`, `CR02423888510307008513-v0`, `CR02659202613039333377-v1`, `CR03027817992239972353-v2`, `CR03108421763743088641-v0`, `CR03206137157399871489-v2`, `CR03484816514127757313-v0`, `CR04545818382796062721-v0`, `CR04575827628530860033-v0`, `CR05109499399780696065-v1`, `CR05133691266931884033-v0`, `CR05241712348716072961-v3`, `CR05616571390594383873-v0`, `CR05632160635820179457-v0`, `CR05683186608233250817-v1`, `CR06027398782121934849-v4`, `CR06931870241270005761-v2`, `CR07542502174287200257-v2`, `CR08275498189423902721-v0`, `CR08346579829453225985-v0`, `CR08425121664466944001-v0`, `CR08816698041911214081-v0`, `CR08819482091252088833-v0`, `CR08820980124305326081-v1`, `CR09436315192540528641-v0`, `CR10248127053014499329-v0`, `CR11158070506656104449-v2`, `CR11170168020779663361-v0`, `CR11282792740569481217-v0`, `CR11453263677629661185-v0`, `CR11693189621755150337-v0`, `CR12275900453417385985-v2`, `CR13878141920200359937-v2`, `CR13966053466888470529-v0`, `CR13966053466888470529-v2`, `CR14312831608858607617-v0`, `CR14491485601078968321-v0`, `CR15275240665001230337-v0`, `CR15491233092848320513-v0`, `CR15953654732501811201-v0`, `CR16217424883355746305-v0`, `CR16437071709896441857-v1`, `CR18404572013168427009-v0`

</details>

<details><summary>Scareware — positives (48)</summary>

`CR00897663841494630401-v0`, `CR00897663841494630401-v1`, `CR00897663841494630401-v3`, `CR01843996981077213185-v0`, `CR01843996981077213185-v1`, `CR01843996981077213185-v2`, `CR02238238583297146881-v1`, `CR02869058778160955393-v0`, `CR03486423424601948161-v0`, `CR04071784723532218369-v2`, `CR04635504884848787457-v0`, `CR04635504884848787457-v1`, `CR04721457144765874177-v0`, `CR04721457144765874177-v1`, `CR04963925022335827969-v0`, `CR05578815492736417793-v1`, `CR05578815492736417793-v2`, `CR06159705464423579649-v0`, `CR06159705464423579649-v1`, `CR06159705464423579649-v2`, `CR06398864799836405761-v1`, `CR06398864799836405761-v2`, `CR06525873073042751489-v0`, `CR06525873073042751489-v1`, `CR06525873073042751489-v2`, `CR08489347566029242369-v0`, `CR08489347566029242369-v1`, `CR08489347566029242369-v2`, `CR11755402162908168193-v0`, `CR11787091050093346817-v1`, `CR12199849252758749185-v3`, `CR12570653364473298945-v0`, `CR12570653364473298945-v1`, `CR13472424010746691585-v0`, `CR13472424010746691585-v2`, `CR14289497901942964225-v1`, `CR14289497901942964225-v2`, `CR14332178331993112577-v0`, `CR14418263744486309889-v0`, `CR14418263744486309889-v1`, `CR14418263744486309889-v2`, `CR14501217258237329409-v0`, `CR14501217258237329409-v1`, `CR14501217258237329409-v2`, `CR16467732278831742977-v0`, `CR16528588297513992193-v1`, `CR16963948198638911489-v0`, `CR17117363889286152193-v0`

</details>

<details><summary>Scareware — negatives (50)</summary>

`CR00395198890300669953-v0`, `CR00517698418508824577-v0`, `CR00693627030641573889-v0`, `CR01040819843114205185-v0`, `CR01373345694458314753-v0`, `CR01405539394919596033-v1`, `CR01568953047877419009-v0`, `CR01832722451407044609-v0`, `CR02078134511878537217-v1`, `CR02423888510307008513-v0`, `CR02781162349757726721-v0`, `CR02930197474410758145-v0`, `CR03073084473638649857-v1`, `CR04302143344538025985-v0`, `CR04484536896063537153-v0`, `CR04595658583458185217-v0`, `CR04597770882734096385-v0`, `CR04901681643317624833-v2`, `CR05067320561781178369-v0`, `CR05534763009613758465-v1`, `CR06455066061812269057-v2`, `CR09066536195030253569-v0`, `CR09228793324964413441-v2`, `CR09393369324921552897-v1`, `CR09732813329197432833-v0`, `CR10002306738721652737-v0`, `CR10158006982603177985-v0`, `CR10612905153814069249-v0`, `CR11071530832651878401-v0`, `CR11211508790400647169-v2`, `CR11366670677311488001-v0`, `CR12190196494149615617-v2`, `CR13112406357208727553-v0`, `CR13195338834209931265-v0`, `CR13207100172653297665-v0`, `CR13348562788927143937-v2`, `CR13361524381751181313-v0`, `CR13535650739707183105-v2`, `CR13665573844008566785-v0`, `CR14120398605638434817-v0`, `CR14195963641768968193-v0`, `CR14501987534852063233-v0`, `CR14746064607818809345-v1`, `CR15633915299445604353-v0`, `CR16129350265349341185-v0`, `CR16532029133253771265-v0`, `CR16721673671906689025-v1`, `CR16833173032955543553-v2`, `CR18174135902362664961-v0`, `CR18404572013168427009-v1`

</details>

<details><summary>Misleading Design — positives (3)</summary>

`CR11211508790400647169-v2`, `CR17625543723964170241-v1`, `CR17625543723964170241-v2`

</details>

<details><summary>Misleading Design — negatives (50)</summary>

`CR00100190231122149377-v0`, `CR00334282853745426433-v0`, `CR00820104677818368001-v0`, `CR02141030871954620417-v0`, `CR02141030871954620417-v1`, `CR02141030871954620417-v2`, `CR03520133600706035713-v0`, `CR04456893069207273473-v1`, `CR04498243064418009089-v0`, `CR05132679140708712449-v0`, `CR05132679140708712449-v1`, `CR05132679140708712449-v2`, `CR05491982360796725249-v0`, `CR05713702179072638977-v0`, `CR05788975483844558849-v1`, `CR06252815819915395073-v0`, `CR06741649644423479297-v0`, `CR06970769837379289089-v1`, `CR08083596877385170945-v0`, `CR08539767157779595265-v0`, `CR08674755574711189505-v0`, `CR09060943529135570945-v0`, `CR09968427761812897793-v0`, `CR09968427761812897793-v2`, `CR10600097226329948161-v0`, `CR10600097226329948161-v1`, `CR10600097226329948161-v2`, `CR10988690216411201537-v0`, `CR12676211978297933825-v0`, `CR12676211978297933825-v1`, `CR12899057945588793345-v0`, `CR12899057945588793345-v1`, `CR12899057945588793345-v2`, `CR13641414524119547905-v0`, `CR13876906412728123393-v0`, `CR14030772920322621441-v0`, `CR14249066222279720961-v0`, `CR14277755779183280129-v0`, `CR15125170968625938433-v0`, `CR15125170968625938433-v1`, `CR15274848998343573505-v0`, `CR15559257472077135873-v0`, `CR15575352123284520961-v0`, `CR15575352123284520961-v1`, `CR15575352123284520961-v2`, `CR16724373247830786049-v0`, `CR16724373247830786049-v1`, `CR17816574909988470785-v0`, `CR17865719094020407297-v0`, `CR17908540398997864449-v0`

</details>

## Open

- Thinking off brings live 2(c) to macro F1 .969 (erratum .997); the rest is classifier recall (both-SAFE misses).
- Rebuild and push `racro/adlens:gpu_v1` (needs docker), merge to `master`, Zenodo.
- Uncommitted changes: `scripts/eval_pipeline_results.py` (this run), plus pre-existing `crawler/creative.js`, `run_all.py`. READMEs untouched.
