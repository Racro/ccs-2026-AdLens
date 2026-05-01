# OCR Pipeline

A modular, high-performance pipeline that ingests ad images (via URL download or local disk), checks them against a fast binary cache, and extracts text/URLs using **PaddleOCR PP-OCRv5** (server models).

> **Legacy engine:** The original LightOnOCR-based pipeline is preserved in [`lightocr_legacy/`](lightocr_legacy/README.md).

## Why PaddleOCR

| | PaddleOCR (primary) | LightOnOCR (legacy) |
|---|---|---|
| Model | PP-OCRv5 server | `lightonai/LightOnOCR-2-1B` |
| Scene text accuracy | +13% over PP-OCRv4 | Baseline |
| Language support | 80+ via `--lang` | English only |
| GPU batching | Per-image (`rec_batch_num=16`) | Native transformer batching |

## Requirements

- Python 3.x
- CUDA-capable GPU (recommended)
- `paddlepaddle-gpu`, `paddleocr>=3.0.0`
- `Pillow`, `beautifulsoup4`, `requests`, `urlextract`

## Installation

### 1. Install PaddlePaddle (GPU)

PaddlePaddle must be installed separately before PaddleOCR, matching your CUDA version.

```bash
# CUDA 12.3 (default in requirements.txt)
pip install paddlepaddle-gpu==3.1.0 \
    --extra-index-url https://www.paddlepaddle.org.cn/packages/stable/cu123/
```

For other CUDA versions, replace `cu123` with your version (e.g. `cu118`, `cu121`). For CPU-only:
```bash
pip install paddlepaddle
```

### 2. Install remaining dependencies

```bash
pip install -r requirements.txt
```

> On first run PaddleOCR will automatically download the PP-OCRv5 server model weights (~200 MB) to `~/.paddleocr/`.

## Usage

Run from the project root (`AdLens/`).

### Local mode
```bash
python ocr/process.py --mode local \
    --input-file results/ads.json \
    --output-file results/results_with_ocr.json
```

### Download mode
```bash
python ocr/process.py --mode download \
    --input-file dataset.json \
    --download-workers 5
```

### Quick test (specific files or directory)
```bash
python ocr/process.py --mode local --images ./test1.png ./test2.jpg
python ocr/process.py --mode local --images-dir ./my_scans
```

## Command-line Arguments

| Argument | Default | Description |
|---|---|---|
| `--mode` | `download` | `download` (URL threads) or `local` (disk) |
| `--input-file` | None | JSON file with image URLs or local paths |
| `--images` | None | One or more local image paths |
| `--images-dir` | None | Directory of images to process recursively |
| `--output-file` | `./results/results_with_ocr.json` | Path for grouped OCR results |
| `--cache-dir` | `./cache` | Persistent binary image cache directory |
| `--url-field` | `imageUrl` | JSON key for image URL (download mode) |
| `--download-dir` | `./shared_downloads` | Directory for downloaded images |
| `--batch-size` | `8` | Images per OCR batch (controls GPU memory) |
| `--max-queue-size` | `50` | Max queued images waiting for OCR |
| `--download-workers` | `3` | Parallel download threads |
| `--max-retries` | `5` | Max download retry attempts |
| `--lang` | `en` | OCR language (`en`, `ch`, `fr`, …) |
| `--device` | `gpu:0` | Inference device (`gpu:0`, `cpu`) |

> Provide at least one of `--input-file`, `--images`, or `--images-dir`.

## Output Structure

Results are grouped by parent Ad ID:

```json
[
    {
        "id": "ad_12345",
        "screenshotPath": ["./images/ad_12345_frame1.png"],
        "images_ocr": [
            {
                "file_path": "./images/ad_12345_frame1.png",
                "ocr_status": "completed",
                "ocr_text": "Extracted text goes here",
                "target_urls": ["https://example.com"]
            }
        ]
    }
]
```

## Status Codes

| Code | Meaning |
|---|---|
| `completed` | Successfully OCR'd |
| `completed_from_cache` | Returned from binary cache (no GPU used) |
| `error_no_url` | Missing URL field in input (download mode) |
| `error_download_failed` | Failed after max retries |
| `error_image_load` | Corrupted or unreadable image |
| `error_file_not_found` | Local path does not exist |

## Performance Tuning

```bash
# Low memory
python ocr/process.py --mode download --input-file data.json --max-queue-size 20 --batch-size 4

# High throughput
python ocr/process.py --mode download --download-workers 10 --batch-size 32 --max-queue-size 200
```

## Troubleshooting

- **CUDA out of memory** — reduce `--batch-size`
- **Stale cache results** — delete `./cache` or set `--cache-dir` to a new path
- **Import errors** — run from the project root (`AdLens/`), not from inside `ocr/`
