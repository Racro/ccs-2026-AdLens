# OCR Pipeline

A modular, high-performance pipeline that ingests ad images (via URL download or local disk), checks them against a fast binary cache, and extracts text/URLs using **PaddleOCR PP-OCRv5** (server models).

## Requirements

- Python 3.12
- CUDA-capable GPU (recommended) — runs fine on CPU
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

### From a JSON file
```bash
python ocr/process.py \
    --input-file results/ads.json \
    --output-file results/results_with_ocr.json
```

### Quick test (specific files or directory)
```bash
python ocr/process.py --images ./test1.png ./test2.jpg
python ocr/process.py --images-dir ./my_scans
```

## Command-line Arguments

| Argument | Default | Description |
|---|---|---|
| `--input-file` | None | JSON file of ad records (needs `screenshotPath` list per record) |
| `--images` | None | One or more local image paths |
| `--images-dir` | None | Directory of images to process recursively |
| `--output-file` | `./results/results_with_ocr.json` | Path for grouped OCR results |
| `--cache-dir` | `./cache` | Persistent binary image cache directory |
| `--batch-size` | `8` | Images per OCR batch (controls memory) |
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
| `completed_from_cache` | Returned from binary cache (no recompute) |
| `error_image_load` | Corrupted or unreadable image |
| `error_file_not_found` | Local path does not exist |

## Performance Tuning

```bash
# Low memory
python ocr/process.py --input-file data.json --batch-size 4

# High throughput
python ocr/process.py --input-file data.json --batch-size 32
```

## Troubleshooting

- **CUDA out of memory** — reduce `--batch-size`
- **Stale cache results** — delete `./cache` or set `--cache-dir` to a new path
- **Import errors** — run from the project root (`AdLens/`), not from inside `ocr/`
