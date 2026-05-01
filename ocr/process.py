import os
import sys
import json
import uuid
import logging
import argparse
from pathlib import Path

# Allow imports from ocr/ dir (ocr_engine) and project root (utils)
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, '..'))
sys.path.insert(0, parent_dir)
sys.path.insert(0, current_dir)

from ocr_engine import OCREngine
from utils import ImgCacheController

os.makedirs("logs", exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - [%(levelname)s] - %(message)s',
    handlers=[
        logging.FileHandler("logs/pipeline.log"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("OCR")


def parse_args():
    parser = argparse.ArgumentParser(description="PaddleOCR Pipeline")

    # Input sources
    parser.add_argument('--input-file', default=None,
                        help='JSON file of ad records. Each record needs a "screenshotPath" list.')
    parser.add_argument('--images', nargs='+', metavar='IMAGE',
                        help='One or more local image file paths to OCR directly.')
    parser.add_argument('--images-dir', default=None, metavar='DIR',
                        help='Directory of images (.png/.jpg/.jpeg/.webp) to OCR recursively.')

    # Config
    parser.add_argument('--output-file', default='./results/results_with_ocr.json', help='Path to save OCR results')
    parser.add_argument('--batch-size', type=int, default=8, help='Images per OCR batch (controls memory)')
    parser.add_argument('--cache-dir', default='./cache', help='Directory for image cache database')

    # PaddleOCR-specific
    parser.add_argument('--lang', default='en', help='OCR language (e.g. en, ch, fr). See PaddleOCR docs.')
    parser.add_argument('--device', default='gpu:0',
                        help='Inference device (default: gpu:0). Use "cpu" for CPU-only sanity checks.')

    args = parser.parse_args()

    if not args.input_file and not args.images and not args.images_dir:
        parser.error("one of --input-file, --images, or --images-dir is required")

    return args


def process_batch(batch, ocr_engine, final_results, args, cache):
    """Runs a batch through PaddleOCR and saves progress."""
    logger.info(f"Running inference on batch of {len(batch)} items...")
    batch_results = ocr_engine.run_batch_inference(batch)

    for i, res in enumerate(batch_results):
        batch[i]['ocr_text'] = res['text']
        batch[i]['target_urls'] = ocr_engine.extract_links(res['text'])
        batch[i]['ocr_status'] = 'completed'

        img_bytes = batch[i].pop('img_bytes', None)
        if img_bytes:
            cache.set_image_data(img_bytes, {
                'ocr_text': batch[i]['ocr_text'],
                'target_urls': batch[i]['target_urls']
            })

        if 'status' in batch[i]:
            del batch[i]['status']

    final_results.extend(batch)
    save_grouped_results(final_results, args.output_file)


def run_local_mode(args, data, ocr_engine, cache):
    logger.info(f"Starting Local Mode (Total: {len(data)})")
    final_results = []
    batch = []

    for task in data:
        if not os.path.isfile(task['file_path']):
            logger.error(f"Local file not found: {task['file_path']}")
            task['ocr_status'] = 'error_file_not_found'
            final_results.append(task)
            continue

        with open(task['file_path'], 'rb') as f:
            img_bytes = f.read()

        cached_data = cache.get_image_data(img_bytes)

        if cached_data:
            logger.info(f"Cache hit for {task.get('ad_id', task.get('id'))}")
            task['ocr_text'] = cached_data['ocr_text']
            task['target_urls'] = cached_data['target_urls']
            task['ocr_status'] = 'completed_from_cache'
            if 'status' in task: del task['status']
            final_results.append(task)
        else:
            task['img_bytes'] = img_bytes
            batch.append(task)

        if len(batch) >= args.batch_size:
            process_batch(batch, ocr_engine, final_results, args, cache)
            batch = []

    if batch:
        process_batch(batch, ocr_engine, final_results, args, cache)

    logger.info(f"Pipeline finished! Results saved to {args.output_file}")


def save_grouped_results(final_results, output_file):
    """Groups flat image results back into their parent ad records."""
    grouped = {}

    for task in final_results:
        ad_id = task.get('ad_id', task.get('id'))

        if ad_id not in grouped:
            grouped[ad_id] = task.get('original_data', {'id': ad_id}).copy()
            grouped[ad_id]['images_ocr'] = []

        grouped[ad_id]['images_ocr'].append({
            'file_path': task.get('file_path'),
            'ocr_status': task.get('ocr_status'),
            'ocr_text': task.get('ocr_text'),
            'target_urls': task.get('target_urls')
        })

    os.makedirs(os.path.dirname(output_file) or '.', exist_ok=True)
    with open(output_file, 'w') as f:
        json.dump(list(grouped.values()), f, indent=4)


def main():
    args = parse_args()
    data = []

    if args.input_file:
        with open(args.input_file, 'r') as f:
            file_tasks = json.load(f)

        for item in file_tasks:
            if not item: continue
            ad_id = item.get('id', str(uuid.uuid4()))
            item['id'] = ad_id
            paths = item.get("screenshotPath", [])
            for p in paths:
                data.append({
                    'ad_id': ad_id,
                    'file_path': p,
                    'original_data': item
                })

    if args.images:
        for path_str in args.images:
            p = Path(path_str)
            data.append({'id': p.stem, 'file_path': str(p.resolve())})

    if args.images_dir:
        _EXTS = {'.png', '.jpg', '.jpeg', '.webp'}
        dir_path = Path(args.images_dir)
        if dir_path.is_dir():
            for p in sorted(p for p in dir_path.rglob('*') if p.suffix.lower() in _EXTS):
                data.append({'id': p.stem, 'file_path': str(p.resolve())})

    if not data:
        logger.error("No tasks to process. Check your inputs.")
        sys.exit(1)

    ocr_engine = OCREngine(lang=args.lang, device=args.device)

    with ImgCacheController(args.cache_dir, save_interval=10) as cache:
        run_local_mode(args, data, ocr_engine, cache)


if __name__ == "__main__":
    main()
