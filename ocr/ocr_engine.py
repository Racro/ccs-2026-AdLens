import logging
from PIL import Image
from bs4 import BeautifulSoup
from urlextract import URLExtract

logger = logging.getLogger("OCR")


class OCREngine:
    """
    PaddleOCR v3.x engine optimized for natural scene/image text (ad screenshots).

    Model choice — PP-OCRv5 server (GPU) / mobile (CPU):
      - Server: 13% better accuracy than PP-OCRv4 on scene text; requires GPU
      - Mobile: suitable for CPU-only inference; much lower memory/compute footprint
      - 'server' variants trade speed for accuracy; appropriate for GPU batch jobs

    PaddleOCR v3 API notes:
      - Constructor uses `device=` instead of v2's `use_gpu=`
      - Inference via `ocr.predict(img_path)` instead of `ocr.ocr()`
      - Results are PaddleOCRResult objects: access `res['rec_texts']` for text lines
    """

    def __init__(self, lang: str = "en", device: str = "gpu:0"):
        try:
            from paddleocr import PaddleOCR
        except ImportError:
            raise ImportError(
                "paddleocr is not installed. Run: pip install paddleocr paddlepaddle-gpu"
            )

        # Server models require GPU — they run full deep convolutions that exhaust
        # CPU memory and get killed by the OS on CPU-only hosts.
        is_cpu = device.lower().startswith("cpu")
        det_model = "PP-OCRv5_mobile_det" if is_cpu else "PP-OCRv5_server_det"
        rec_model = "PP-OCRv5_mobile_rec" if is_cpu else "PP-OCRv5_server_rec"
        variant = "mobile" if is_cpu else "server"

        logger.info(f"Loading PaddleOCR PP-OCRv5 {variant} (lang={lang}, device={device})...")
        self.ocr = PaddleOCR(
            text_detection_model_name=det_model,
            text_recognition_model_name=rec_model,

            # Angle classifier: handles rotated/tilted text in ad screenshots
            use_angle_cls=True,

            lang=lang,
            device=device,

            enable_hpi=False,

            # oneDNN (mkldnn) run mode + PIR API crashes in Paddle 3.x on CPU:
            # ConvertPirAttribute2RuntimeAttribute fails for ArrayAttribute<DoubleAttribute>.
            # Force run_mode="paddle" by disabling mkldnn regardless of device.
            enable_mkldnn=False,

            # Lower box threshold than default (0.6) to avoid missing small
            # fine-print text or thin URL text in ad images
            det_db_box_thresh=0.5,

            # Larger recognition batch = better GPU utilization per predict() call
            rec_batch_num=16,
        )
        logger.info("PaddleOCR ready.")

    def extract_links(self, ocr_text: str) -> list:
        text = ocr_text.replace("-\n", "").replace("\n", " ")
        extractor = URLExtract()
        return list(extractor.gen_urls(text))

    def _result_to_text(self, result) -> str:
        """
        Flattens PaddleOCR v3 output into a plain newline-separated string.

        v3 result structure (one element per input image):
            result[i]['rec_texts']  — list of recognized text strings
            result[i]['rec_scores'] — list of confidence floats (parallel to rec_texts)

        PaddleOCR returns detections in top-to-bottom reading order already.
        """
        if not result:
            return ""
        lines = []
        for res in result:
            texts = res.get("rec_texts", []) if hasattr(res, "get") else []
            lines.extend(t for t in texts if t and t.strip())
        return "\n".join(lines)

    def run_batch_inference(self, tasks: list) -> list:
        """
        Args:
            tasks: list of task dicts, each must have 'file_path'.

        Returns:
            list of {"task": task_dict, "text": clean_text}.
            Failed tasks are skipped (task['status'] set to 'error_image_load').

        PaddleOCR v3 does not support cross-image GPU batching through its public
        API, so we call predict() per image. The internal rec_batch_num still
        batches the recognition stage across text regions within one image.
        """
        results = []
        for task in tasks:
            try:
                img_path = task["file_path"]
                # Verify readability before handing to PaddleOCR
                with Image.open(img_path) as img:
                    img.verify()

                paddle_result = self.ocr.predict(img_path)
                raw_text = self._result_to_text(paddle_result)
                # Strip any residual HTML tags from ad image overlays
                clean_text = BeautifulSoup(raw_text, "html.parser").get_text(separator="\n")
                results.append({"task": task, "text": clean_text})

            except Exception as e:
                logger.error(f"PaddleOCR failed for {task.get('file_path')}: {e}")
                task["status"] = "error_image_load"

        return results
