import os
import time
import logging
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

logger = logging.getLogger("OCR")

def get_session_with_retries():
    session = requests.Session()
    retries = Retry(total=3, backoff_factor=1, status_forcelist=[500, 502, 503, 504])
    session.mount('http://', HTTPAdapter(max_retries=retries))
    session.mount('https://', HTTPAdapter(max_retries=retries))
    return session

def download_file(url, file_id, download_dir, max_retries=5):
    local_filename = os.path.join(download_dir, f"{file_id}.png")
    session = get_session_with_retries()
    attempt = 0

    while attempt < max_retries:
        try:
            with session.get(url, stream=True, timeout=30) as response:
                if response.status_code == 429:
                    attempt += 1
                    retry_after = int(response.headers.get("Retry-After", 5))
                    logger.warning(f"Rate limit hit for {url}. Sleeping {retry_after}s")
                    time.sleep(retry_after)
                    continue

                response.raise_for_status()

                with open(local_filename, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        if chunk: f.write(chunk)
                
                return True, local_filename

        except Exception as e:
            logger.error(f"Error downloading {url}: {e}")
            attempt += 1
            time.sleep(2 * attempt)

    logger.error(f"Failed to download {url} after {max_retries} attempts.")
    return False, None