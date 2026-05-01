import hashlib
import json
import os
import sys
import logging
import datetime

SCRIPT_NAME = "ImageCache"

os.makedirs("logs", exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - [%(levelname)s] - %(message)s",
    handlers=[
        logging.FileHandler(f"logs/{SCRIPT_NAME}.log"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(SCRIPT_NAME)


class ImgCacheController:
    db = {}
    
    def __init__(self, save_dir, save_interval=10):
        self.save_interval = save_interval
        self.unsaved_count = 0
        self.save_dir = save_dir
        self.db_file = os.path.join(save_dir, "image_cache.json")
        os.makedirs(save_dir, exist_ok=True)
        logger.info(f"Initializing ImgCacheController with save_dir: {save_dir}, save_interval: {save_interval}")
        self.__load_db()
    
    def __save_db(self):
        """Write the database to a JSON file."""
        try:
            with open(self.db_file, 'w') as f:
                json.dump(self.db, f, indent=4)
            logger.info(f"Cache database saved to {self.db_file} ({len(self.db)} entries)")
            self.unsaved_count = 0
        except Exception as e:
            logger.error(f"Error saving cache database: {e}")
        
    def __load_db(self):
        """Read the database from a JSON file."""
        if os.path.exists(self.db_file):
            try:
                with open(self.db_file, 'r') as f:
                    self.db = json.load(f)
                logger.info(f"Cache database loaded from {self.db_file} ({len(self.db)} entries)")
            except Exception as e:
                logger.warning(f"Error loading cache database: {e}. Starting with empty cache.")
                self.db = {}
        else:
            logger.info(f"Cache database file not found. Starting with empty cache.")
            self.db = {}
    
    def __get_hash(self, img):
        return hashlib.md5(img).hexdigest()
    
    def get_image_data(self, img) -> dict:
        """Get cached image data by hash. Returns None if not found."""
        hash = self.__get_hash(img)
        
        if hash in self.db:
            logger.debug(f"Cache hit for hash: {hash}")
            return self.db[hash]["data"]
        
        logger.debug(f"Cache miss for hash: {hash}")
        return None
    
    def set_image_data(self, img, data) -> str:
        """Set image data in cache and save if needed."""
        hash = self.__get_hash(img)
        
        self.db[hash] = {
            "data": data,
            "cached_at": datetime.datetime.now().isoformat()
        }
        self.unsaved_count += 1
        logger.debug(f"Cached data for hash: {hash} (unsaved: {self.unsaved_count}/{self.save_interval})")
        
        if self.save_interval <= self.unsaved_count:
            logger.info(f"Reaching save interval threshold. Saving cache to disk.")
            self.__save_db()
    
    def process_with_cache(self, img, processor):
        """Process image with caching. If cached, return cached result, else process and cache."""
        hash = self.__get_hash(img)
        data = self.get_image_data(img)
        
        if data:
            logger.info(f"Using cached result for image hash: {hash}")
            return data
        
        logger.info(f"Processing new image with hash: {hash}")
        data = processor(img)
        self.set_image_data(img, data)
        
        return data
    
    def __enter__(self):
        """Enter context manager."""
        logger.debug("Entering ImgCacheController context manager")
        return self
    
    def __exit__(self, exc_type, exc, tb):
        """Exit context manager and save database."""
        if exc_type:
            logger.warning(f"Exiting context manager with exception: {exc_type.__name__}: {exc}")
        
        logger.info("Saving cache database on exit")
        self.__save_db()


if __name__ == "__main__":
    #################### THIS IS AN EXAMPLE #################### 
    ############# DON'T RUN THIS SCRIPT STANDALONE ############# 
    def process_image(img_bytes):
        """Example processor: computes image size and additional metadata."""
        return {
            "size_bytes": len(img_bytes),
            "hash": hashlib.sha256(img_bytes).hexdigest()
        }
    
    # Use the cache controller with context manager
    screenshots_dir = "./results/screenshots"
    cache_dir = "./results/cache"
    
    logger.info("Starting image cache controller example")
    logger.info(f"Processing screenshots from: {screenshots_dir}")
    
    with ImgCacheController(cache_dir, save_interval=5) as cache:
        # Process all PNG files in the screenshots directory
        if os.path.exists(screenshots_dir):
            png_files = [f for f in os.listdir(screenshots_dir) if f.lower().endswith('.png')]
            logger.info(f"Found {len(png_files)} PNG files to process")
            
            for i, filename in enumerate(png_files):
                filepath = os.path.join(screenshots_dir, filename)
                try:
                    with open(filepath, 'rb') as f:
                        img_bytes = f.read()
                    
                    # Method 1: Using process_with_cache (automatic caching)
                    result = cache.process_with_cache(img_bytes, process_image)
                    logger.info(f"[{i+1}/{len(png_files)}] {filename}: {result['size_bytes']} bytes")
                    
                except Exception as e:
                    logger.error(f"Error processing {filename}: {e}")
            
            logger.info(f"Cache statistics: {len(cache.db)} total cached entries")
            
            # Method 2: Using set_image_data and get_image_data directly
            logger.info("\n--- Demonstrating direct get_image_data and set_image_data usage ---")
            
            if png_files:
                # Pick first file as example
                first_file = png_files[0]
                test_filepath = os.path.join(screenshots_dir, first_file)
                
                with open(test_filepath, 'rb') as f:
                    test_img_bytes = f.read()
                
                # Manually set data using set_image_data
                custom_data = {
                    "filename": first_file,
                    "processed_at": str(__import__('datetime').datetime.now()),
                    "format": "PNG"
                }
                cache.set_image_data(test_img_bytes, custom_data)
                logger.info(f"Manually cached data for {first_file}")
                
                # Retrieve data using get_image_data
                retrieved_data = cache.get_image_data(test_img_bytes)
                if retrieved_data:
                    logger.info(f"Retrieved cached data: {retrieved_data}")
                else:
                    logger.warning("Failed to retrieve cached data")
        else:
            logger.error(f"Screenshots directory not found: {screenshots_dir}")
    
    logger.info("Example completed - cache saved automatically")