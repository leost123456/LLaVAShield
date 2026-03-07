import logging
import os

from datetime import datetime
from config import Config

log_dir = Config.output_dir
os.makedirs(log_dir, exist_ok=True)

NOW = datetime.now()
ts = NOW.strftime("%Y-%m-%d_%H-%M-%S")

formatter = logging.Formatter(
    "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO) 
console_handler.setFormatter(formatter)

file_handler = logging.FileHandler(os.path.join(log_dir, f"pipeline_{ts}.log"), mode="w", encoding="utf-8")
file_handler.setLevel(logging.DEBUG)  # 文件里保留所有调试信息
file_handler.setFormatter(formatter)

def get_logger(name):
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    if not logger.hasHandlers():
        logger.addHandler(console_handler)
        logger.addHandler(file_handler)
    return logger