#!/usr/bin/env python3


import logging
import logging.handlers
from pathlib import Path
from typing import Dict, Any, Optional
import sys


def setup_logging(config: Optional[Dict[str, Any]] = None):
    """
    设置项目日志配置

    参数:
        config: 日志配置字典
    """

    if config is None:
        config = {}

    # 默认日志级别
    level_str = config.get("level", "INFO").upper()
    level = getattr(logging, level_str, logging.INFO)

    root_logger = logging.getLogger()
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    # 设置根日志器级别
    root_logger.setLevel(level)

    # 创建格式器
    formatter = logging.Formatter(
        fmt="%(asctime)s - %(name)s - %(levelname)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )

    # 控制台处理器
    console_config = config.get("console", {"enabled": True})
    if console_config.get("enabled", True):
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(level)
        console_handler.setFormatter(formatter)
        root_logger.addHandler(console_handler)

    # 文件处理器
    file_config = config.get("file", {})
    if file_config.get("enabled", True):
        log_file = file_config.get("path", "logs/training.log")
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)

        max_size = file_config.get("max_size", "10MB")
        backup_count = file_config.get("backup_count", 5)

        # 解析文件大小
        if isinstance(max_size, str):
            if max_size.endswith("MB"):
                max_bytes = int(max_size[:-2]) * 1024 * 1024
            elif max_size.endswith("KB"):
                max_bytes = int(max_size[:-2]) * 1024
            else:
                max_bytes = int(max_size)
        else:
            max_bytes = max_size

        file_handler = logging.handlers.RotatingFileHandler(
            log_path, maxBytes=max_bytes, backupCount=backup_count, encoding="utf-8"
        )
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)

    # 设置特定库的日志级别
    logging.getLogger("transformers").setLevel(logging.WARNING)
    logging.getLogger("pytorch_lightning").setLevel(logging.INFO)
    logging.getLogger("torch").setLevel(logging.WARNING)

    logger = logging.getLogger(__name__)
    logger.info(f"Logging system initialized (level: {level_str})")

    return root_logger
