#!/usr/bin/env python3
"""
路径管理工具

提供项目路径管理和目录创建功能。
"""

from pathlib import Path
from typing import Optional, Union
import logging

logger = logging.getLogger(__name__)


class PathManager:
    """
    路径管理器

    管理项目的各种输出路径，确保目录存在。
    """

    def __init__(self, base_dir: Union[str, Path], experiment_name: Optional[str] = None):
        """
        初始化路径管理器

        参数:
            base_dir: 基础输出目录
            experiment_name: 实验名称（用于创建子目录）
        """
        self.base_dir = Path(base_dir)

        if experiment_name:
            self.base_dir = self.base_dir / experiment_name

        # 定义子目录
        self.checkpoints_dir = self.base_dir / "checkpoints"
        self.logs_dir = self.base_dir / "logs"
        self.results_dir = self.base_dir / "results"
        self.plots_dir = self.base_dir / "plots"
        self.predictions_dir = self.base_dir / "predictions"

        # 所有需要创建的目录
        self.all_dirs = [
            self.base_dir,
            self.checkpoints_dir,
            self.logs_dir,
            self.results_dir,
            self.plots_dir,
            self.predictions_dir,
        ]

    def setup_directories(self):
        """创建所有必需的目录"""

        logger.info("Setting up output directories...")

        for directory in self.all_dirs:
            directory.mkdir(parents=True, exist_ok=True)
            logger.debug(f"   Created directory: {directory}")

        logger.info(f"Output directories setup completed: {self.base_dir}")

    def get_unique_path(self, directory: Path, filename: str, extension: str = "") -> Path:
        """
        获取唯一的文件路径（避免覆盖）

        参数:
            directory: 目标目录
            filename: 文件名
            extension: 文件扩展名

        返回:
            唯一的文件路径
        """
        base_path = directory / f"{filename}{extension}"

        if not base_path.exists():
            return base_path

        # 如果文件已存在，添加数字后缀
        counter = 1
        while True:
            new_path = directory / f"{filename}_{counter}{extension}"
            if not new_path.exists():
                return new_path
            counter += 1


def ensure_dir(path: Union[str, Path]) -> Path:
    """
    确保目录存在

    参数:
        path: 目录路径

    返回:
        Path对象
    """
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path
