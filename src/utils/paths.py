#!/usr/bin/env python3


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
        初始化

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

        self.all_dirs = [
            self.base_dir,
            self.checkpoints_dir,
            self.logs_dir,
            self.results_dir,
            self.plots_dir,
            self.predictions_dir,
        ]

    def setup_directories(self):
        """创建必需目录"""

        logger.info("Setting up output directories...")

        for directory in self.all_dirs:
            directory.mkdir(parents=True, exist_ok=True)
            logger.debug(f"   Created directory: {directory}")

        logger.info(f"Output directories setup completed: {self.base_dir}")
