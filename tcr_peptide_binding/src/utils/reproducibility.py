#!/usr/bin/env python3
"""
可重现性工具

提供随机种子设置和确定性训练配置。
"""

import os
import random
import numpy as np
import torch
import pytorch_lightning as pl
import logging

logger = logging.getLogger(__name__)


def set_seed(seed: int = 42):
    """
    设置所有相关库的随机种子

    参数:
        seed: 随机种子值
    """

    logger.info(f"Setting random seed: {seed}")

    # Python随机数
    random.seed(seed)

    # NumPy随机数
    np.random.seed(seed)

    # PyTorch随机数
    torch.manual_seed(seed)

    # CUDA随机数
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    # PyTorch Lightning种子
    pl.seed_everything(seed, workers=True)

    # 环境变量设置
    os.environ["PYTHONHASHSEED"] = str(seed)


def set_deterministic():
    """
    设置确定性训练模式

    注意：这可能会影响性能，但能确保完全可重现的结果
    """

    logger.info("Enabling deterministic training mode")

    # PyTorch确定性设置
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # 使用确定性算法
    torch.use_deterministic_algorithms(True, warn_only=True)

    # 环境变量设置
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
