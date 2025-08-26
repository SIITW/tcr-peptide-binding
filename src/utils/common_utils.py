#!/usr/bin/env python3
"""
通用工具函数

提供项目中常用的工具功能，包括文件操作、时间处理、数据转换等实用工具。
"""

import json
import pickle
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import logging
import hashlib
import pandas as pd
import torch
import numpy as np

logger = logging.getLogger(__name__)


class FileUtils:
    """
    文件操作工具类
    
    提供常用文件操作功能的封装
    """

    @staticmethod
    def ensure_dir(path: Union[str, Path]) -> Path:
        """
        确保目录存在，不存在就创建

        参数:
            path: 目录路径

        返回:
            Path对象
        """
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        return path

    @staticmethod
    def safe_save_json(
        data: Any, filepath: Union[str, Path], indent: int = 2, ensure_ascii: bool = False
    ) -> bool:
        """
        安全保存JSON文件

        参数:
            data: 要保存的数据
            filepath: 文件路径
            indent: JSON缩进
            ensure_ascii: 是否强制ASCII编码

        返回:
            是否成功
        """
        try:
            filepath = Path(filepath)
            FileUtils.ensure_dir(filepath.parent)

            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=indent, ensure_ascii=ensure_ascii)

            logger.debug(f"JSON file saved successfully: {filepath}")
            return True

        except Exception as e:
            logger.error(f"JSON file save failed: {filepath}, error: {e}")
            return False

    @staticmethod
    def safe_load_json(filepath: Union[str, Path]) -> Optional[Any]:
        """
        安全加载JSON文件

        参数:
            filepath: 文件路径

        返回:
            加载的数据，失败返回None
        """
        try:
            filepath = Path(filepath)
            if not filepath.exists():
                logger.warning(f"JSON file does not exist: {filepath}")
                return None

            with open(filepath, "r", encoding="utf-8") as f:
                data = json.load(f)

            logger.debug(f"JSON file loaded successfully: {filepath}")
            return data

        except Exception as e:
            logger.error(f"JSON file load failed: {filepath}, error: {e}")
            return None

    @staticmethod
    def safe_save_pickle(data: Any, filepath: Union[str, Path]) -> bool:
        """
        安全保存Pickle文件

        参数:
            data: 要保存的数据
            filepath: 文件路径

        返回:
            是否成功
        """
        try:
            filepath = Path(filepath)
            FileUtils.ensure_dir(filepath.parent)

            with open(filepath, "wb") as f:
                pickle.dump(data, f)

            logger.debug(f"Pickle file saved successfully: {filepath}")
            return True

        except Exception as e:
            logger.error(f"Pickle file save failed: {filepath}, error: {e}")
            return False

    @staticmethod
    def safe_load_pickle(filepath: Union[str, Path]) -> Optional[Any]:
        """
        安全加载Pickle文件

        参数:
            filepath: 文件路径

        返回:
            加载的数据，失败返回None
        """
        try:
            filepath = Path(filepath)
            if not filepath.exists():
                logger.warning(f"Pickle file does not exist: {filepath}")
                return None

            with open(filepath, "rb") as f:
                data = pickle.load(f)

            logger.debug(f"Pickle file loaded successfully: {filepath}")
            return data

        except Exception as e:
            logger.error(f"Pickle file load failed: {filepath}, error: {e}")
            return None


class TimeUtils:
    """
    时间处理工具类

    提供常用的时间处理和计时功能
    """

    @staticmethod
    def get_timestamp() -> str:
        """获取时间戳字符串"""
        return datetime.now().strftime("%Y%m%d_%H%M%S")

    @staticmethod
    def get_readable_time() -> str:
        """获取可读时间字符串"""
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    @staticmethod
    def format_duration(seconds: float) -> str:
        """
        格式化时长

        参数:
            seconds: 秒数

        返回:
            格式化的时长字符串
        """
        if seconds < 60:
            return f"{seconds:.1f}s"
        elif seconds < 3600:
            minutes = int(seconds // 60)
            secs = seconds % 60
            return f"{minutes}m{secs:.1f}s"
        else:
            hours = int(seconds // 3600)
            minutes = int((seconds % 3600) // 60)
            secs = seconds % 60
            return f"{hours}h{minutes}m{secs:.1f}s"

    @staticmethod
    def timer_context():
        """
        计时器上下文管理器

        使用方法:
            with TimeUtils.timer_context() as timer:
                # 你的代码
                pass
            print(f"耗时: {timer.duration}")
        """

        class TimerContext:
            def __init__(self):
                self.start_time = None
                self.end_time = None
                self.duration = None

            def __enter__(self):
                self.start_time = time.time()
                return self

            def __exit__(self, exc_type, exc_val, exc_tb):
                self.end_time = time.time()
                self.duration = self.end_time - self.start_time

        return TimerContext()


class DataUtils:
    """
    数据处理工具类

    提供常用的数据处理和转换功能
    """

    @staticmethod
    def safe_float_convert(value: Any, default: float = 0.0) -> float:
        """
        安全转换为float类型

        参数:
            value: 要转换的值
            default: 转换失败时的默认值

        返回:
            转换后的float值
        """
        try:
            return float(value)
        except (ValueError, TypeError):
            logger.debug(f"Float conversion failed, using default value: {value} -> {default}")
            return default

    @staticmethod
    def safe_int_convert(value: Any, default: int = 0) -> int:
        """
        安全转换为int类型

        参数:
            value: 要转换的值
            default: 转换失败时的默认值

        返回:
            转换后的int值
        """
        try:
            return int(float(value))  # 先转float再转int，处理科学计数法
        except (ValueError, TypeError):
            logger.debug(f"Int conversion failed, using default value: {value} -> {default}")
            return default

    @staticmethod
    def calculate_class_weights(
        labels: Union[List[int], np.ndarray, torch.Tensor],
    ) -> Dict[int, float]:
        """
        计算类别权重 - 用于不平衡数据集

        参数:
            labels: 标签列表

        返回:
            类别权重字典
        """
        if isinstance(labels, torch.Tensor):
            labels = labels.cpu().numpy()
        elif isinstance(labels, list):
            labels = np.array(labels)

        # 统计各类别数量
        unique_labels, counts = np.unique(labels, return_counts=True)
        total_samples = len(labels)

        # 计算权重 - 采用 total_samples / (num_classes * class_count) 的方法
        num_classes = len(unique_labels)
        weights = {}

        for label, count in zip(unique_labels, counts):
            weights[int(label)] = total_samples / (num_classes * count)

        logger.info("Calculating class weights:")
        for label, weight in weights.items():
            count = dict(zip(unique_labels, counts))[label]
            logger.info(f"   Class {label}: {count} samples, weight {weight:.4f}")

        return weights

    @staticmethod
    def split_dataframe_by_ratio(
        df: pd.DataFrame, ratios: List[float], random_state: int = 42
    ) -> List[pd.DataFrame]:
        """
        按比例分割DataFrame

        参数:
            df: 要分割的DataFrame
            ratios: 分割比例列表，如 [0.7, 0.2, 0.1]
            random_state: 随机种子

        返回:
            分割后的DataFrame列表
        """
        if abs(sum(ratios) - 1.0) > 1e-6:
            raise ValueError(f"Ratios must sum to 1.0, got: {sum(ratios)}")

        # 打乱数据
        df_shuffled = df.sample(frac=1, random_state=random_state).reset_index(drop=True)

        splits = []
        start_idx = 0

        for i, ratio in enumerate(ratios[:-1]):  # 最后一个用剩余的所有数据
            end_idx = start_idx + int(len(df) * ratio)
            splits.append(df_shuffled.iloc[start_idx:end_idx])
            start_idx = end_idx

        # 最后一个分割包含剩余所有数据
        splits.append(df_shuffled.iloc[start_idx:])

        logger.info("DataFrame splitting completed:")
        for i, split_df in enumerate(splits):
            logger.info(f"   Split {i}: {len(split_df)} rows ({100*len(split_df)/len(df):.1f}%)")

        return splits


class HashUtils:
    """
    哈希工具类

    提供常用的哈希值计算功能
    """

    @staticmethod
    def get_file_hash(filepath: Union[str, Path], algorithm: str = "md5") -> Optional[str]:
        """
        获取文件哈希值

        参数:
            filepath: 文件路径
            algorithm: 哈希算法 (md5, sha1, sha256)

        返回:
            哈希值字符串
        """
        try:
            filepath = Path(filepath)
            if not filepath.exists():
                return None

            hash_obj = hashlib.new(algorithm)

            with open(filepath, "rb") as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    hash_obj.update(chunk)

            return hash_obj.hexdigest()

        except Exception as e:
            logger.error(f"File hash calculation failed: {filepath}, error: {e}")
            return None

    @staticmethod
    def get_string_hash(text: str, algorithm: str = "md5") -> str:
        """
        获取字符串哈希值

        参数:
            text: 要哈希的字符串
            algorithm: 哈希算法

        返回:
            哈希值字符串
        """
        hash_obj = hashlib.new(algorithm)
        hash_obj.update(text.encode("utf-8"))
        return hash_obj.hexdigest()


# 便捷函数 - 提供更简洁的接口
def ensure_dir(path: Union[str, Path]) -> Path:
    """确保目录存在的便捷函数"""
    return FileUtils.ensure_dir(path)


def save_json(data: Any, filepath: Union[str, Path], **kwargs) -> bool:
    """保存JSON的便捷函数"""
    return FileUtils.safe_save_json(data, filepath, **kwargs)


def load_json(filepath: Union[str, Path]) -> Optional[Any]:
    """加载JSON的便捷函数"""
    return FileUtils.safe_load_json(filepath)


def get_timestamp() -> str:
    """获取时间戳的便捷函数"""
    return TimeUtils.get_timestamp()


def timer():
    """计时器的便捷函数"""
    return TimeUtils.timer_context()


def safe_float(value: Any, default: float = 0.0) -> float:
    """安全float转换的便捷函数"""
    return DataUtils.safe_float_convert(value, default)


def safe_int(value: Any, default: int = 0) -> int:
    """安全int转换的便捷函数"""
    return DataUtils.safe_int_convert(value, default)
