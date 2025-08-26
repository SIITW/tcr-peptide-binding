#!/usr/bin/env python3
"""
分词器管理工具

用于解决每次初始化数据集都要重新加载ESM++模型的问题。
使用全局单例来管理分词器，节省内存占用。
"""

from typing import Optional, Dict, Any
import logging
from functools import lru_cache

logger = logging.getLogger(__name__)

# 为测试与延迟导入兼容，提供可打补丁的占位符（中文注释）
AutoModelForMaskedLM = None  # will be set on first use


class TokenizerManager:
    """
    分词器管理器 - 单例模式（中文注释）

    负责管理ESM++分词器，避免重复加载浪费内存。
    使用单例模式确保全局只有一个实例，节省内存占用。
    """

    _instance: Optional["TokenizerManager"] = None
    _tokenizers: Dict[str, Any] = {}

    def __new__(cls) -> "TokenizerManager":
        """单例模式实现 - 只能有一个实例"""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            logger.info("Creating tokenizer manager instance")
        return cls._instance

    @classmethod
    @lru_cache(maxsize=5)
    def get_tokenizer(cls, tokenizer_name: str = "Synthyra/ESMplusplus_large"):
        """
        获取分词器 - 带缓存的静态方法（中文注释）

        参数:
            tokenizer_name: 分词器模型名称

        返回:
            分词器实例

        注意（中文注释）：使用LRU缓存，最多缓存5个不同的分词器
        """

        if tokenizer_name in cls._tokenizers:
            logger.debug(f"Reusing cached tokenizer: {tokenizer_name}")
            return cls._tokenizers[tokenizer_name]

        logger.info(f"Loading tokenizer for first time: {tokenizer_name}")
        logger.info("This step may take some time, please wait...")

        try:
            # 延迟导入transformers - 首次使用时再导入，并写回模块级变量（中文注释）
            global AutoModelForMaskedLM
            if AutoModelForMaskedLM is None:
                from transformers import AutoModelForMaskedLM as _AutoModelForMaskedLM
                AutoModelForMaskedLM = _AutoModelForMaskedLM

            # 加载ESM++模型和分词器
            model = AutoModelForMaskedLM.from_pretrained(tokenizer_name, trust_remote_code=True)
            tokenizer = model.tokenizer

            # 缓存分词器
            cls._tokenizers[tokenizer_name] = tokenizer
            logger.info(f"Tokenizer loaded successfully and cached: {tokenizer_name}")

            return tokenizer

        except Exception as e:
            logger.error(f"Tokenizer loading failed: {e}")
            logger.error("Check network connection or use a local model path")
            raise RuntimeError(f"Failed to load tokenizer: {tokenizer_name}") from e

    @classmethod
    def clear_cache(cls):
        """清除分词器缓存 - 释放内存（中文注释）"""
        logger.info("Clearing tokenizer cache")
        cls._tokenizers.clear()
        # 清除LRU缓存
        cls.get_tokenizer.cache_clear()

    @classmethod
    def get_cache_info(cls) -> Dict[str, Any]:
        """获取缓存信息 - 查看内存占用情况（中文注释）"""
        cache_info = cls.get_tokenizer.cache_info()

        return {
            "cached_tokenizers": list(cls._tokenizers.keys()),
            "cache_hits": cache_info.hits,
            "cache_misses": cache_info.misses,
            "cache_size": cache_info.currsize,
            "max_cache_size": cache_info.maxsize,
        }


# 便捷函数
def get_tokenizer(tokenizer_name: str = "Synthyra/ESMplusplus_large"):
    """
    获取分词器的便捷函数

    参数:
        tokenizer_name: 分词器名称

    返回:
        分词器实例
    """
    return TokenizerManager.get_tokenizer(tokenizer_name)


def clear_tokenizer_cache():
    """清除分词器缓存的便捷函数"""
    TokenizerManager.clear_cache()


def get_tokenizer_cache_info() -> Dict[str, Any]:
    """获取分词器缓存信息的便捷函数"""
    return TokenizerManager.get_cache_info()
