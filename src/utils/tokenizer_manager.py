#!/usr/bin/env python3


from typing import Optional, Dict, Any
import logging
from functools import lru_cache

logger = logging.getLogger(__name__)

AutoModelForMaskedLM = None  # 首次使用时设置


class TokenizerManager:
    """
    分词器管理器 - 单例模式

    负责管理ESM++分词器，避免重复加载浪费内存。
    使用单例模式确保全局只有一个实例，节省内存占用。
    """

    _instance: Optional["TokenizerManager"] = None
    _tokenizers: Dict[str, Any] = {}

    def __new__(cls) -> "TokenizerManager":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            logger.info("Creating tokenizer manager instance")
        return cls._instance

    @classmethod
    @lru_cache(maxsize=5)
    def get_tokenizer(cls, tokenizer_name: str = "Synthyra/ESMplusplus_large"):
        """
        获取分词器 - 带缓存的静态方法

        参数:
            tokenizer_name: 分词器模型名称

        返回:
            分词器实例
        """

        if tokenizer_name in cls._tokenizers:
            logger.debug(f"Reusing cached tokenizer: {tokenizer_name}")
            return cls._tokenizers[tokenizer_name]

        logger.info(f"Loading tokenizer for first time: {tokenizer_name}")
        logger.info("This step may take some time, please wait...")

        try:
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



def get_tokenizer(tokenizer_name: str = "Synthyra/ESMplusplus_large"):
    """
    获取分词器的便捷函数

    参数:
        tokenizer_name: 分词器名称

    返回:
        分词器实例
    """
    return TokenizerManager.get_tokenizer(tokenizer_name)
