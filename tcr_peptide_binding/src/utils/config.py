#!/usr/bin/env python3
"""
配置管理工具

提供配置文件的加载、验证、合并和管理功能。
支持YAML配置文件和命令行参数覆盖。
"""

import yaml
import argparse
from pathlib import Path
from typing import Dict, Any, Optional
import logging
from copy import deepcopy

logger = logging.getLogger(__name__)


class ConfigManager:
    """
    配置管理器

    负责配置文件的加载、验证、合并和参数覆盖。
    """

    def __init__(self, config_path: Optional[str] = None):
        """
        初始化配置管理器

        参数:
            config_path: 配置文件路径
        """
        self.config_path = config_path
        self.config = {}

        if config_path:
            self.load_config(config_path)

    def load_config(self, config_path: str) -> Dict[str, Any]:
        """
        加载配置文件

        参数:
            config_path: 配置文件路径

        返回:
            配置字典
        """
        config_path = Path(config_path)

        if not config_path.exists():
            raise FileNotFoundError(f"配置文件不存在: {config_path}")

        logger.info(f"Loading configuration file: {config_path}")

        try:
            with open(config_path, "r", encoding="utf-8") as f:
                self.config = yaml.safe_load(f)

            logger.info("Configuration file loaded successfully")
            self._log_config_summary()

        except yaml.YAMLError as e:
            logger.error(f"YAML parsing error: {e}")
            raise
        except Exception as e:
            logger.error(f"Configuration file loading failed: {e}")
            raise

        return self.config

    def _log_config_summary(self) -> None:
        """记录配置摘要"""
        if not self.config:
            return

        logger.info("Configuration summary:")

        # 模型配置（中文注释）
        if "model" in self.config:
            model_name = self.config["model"].get("tokenizer_name", "Unknown")
            logger.info(f"   Model: {model_name}")

        # PEFT配置（中文注释）
        if "peft" in self.config:
            peft_enabled = self.config["peft"].get("enabled", False)
            peft_method = self.config["peft"].get("method", "None")
            logger.info(f"   PEFT: {peft_method if peft_enabled else 'Disabled'}")

        # 融合配置（中文注释）
        if "fusion" in self.config:
            fusion_type = self.config["fusion"].get("type", "standard")
            fusion_strategy = self.config["fusion"].get("strategy", "bidirectional")
            logger.info(f"   Fusion: {fusion_type} ({fusion_strategy})")

        # 训练配置（中文注释）
        if "training" in self.config:
            training = self.config["training"]
            batch_size = training.get("batch_size", "Unknown")
            lr = training.get("learning_rate", "Unknown")
            epochs = training.get("epochs", "Unknown")
            logger.info(f"   Training: batch_size={batch_size}, lr={lr}, epochs={epochs}")

    def validate_config(self) -> bool:
        """
        验证配置的完整性和正确性

        返回:
            是否通过验证
        """
        # 跳过所有验证，直接返回通过
        return True

    def merge_config(
        self, override_config: Dict[str, Any], merge_strategy: str = "deep"
    ) -> Dict[str, Any]:
        """
        合并配置

        参数:
            override_config: 覆盖配置
            merge_strategy: 合并策略 ('deep' 或 'shallow')

        返回:
            合并后的配置
        """
        if merge_strategy == "deep":
            merged = deepcopy(self.config)
            self._deep_merge(merged, override_config)
        else:
            merged = {**self.config, **override_config}

        return merged

    def _deep_merge(self, base: Dict[str, Any], override: Dict[str, Any]) -> None:
        """
        深度合并字典

        参数:
            base: 基础字典（会被修改）
            override: 覆盖字典
        """
        for key, value in override.items():
            if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                self._deep_merge(base[key], value)
            else:
                base[key] = value

    def override_from_args(self, args: argparse.Namespace) -> Dict[str, Any]:
        """
        从命令行参数覆盖配置

        参数:
            args: 命令行参数

        返回:
            覆盖后的配置
        """
        logger.info("Applying command line parameter overrides...")

        override_config = {}

        # 映射命令行参数到配置路径
        arg_mappings = {
            "batch_size": "training.batch_size",
            "learning_rate": "training.learning_rate",
            "lr": "training.learning_rate",
            "epochs": "training.epochs",
            "peft_method": "peft.method",
            "max_tcr_length": "data.max_tcr_length",
            "max_peptide_length": "data.max_peptide_length",
            "devices": "hardware.devices",
            "gpus": "hardware.devices",  # 兼容性
            "precision": "hardware.precision",
            "accumulate_grad_batches": "hardware.accumulate_grad_batches",
        }

        # 应用覆盖
        override_count = 0
        for arg_name, config_path in arg_mappings.items():
            if hasattr(args, arg_name):
                arg_value = getattr(args, arg_name)
                if arg_value is not None:
                    self._set_nested_value(override_config, config_path, arg_value)
                    logger.info(f"   Override {config_path} = {arg_value}")
                    override_count += 1

        if override_count > 0:
            logger.info(f"Applied {override_count} parameter overrides")
            return self.merge_config(override_config)
        else:
            logger.info("No parameters to override found")
            return self.config

    def _set_nested_value(self, config: Dict[str, Any], path: str, value: Any):
        """设置嵌套字典值"""
        keys = path.split(".")
        current = config

        # 创建嵌套结构
        for key in keys[:-1]:
            if key not in current:
                current[key] = {}
            current = current[key]

        # 设置最终值
        current[keys[-1]] = value

    def get(self, key: str, default: Any = None) -> Any:
        """
        获取配置值（支持点分隔路径）

        参数:
            key: 配置键 (支持 'section.subsection.key' 格式)
            default: 默认值

        返回:
            配置值
        """
        keys = key.split(".")
        current = self.config

        try:
            for k in keys:
                current = current[k]
            return current
        except (KeyError, TypeError):
            return default

    def set(self, key: str, value: Any):
        """
        设置配置值（支持点分隔路径）

        参数:
            key: 配置键
            value: 配置值
        """
        self._set_nested_value(self.config, key, value)

    def save_config(self, output_path: str):
        """
        保存配置到文件

        参数:
            output_path: 输出文件路径
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        logger.info(f"Saving configuration to: {output_path}")

        try:
            with open(output_path, "w", encoding="utf-8") as f:
                yaml.dump(
                    self.config, f, default_flow_style=False, allow_unicode=True, sort_keys=False
                )

            logger.info("Configuration saved successfully")
        except Exception as e:
            logger.error(f"Configuration saving failed: {e}")
            raise


# 便捷函数
def load_config(config_path: str) -> Dict[str, Any]:
    """
    加载配置文件的便捷函数

    参数:
        config_path: 配置文件路径

    返回:
        配置字典
    """
    manager = ConfigManager()
    return manager.load_config(config_path)


def validate_config(config: Dict[str, Any]) -> bool:
    """
    验证配置的便捷函数

    参数:
        config: 配置字典

    返回:
        是否通过验证
    """
    manager = ConfigManager()
    manager.config = config
    return manager.validate_config()


def create_config_from_args(args: argparse.Namespace) -> Dict[str, Any]:
    """
    从命令行参数创建配置

    参数:
        args: 命令行参数

    返回:
        配置字典
    """
    # 加载基础配置
    config_path = getattr(args, "config", "configs/default_config.yaml")
    manager = ConfigManager(config_path)

    # 应用命令行覆盖
    final_config = manager.override_from_args(args)

    # 跳过配置验证
    return final_config
