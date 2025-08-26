#!/usr/bin/env python3
"""
数据验证和检查工具

提供统一的数据验证和检查功能。
包含序列验证、配置验证、数据格式检查等验证功能。
"""

from typing import List, Dict, Any, Optional
import logging
import pandas as pd

logger = logging.getLogger(__name__)


class SequenceValidator:
    """
    蛋白质序列验证器

    用于检查序列质量，包括：
    - 氨基酸字符检查
    - 序列长度验证
    - 非标准字符检测
    """

    # 标准20种氨基酸
    STANDARD_AA = set("ACDEFGHIKLMNPQRSTVWY")

    # 扩展氨基酸（包括一些特殊情况）
    EXTENDED_AA = STANDARD_AA | set("BJOUXZ*-")

    @classmethod
    def is_valid_protein_sequence(cls, sequence: str, allow_extended: bool = False) -> bool:
        """
        检查是否为有效的蛋白质序列

        参数:
            sequence: 蛋白质序列字符串
            allow_extended: 是否允许扩展氨基酸字符

        返回:
            是否有效
        """
        if not sequence or not isinstance(sequence, str):
            return False

        # 转换为大写
        sequence = sequence.upper().strip()

        # 检查字符集
        allowed_chars = cls.EXTENDED_AA if allow_extended else cls.STANDARD_AA

        return set(sequence).issubset(allowed_chars)

    @classmethod
    def get_invalid_characters(cls, sequence: str, allow_extended: bool = False) -> List[str]:
        """
        获取序列中的非法字符

        参数:
            sequence: 蛋白质序列
            allow_extended: 是否允许扩展氨基酸

        返回:
            非法字符列表
        """
        if not sequence:
            return []

        sequence = sequence.upper().strip()
        allowed_chars = cls.EXTENDED_AA if allow_extended else cls.STANDARD_AA

        invalid_chars = list(set(sequence) - allowed_chars)
        return sorted(invalid_chars)

    @classmethod
    def validate_sequence_batch(
        cls,
        sequences: List[str],
        sequence_type: str = "protein",
        min_length: int = 1,
        max_length: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        批量验证序列

        参数:
            sequences: 序列列表
            sequence_type: 序列类型（用于日志）
            min_length: 最小长度
            max_length: 最大长度

        返回:
            验证结果统计
        """
        if not sequences:
            return {"total": 0, "valid": 0, "invalid": 0, "invalid_indices": [], "issues": []}

        valid_count = 0
        invalid_indices = []
        issues = []

        for i, seq in enumerate(sequences):
            seq_issues = []

            # 检查基本有效性
            if not cls.is_valid_protein_sequence(seq):
                invalid_chars = cls.get_invalid_characters(seq)
                if invalid_chars:
                    seq_issues.append(f"Invalid characters: {invalid_chars}")

            # 检查长度
            if len(seq) < min_length:
                seq_issues.append(f"Sequence too short: {len(seq)} < {min_length}")

            if max_length and len(seq) > max_length:
                seq_issues.append(f"Sequence too long: {len(seq)} > {max_length}")

            # 检查空序列
            if not seq.strip():
                seq_issues.append("Empty sequence")

            if seq_issues:
                invalid_indices.append(i)
                issues.append(
                    {
                        "index": i,
                        "sequence": seq[:20] + "..." if len(seq) > 20 else seq,
                        "issues": seq_issues,
                    }
                )
            else:
                valid_count += 1

        result = {
            "total": len(sequences),
            "valid": valid_count,
            "invalid": len(invalid_indices),
            "invalid_indices": invalid_indices,
            "issues": issues,
        }

        # 记录验证结果
        logger.info(f"{sequence_type} sequence validation results:")
        logger.info(f"   Total: {result['total']}")
        logger.info(f"   Valid: {result['valid']} ({100*result['valid']/result['total']:.1f}%)")
        logger.info(f"   Invalid: {result['invalid']} ({100*result['invalid']/result['total']:.1f}%)")

        # 报告问题
        if issues:
            logger.warning(f"Found {len(issues)} problematic sequences:")
            for issue in issues[:5]:  # 只显示前5个（中文注释）
                logger.warning(f"   Sequence {issue['index']}: {', '.join(issue['issues'])}")
            if len(issues) > 5:
                logger.warning(f"   ... and {len(issues)-5} more problematic sequences")

        return result


class DataFrameValidator:
    """
    DataFrame验证器

    用于检查训练数据格式的工具
    """

    @classmethod
    def validate_tcr_peptide_dataframe(
        cls, df: pd.DataFrame, required_columns: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        验证TCR-peptide数据框

        参数:
            df: 数据框
            required_columns: 必需列名

        返回:
            验证结果
        """
        if required_columns is None:
            required_columns = ["TCR", "Peptide", "Label"]

        issues = []

        # 检查基本格式
        if df is None:
            return {"valid": False, "issues": ["No data found"]}

        if df.empty:
            return {"valid": False, "issues": ["No data found"]}

        # 检查必需列
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            issues.append(f"Missing required columns: {missing_columns}")

        # 检查数据类型
        if "Label" in df.columns:
            unique_labels = df["Label"].unique()
            if not set(unique_labels).issubset({0, 1}):
                issues.append(f"Invalid label values, expected 0/1, got: {unique_labels}")

        # 检查缺失值
        missing_stats = {}
        for col in required_columns:
            if col in df.columns:
                missing_count = df[col].isna().sum()
                if missing_count > 0:
                    missing_stats[col] = missing_count
                    issues.append(
                        f"Missing values found: column '{col}' has {missing_count} missing values"
                    )

        # 检查重复数据
        duplicate_count = df.duplicated().sum()
        if duplicate_count > 0:
            issues.append(f"Found {duplicate_count} duplicated rows")

        # 序列验证
        if "TCR" in df.columns and "Peptide" in df.columns:
            tcr_validation = SequenceValidator.validate_sequence_batch(
                df["TCR"].fillna("").tolist(), sequence_type="TCR"
            )
            peptide_validation = SequenceValidator.validate_sequence_batch(
                df["Peptide"].fillna("").tolist(), sequence_type="Peptide"
            )

            if tcr_validation["invalid"] > 0:
                issues.append(f"TCR sequence validation failed: {tcr_validation['invalid']} invalid sequences")

            if peptide_validation["invalid"] > 0:
                issues.append(f"Peptide sequence validation failed: {peptide_validation['invalid']} invalid sequences")

        return {
            "valid": len(issues) == 0,
            "issues": issues,
            "missing_stats": missing_stats,
            "duplicate_count": duplicate_count,
            "total_rows": len(df),
        }


class ConfigValidator:
    """
    配置验证器

    用于检查配置文件的正确性
    """

    @classmethod
    def validate_training_config(cls, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        验证训练配置

        参数:
            config: 配置字典

        返回:
            验证结果
        """
        issues = []
        warnings = []

        # 检查必需的配置节
        required_sections = ["model", "training", "data"]
        missing_sections = [s for s in required_sections if s not in config]
        if missing_sections:
            issues.append(f"Missing config sections: {missing_sections}")

        # 验证训练参数
        if "training" in config:
            training = config["training"]

            # 批次大小
            batch_size = training.get("batch_size", 4)
            if not isinstance(batch_size, int) or batch_size <= 0:
                issues.append(f"batch_size must be positive integer, got: {batch_size}")
            elif batch_size > 64:
                warnings.append(f"Large batch_size ({batch_size}) may cause OOM")

            # 学习率
            lr = training.get("learning_rate", 2e-5)
            try:
                lr_float = float(lr)
                if lr_float <= 0 or lr_float > 1:
                    warnings.append(f"Learning rate may be unreasonable: {lr_float}")
            except (ValueError, TypeError):
                issues.append(f"Invalid learning rate format: {lr}")

            # 训练轮数
            epochs = training.get("epochs", 20)
            if not isinstance(epochs, int) or epochs <= 0:
                issues.append(f"epochs must be a positive integer, got: {epochs}")

        # 验证数据配置
        if "data" in config:
            data = config["data"]

            # 序列长度
            max_tcr_length = data.get("max_tcr_length", 128)
            max_peptide_length = data.get("max_peptide_length", 64)

            if not isinstance(max_tcr_length, int) or max_tcr_length <= 0:
                issues.append(f"max_tcr_length must be a positive integer, got: {max_tcr_length}")

            if not isinstance(max_peptide_length, int) or max_peptide_length <= 0:
                issues.append(f"max_peptide_length must be a positive integer, got: {max_peptide_length}")

        # 验证PEFT配置
        if "peft" in config:
            peft = config["peft"]
            method = str(peft.get("method", "lora")).lower()
            valid_methods = ["lora", "adalora", "vera", "boft", "fourierft", "oft", "ia3"]

            if method not in valid_methods:
                issues.append(f"Invalid PEFT method: {method}, valid: {valid_methods}")

        result = {"valid": len(issues) == 0, "issues": issues, "warnings": warnings}

        # 记录验证结果
        if issues:
            logger.error(f"Configuration validation failed, found {len(issues)} issues:")
            for issue in issues:
                logger.error(f"   - {issue}")

        if warnings:
            logger.warning(f"Configuration validation warnings, found {len(warnings)} issues:")
            for warning in warnings:
                logger.warning(f"   - {warning}")

        if result["valid"] and not warnings:
            logger.info("Configuration validation passed")

        return result


# 便捷函数
def validate_sequences(
    sequences: List[str], sequence_type: str = "protein", **kwargs
) -> Dict[str, Any]:
    """验证序列列表的便捷函数"""
    return SequenceValidator.validate_sequence_batch(sequences, sequence_type, **kwargs)


def validate_dataframe(df: pd.DataFrame, **kwargs) -> Dict[str, Any]:
    """验证DataFrame的便捷函数"""
    return DataFrameValidator.validate_tcr_peptide_dataframe(df, **kwargs)


def validate_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """验证配置的便捷函数"""
    return ConfigValidator.validate_training_config(config)
