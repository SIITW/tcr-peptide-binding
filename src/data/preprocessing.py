#!/usr/bin/env python3

import pandas as pd
import numpy as np
from typing import List, Tuple, Optional, Dict
import re
import logging

logger = logging.getLogger(__name__)


class SequencePreprocessor:
    """
    序列预处理器
    
    提供氨基酸序列的清理、验证与标准化功能
    """

    # 标准氨基酸字母 - 严格限制20种标准氨基酸
    STANDARD_AA = set("ACDEFGHIKLMNPQRSTVWY")

    def __init__(
        self,
        min_length: int = 3,
        max_length: Optional[int] = None,
    ):
        """
        初始化预处理器（仅允许标准20种氨基酸）
        说明：不再移除非法字符，包含非标准氨基酸的序列将被直接判定为不合法

        参数:
            min_length: 最小序列长度
            max_length: 最大序列长度（可选）
        """
        self.min_length = min_length
        self.max_length = max_length

        self.valid_chars = self.STANDARD_AA

        logger.info("Sequence preprocessor configuration:")
        logger.info("   Only standard 20 amino acids allowed: ACDEFGHIKLMNPQRSTVWY")
        logger.info("   Non-standard sequences will be REJECTED entirely")
        logger.info(f"   Length range: {min_length} - {max_length or 'unlimited'}")

    def clean_sequence(self, sequence: str) -> str:
        """
        清理单条序列
        含非标准氨基酸的序列会在 validate_sequence 中被判定为不合法

        参数:
            sequence: 原始序列

        返回:
            清理后的序列
        """
        if not sequence or not isinstance(sequence, str):
            return ""
        seq = sequence.upper().strip()
        seq = re.sub(r"\s+", "", seq)

        return seq

    def validate_sequence(self, sequence: str) -> Tuple[bool, List[str]]:
        """
        验证序列

        参数:
            sequence: 序列

        返回:
            (是否有效, 问题列表)
        """
        issues = []

        # 检查长度
        if len(sequence) < self.min_length:
            issues.append(f"Sequence too short ({len(sequence)} < {self.min_length})")

        if self.max_length and len(sequence) > self.max_length:
            issues.append(f"Sequence too long ({len(sequence)} > {self.max_length})")

        # 检查字符
        invalid_chars = set(sequence) - self.valid_chars
        if invalid_chars:
            issues.append(f"Contains invalid characters: {sorted(invalid_chars)}")

        return len(issues) == 0, issues

    def process_sequences(
        self, sequences: List[str], sequence_type: str = "sequence"
    ) -> Tuple[List[str], Dict]:
        """
        批量处理序列

        参数:
            sequences: 序列列表
            sequence_type: 序列类型
        返回:
            处理后的序列列表, 统计信息
        """
        logger.info(f"Processing {len(sequences)} {sequence_type} sequences...")

        processed_sequences = []
        stats = {
            "original_count": len(sequences),
            "cleaned_count": 0,
            "valid_count": 0,
            "invalid_count": 0,
            "empty_count": 0,
            "issues": [],
        }

        for i, seq in enumerate(sequences):
            # 清理序列
            cleaned_seq = self.clean_sequence(seq)
            processed_sequences.append(cleaned_seq)

            if cleaned_seq:
                stats["cleaned_count"] += 1

                # 验证序列
                is_valid, issues = self.validate_sequence(cleaned_seq)

                if is_valid:
                    stats["valid_count"] += 1
                else:
                    stats["invalid_count"] += 1
                    stats["issues"].append(f"Sequence {i+1}: {', '.join(issues)}")
            else:
                stats["empty_count"] += 1
                stats["issues"].append(f"Sequence {i+1}: empty after cleaning")

        # 记录统计信息
        logger.info(f"{sequence_type} processing completed:")
        logger.info(f"   Original sequences: {stats['original_count']}")
        logger.info(f"   Non-empty after cleaning: {stats['cleaned_count']}")
        logger.info(f"   Valid sequences: {stats['valid_count']}")
        logger.info(f"   Invalid sequences: {stats['invalid_count']}")
        logger.info(f"   Empty sequences: {stats['empty_count']}")

        if stats["issues"]:
            logger.warning(f"   Found issues: {len(stats['issues'])}")
            # 只显示前几个问题
            for issue in stats["issues"][:5]:
                logger.warning(f"     - {issue}")
            if len(stats["issues"]) > 5:
                logger.warning(f"     - ... and {len(stats['issues'])-5} more issues")

        return processed_sequences, stats


def analyze_data_quality(df: pd.DataFrame) -> Dict:
    """
    分析数据质量（中文注释）

    参数:
        df: 数据DataFrame

    返回:
        质量分析报告
    """

    logger.info("Analyzing data quality...")

    report = {
        "basic_stats": {},
        "sequence_stats": {},
        "label_distribution": {},
        "quality_issues": [],
    }

    # 基础统计
    report["basic_stats"] = {
        "total_samples": len(df),
        "missing_tcr": df["TCR"].isna().sum(),
        "missing_peptide": df["Peptide"].isna().sum(),
        "missing_label": df["Label"].isna().sum() if "Label" in df else 0,
    }

    # 序列统计
    if "TCR" in df:
        tcr_lengths = df["TCR"].str.len()
        report["sequence_stats"]["tcr"] = {
            "mean_length": tcr_lengths.mean(),
            "min_length": tcr_lengths.min(),
            "max_length": tcr_lengths.max(),
            "std_length": tcr_lengths.std(),
        }

    if "Peptide" in df:
        peptide_lengths = df["Peptide"].str.len()
        report["sequence_stats"]["peptide"] = {
            "mean_length": peptide_lengths.mean(),
            "min_length": peptide_lengths.min(),
            "max_length": peptide_lengths.max(),
            "std_length": peptide_lengths.std(),
        }

    # 标签分布
    if "Label" in df:
        label_counts = df["Label"].value_counts()
        report["label_distribution"] = {
            "positive_count": label_counts.get(1, 0),
            "negative_count": label_counts.get(0, 0),
            "positive_ratio": label_counts.get(1, 0) / len(df),
            "class_balance": (
                min(label_counts) / max(label_counts) if len(label_counts) > 1 else 1.0
            ),
        }

    # 质量问题检测
    preprocessor = SequencePreprocessor()

    # 检查TCR序列
    if "TCR" in df:
        for i, seq in enumerate(df["TCR"]):
            if pd.isna(seq) or seq == "":
                report["quality_issues"].append(f"TCR sequence {i+1}: empty")
                continue

            is_valid, issues = preprocessor.validate_sequence(seq)
            if not is_valid:
                report["quality_issues"].append(f"TCR sequence {i+1}: {', '.join(issues)}")

    # 检查肽序列
    if "Peptide" in df:
        for i, seq in enumerate(df["Peptide"]):
            if pd.isna(seq) or seq == "":
                report["quality_issues"].append(f"Peptide sequence {i+1}: empty")
                continue

            is_valid, issues = preprocessor.validate_sequence(seq)
            if not is_valid:
                report["quality_issues"].append(f"Peptide sequence {i+1}: {', '.join(issues)}")

    # 打印报告摘要
    logger.info("Data quality report:")
    logger.info(f"   Total samples: {report['basic_stats']['total_samples']}")

    if "Label" in df:
        pos_ratio = report["label_distribution"]["positive_ratio"]
        balance = report["label_distribution"]["class_balance"]
        logger.info(f"   Positive ratio: {pos_ratio:.1%}")
        logger.info(f"   Class balance: {balance:.3f}")

    if report["quality_issues"]:
        logger.warning(f"   Found quality issues: {len(report['quality_issues'])}")
        for issue in report["quality_issues"][:5]:
            logger.warning(f"     - {issue}")
        if len(report["quality_issues"]) > 5:
            logger.warning(f"     - ... and {len(report['quality_issues'])-5} more issues")
    else:
        logger.info("   No obvious quality issues found")

    return report
