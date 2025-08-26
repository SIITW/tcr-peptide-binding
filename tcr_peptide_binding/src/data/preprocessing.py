#!/usr/bin/env python3
"""
数据预处理模块

提供数据预处理和示例数据生成功能：
1. 序列预处理：清理、验证、标准化
2. 示例数据生成：用于测试和演示
3. 数据质量检查：统计和验证
"""

import pandas as pd
import numpy as np
from typing import List, Tuple, Optional, Dict
import re
import logging

logger = logging.getLogger(__name__)


class SequencePreprocessor:
    """
    序列预处理器

    提供氨基酸序列的清理、验证和标准化功能
    """

    # 标准氨基酸字母
    STANDARD_AA = set("ACDEFGHIKLMNPQRSTVWY")

    # 扩展氨基酸（包括一些变异）
    EXTENDED_AA = STANDARD_AA | set("BJOUXZ")

    def __init__(
        self,
        allow_extended_aa: bool = False,
        remove_invalid_chars: bool = True,
        min_length: int = 3,
        max_length: Optional[int] = None,
    ):
        """
        初始化预处理器

        参数:
            allow_extended_aa: 是否允许扩展氨基酸
            remove_invalid_chars: 是否移除无效字符
            min_length: 序列最小长度
            max_length: 序列最大长度（可选）
        """
        self.allow_extended_aa = allow_extended_aa
        self.remove_invalid_chars = remove_invalid_chars
        self.min_length = min_length
        self.max_length = max_length

        self.valid_chars = self.EXTENDED_AA if allow_extended_aa else self.STANDARD_AA

        logger.info("Sequence preprocessor configuration:")
        logger.info(f"   Allow extended amino acids: {allow_extended_aa}")
        logger.info(f"   Remove invalid characters: {remove_invalid_chars}")
        logger.info(f"   Length range: {min_length} - {max_length or 'unlimited'}")

    def clean_sequence(self, sequence: str) -> str:
        """
        清理单个序列

        参数:
            sequence: 原始序列

        返回:
            清理后的序列
        """
        if not sequence or not isinstance(sequence, str):
            return ""

        # 转换为大写
        seq = sequence.upper().strip()

        # 移除空白字符
        seq = re.sub(r"\s+", "", seq)

        # 移除无效字符
        if self.remove_invalid_chars:
            seq = "".join(char for char in seq if char in self.valid_chars)

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
            sequence_type: 序列类型（用于日志）

        返回:
            (处理后的序列列表, 统计信息)
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
            # 只显示前几个问题（中文注释）
            for issue in stats["issues"][:5]:
                logger.warning(f"     - {issue}")
            if len(stats["issues"]) > 5:
                logger.warning(f"     - ... and {len(stats['issues'])-5} more issues")

        return processed_sequences, stats


def create_sample_data(
    output_path: str,
    num_samples: int = 1000,
    positive_ratio: float = 0.5,
    tcr_length_range: Tuple[int, int] = (15, 30),
    peptide_length_range: Tuple[int, int] = (8, 15),
    random_seed: Optional[int] = 42,
) -> pd.DataFrame:
    """
    创建示例TCR-肽结合数据（中文注释）

    参数:
        output_path: 输出文件路径
        num_samples: 样本数量
        positive_ratio: 正样本比例
        tcr_length_range: TCR长度范围 (最小, 最大)
        peptide_length_range: 肽长度范围 (最小, 最大)
        random_seed: 随机种子

    返回:
        生成的数据DataFrame
    """

    if random_seed is not None:
        np.random.seed(random_seed)

    logger.info("Generating sample data:")
    logger.info(f"   Number of samples: {num_samples}")
    logger.info(f"   Positive ratio: {positive_ratio:.1%}")
    logger.info(f"   TCR length range: {tcr_length_range}")
    logger.info(f"   Peptide length range: {peptide_length_range}")

    # 氨基酸频率（基于真实分布的近似）
    aa_weights = {
        "A": 0.074,
        "C": 0.025,
        "D": 0.054,
        "E": 0.054,
        "F": 0.047,
        "G": 0.074,
        "H": 0.026,
        "I": 0.068,
        "K": 0.058,
        "L": 0.099,
        "M": 0.025,
        "N": 0.045,
        "P": 0.050,
        "Q": 0.039,
        "R": 0.057,
        "S": 0.081,
        "T": 0.062,
        "V": 0.068,
        "W": 0.013,
        "Y": 0.032,
    }

    amino_acids = list(aa_weights.keys())
    weights = list(aa_weights.values())

    def generate_sequence(length_range: Tuple[int, int]) -> str:
        """生成指定长度范围的随机序列"""
        length = np.random.randint(length_range[0], length_range[1] + 1)
        return "".join(np.random.choice(amino_acids, size=length, p=weights))

    # 生成序列
    tcr_sequences = []
    peptide_sequences = []
    labels = []

    # 确定正负样本数量
    num_positive = int(num_samples * positive_ratio)
    num_negative = num_samples - num_positive

    logger.info(f"Generating {num_positive} positive and {num_negative} negative samples...")

    # 生成正样本
    for i in range(num_positive):
        tcr_seq = generate_sequence(tcr_length_range)
        peptide_seq = generate_sequence(peptide_length_range)

        tcr_sequences.append(tcr_seq)
        peptide_sequences.append(peptide_seq)
        labels.append(1)

    # 生成负样本
    for i in range(num_negative):
        tcr_seq = generate_sequence(tcr_length_range)
        peptide_seq = generate_sequence(peptide_length_range)

        tcr_sequences.append(tcr_seq)
        peptide_sequences.append(peptide_seq)
        labels.append(0)

    # 创建DataFrame
    df = pd.DataFrame({"TCR": tcr_sequences, "Peptide": peptide_sequences, "Label": labels})

    # 打乱顺序
    df = df.sample(frac=1, random_state=random_seed).reset_index(drop=True)

    # 添加额外信息
    df["TCR_Length"] = df["TCR"].str.len()
    df["Peptide_Length"] = df["Peptide"].str.len()
    df["Sample_ID"] = [f"Sample_{i+1:04d}" for i in range(len(df))]

    # 保存文件
    df.to_csv(output_path, index=False)

    logger.info("Data generation completed:")
    logger.info(
        f"   TCR length: mean={df['TCR_Length'].mean():.1f}, "
        f"range={df['TCR_Length'].min()}-{df['TCR_Length'].max()}"
    )
    logger.info(
        f"   Peptide length: mean={df['Peptide_Length'].mean():.1f}, "
        f"range={df['Peptide_Length'].min()}-{df['Peptide_Length'].max()}"
    )
    logger.info(f"   Saved to: {output_path}")

    return df


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
