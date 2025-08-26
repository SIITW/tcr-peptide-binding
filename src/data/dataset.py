#!/usr/bin/env python3
"""
TCR-肽数据集处理模块

处理TCR-肽结合预测的数据集，包括：
1. 数据集类：处理序列对和标签
2. 数据加载器创建：训练/验证/测试数据流
3. 批处理：动态padding和掩码生成

数据格式：
- 输入：TCR序列 + 肽序列
- 输出：结合标签（0/1）
"""

import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from typing import List, Tuple, Dict, Optional
import numpy as np
from sklearn.model_selection import train_test_split
import logging

from ..utils.tokenizer_manager import get_tokenizer

logger = logging.getLogger(__name__)


class TCRPeptideDataset(Dataset):
    """
    TCR-肽结合预测数据集

    处理TCR和肽序列对，进行分词、编码和批处理。
    支持动态padding和注意力掩码生成。
    """

    def __init__(
        self,
        tcr_sequences: List[str],
        peptide_sequences: List[str],
        labels: List[int],
        tokenizer_name: str = "Synthyra/ESMplusplus_large",
        max_tcr_length: int = 128,
        max_peptide_length: int = 64,
    ):
        """
        初始化数据集

        参数:
            tcr_sequences: TCR氨基酸序列列表
            peptide_sequences: 肽氨基酸序列列表
            labels: 结合标签列表 (0: 不结合, 1: 结合)
            tokenizer_name: ESM++分词器名称
            max_tcr_length: TCR序列最大长度
            max_peptide_length: 肽序列最大长度
        """
        self.tcr_sequences = tcr_sequences
        self.peptide_sequences = peptide_sequences
        self.labels = labels
        self.max_tcr_length = max_tcr_length
        self.max_peptide_length = max_peptide_length

        # 验证输入数据长度一致
        assert (
            len(tcr_sequences) == len(peptide_sequences) == len(labels)
        ), (
            f"Mismatched lengths: TCR={len(tcr_sequences)}, "
            f"Peptide={len(peptide_sequences)}, Labels={len(labels)}"
        )

        logger.info("Dataset statistics:")
        logger.info(f"   Total samples: {len(self)}")
        logger.info(f"   Positive: {sum(labels)} ({100*sum(labels)/len(labels):.1f}%)")
        logger.info(
            f"   Negative: {len(labels)-sum(labels)} "
            f"({100*(len(labels)-sum(labels))/len(labels):.1f}%)"
        )

        # 使用分词器管理器获取分词器，避免重复加载ESM++模型
        try:
            logger.info(f"Getting tokenizer: {tokenizer_name}")
            self.tokenizer = get_tokenizer(tokenizer_name)
            logger.info("Tokenizer obtained successfully")
        except Exception as e:
            logger.error(f"Tokenizer acquisition failed: {e}")
            raise

        # 数据质量检查
        self._validate_sequences()

    def _validate_sequences(self):
        """验证序列质量（中文注释）"""
        logger.info("Validating sequence quality...")

        # 检查序列长度分布
        tcr_lengths = [len(seq) for seq in self.tcr_sequences]
        peptide_lengths = [len(seq) for seq in self.peptide_sequences]

        logger.info(
            f"   TCR length: mean={np.mean(tcr_lengths):.1f}, "
            f"min={min(tcr_lengths)}, max={max(tcr_lengths)}"
        )
        logger.info(
            f"   Peptide length: mean={np.mean(peptide_lengths):.1f}, "
            f"min={min(peptide_lengths)}, max={max(peptide_lengths)}"
        )

        # 检查过长序列
        long_tcr = sum(1 for length in tcr_lengths if length > self.max_tcr_length)
        long_peptide = sum(1 for length in peptide_lengths if length > self.max_peptide_length)

        if long_tcr > 0:
            logger.warning(f"{long_tcr} TCR sequences exceed max length {self.max_tcr_length}, will be truncated")
        if long_peptide > 0:
            logger.warning(
                f"{long_peptide} peptide sequences exceed max length {self.max_peptide_length}, will be truncated"
            )

        # 检查非标准氨基酸
        standard_aa = set("ACDEFGHIKLMNPQRSTVWY")
        non_standard_count = 0

        for seq in self.tcr_sequences + self.peptide_sequences:
            if not set(seq.upper()).issubset(standard_aa):
                non_standard_count += 1

        if non_standard_count > 0:
            logger.warning(f"{non_standard_count} sequences contain non-standard amino acids")

    def __len__(self):
        return len(self.tcr_sequences)

    def __getitem__(self, idx) -> Dict[str, torch.Tensor]:
        """
        获取单个样本

        返回:
            包含分词结果和标签的字典
        """
        tcr_seq = self.tcr_sequences[idx]
        peptide_seq = self.peptide_sequences[idx]
        label = self.labels[idx]

        # TCR序列分词
        tcr_tokens = self.tokenizer(
            tcr_seq,
            max_length=self.max_tcr_length,
            padding="max_length",  # 填充到最大长度
            truncation=True,  # 截断超长序列
            return_tensors="pt",  # 返回PyTorch张量
        )

        # 肽序列分词
        peptide_tokens = self.tokenizer(
            peptide_seq,
            max_length=self.max_peptide_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        return {
            # TCR相关
            "tcr_input_ids": tcr_tokens["input_ids"].squeeze(0),  # [tcr_max_len]
            "tcr_attention_mask": tcr_tokens["attention_mask"].squeeze(0),  # [tcr_max_len]
            # 肽相关
            "peptide_input_ids": peptide_tokens["input_ids"].squeeze(0),  # [pep_max_len]
            "peptide_attention_mask": peptide_tokens["attention_mask"].squeeze(0),  # [pep_max_len]
            # 标签
            "labels": torch.tensor(label, dtype=torch.long),  # []
            # 元信息（可选，用于调试）
            "tcr_sequence": tcr_seq,
            "peptide_sequence": peptide_seq,
            "sample_idx": idx,
        }


def create_dataloaders(
    train_df: pd.DataFrame,
    val_df: Optional[pd.DataFrame],
    test_df: pd.DataFrame,
    tokenizer_name: str = "Synthyra/ESMplusplus_large",
    batch_size: int = 8,
    max_tcr_length: int = 128,
    max_peptide_length: int = 64,
    num_workers: int = 4,
    shuffle_train: bool = True,
) -> Tuple[DataLoader, Optional[DataLoader], DataLoader]:
    """
    创建训练、验证、测试数据加载器

    参数:
        train_df: 训练数据DataFrame
        val_df: 验证数据DataFrame（可选）
        test_df: 测试数据DataFrame
        tokenizer_name: 分词器名称
        batch_size: 批次大小
        max_tcr_length: TCR最大长度
        max_peptide_length: 肽最大长度
        num_workers: 数据加载工作进程数
        shuffle_train: 是否打乱训练数据

    返回:
        (训练加载器, 验证加载器, 测试加载器)
    """

    logger.info("Creating data loaders...")

    # 创建训练数据集
    train_dataset = TCRPeptideDataset(
        tcr_sequences=train_df["TCR"].tolist(),
        peptide_sequences=train_df["Peptide"].tolist(),
        labels=train_df["Label"].tolist(),
        tokenizer_name=tokenizer_name,
        max_tcr_length=max_tcr_length,
        max_peptide_length=max_peptide_length,
    )

    # 创建验证数据集（如果提供）
    val_dataset = None
    if val_df is not None and len(val_df) > 0:
        val_dataset = TCRPeptideDataset(
            tcr_sequences=val_df["TCR"].tolist(),
            peptide_sequences=val_df["Peptide"].tolist(),
            labels=val_df["Label"].tolist(),
            tokenizer_name=tokenizer_name,
            max_tcr_length=max_tcr_length,
            max_peptide_length=max_peptide_length,
        )

    # 创建测试数据集
    test_dataset = TCRPeptideDataset(
        tcr_sequences=test_df["TCR"].tolist(),
        peptide_sequences=test_df["Peptide"].tolist(),
        labels=test_df["Label"].tolist(),
        tokenizer_name=tokenizer_name,
        max_tcr_length=max_tcr_length,
        max_peptide_length=max_peptide_length,
    )

    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=shuffle_train,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),  # 如果有GPU则使用pin_memory加速
        drop_last=False,  # 保留最后一个不完整的batch
    )

    val_loader = None
    if val_dataset is not None:
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,  # 验证集不需要打乱
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
            drop_last=False,
        )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,  # 测试集不需要打乱
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )

    logger.info("Data loader creation completed:")
    logger.info(f"   Train: {len(train_loader)} batches ({len(train_dataset)} samples)")
    if val_loader:
        logger.info(f"   Val: {len(val_loader)} batches ({len(val_dataset)} samples)")
    logger.info(f"   Test: {len(test_loader)} batches ({len(test_dataset)} samples)")
    logger.info(f"   Batch size: {batch_size}")

    return train_loader, val_loader, test_loader


def prepare_data_splits(
    data_path: str,
    test_size: float = 0.2,
    val_size: float = 0.1,
    random_state: int = 42,
    stratify: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    准备数据分割

    参数:
        data_path: 数据文件路径
        test_size: 测试集比例
        val_size: 验证集比例
        random_state: 随机种子
        stratify: 是否按标签分层采样

    返回:
        (训练集, 验证集, 测试集) DataFrame
    """

    logger.info(f"Loading data: {data_path}")

    # 读取数据
    df = pd.read_csv(data_path)

    # 验证数据格式
    required_columns = ["TCR", "Peptide", "Label"]
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(f"数据文件缺少必需的列: {missing_columns}")

    logger.info("Original data statistics:")
    logger.info(f"   Total samples: {len(df)}")
    logger.info(f"   Positive: {(df['Label']==1).sum()} ({100*(df['Label']==1).mean():.1f}%)")
    logger.info(f"   Negative: {(df['Label']==0).sum()} ({100*(df['Label']==0).mean():.1f}%)")

    # 数据清理
    df = df.dropna(subset=["TCR", "Peptide", "Label"])  # 删除缺失值
    df = df[df["TCR"].str.len() > 0]  # 删除空序列
    df = df[df["Peptide"].str.len() > 0]

    logger.info(f"   Samples after cleaning: {len(df)}")

    # 分割数据
    stratify_column = df["Label"] if stratify else None

    # 首先分出测试集
    train_val_df, test_df = train_test_split(
        df, test_size=test_size, random_state=random_state, stratify=stratify_column
    )

    # 从训练+验证集中分出验证集
    if val_size > 0:
        val_ratio = val_size / (1 - test_size)  # 调整验证集比例
        stratify_train_val = train_val_df["Label"] if stratify else None

        train_df, val_df = train_test_split(
            train_val_df,
            test_size=val_ratio,
            random_state=random_state,
            stratify=stratify_train_val,
        )
    else:
        train_df = train_val_df
        val_df = pd.DataFrame()  # 空DataFrame

    logger.info("Data split results:")
    logger.info(f"   Train: {len(train_df)} samples ({100*len(train_df)/len(df):.1f}%)")
    if len(val_df) > 0:
        logger.info(f"   Val: {len(val_df)} samples ({100*len(val_df)/len(df):.1f}%)")
    logger.info(f"   Test: {len(test_df)} samples ({100*len(test_df)/len(df):.1f}%)")

    return train_df, val_df, test_df
