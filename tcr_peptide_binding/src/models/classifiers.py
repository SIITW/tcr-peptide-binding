#!/usr/bin/env python3
"""
分类器模块

包含用于TCR-肽结合预测的分类器实现：
1. 基础分类器 - 简单有效的分类头
2. 增强分类器 - 包含自适应融合等高级特性

主要功能：
- 序列表示池化（CLS token、平均池化、最大池化）
- 特征融合（连接、相加、相乘、自适应）
- 二分类预测（结合/不结合）
"""

import torch
import torch.nn as nn
from typing import Dict, Optional
import logging

logger = logging.getLogger(__name__)


class BindingClassifier(nn.Module):
    """
    基础结合预测分类器

    将经过Cross Attention融合的TCR和肽表示进行分类预测。

    工作流程：
    1. 序列池化：将变长序列转换为固定长度表示
    2. 特征融合：组合TCR和肽的表示
    3. 分类预测：输出结合概率
    """

    def __init__(
        self,
        hidden_dim: int,
        num_classes: int = 2,
        pooling_strategy: str = "cls",
        fusion_method: str = "concat",
        dropout: float = 0.1,
    ):
        """
        初始化分类器

        参数:
            hidden_dim: 输入特征的隐藏维度
            num_classes: 分类类别数（默认2：结合/不结合）
            pooling_strategy: 序列池化策略
                - 'cls': 使用[CLS] token（第一个位置）
                - 'mean': 平均池化
                - 'max': 最大池化
            fusion_method: 特征融合方法
                - 'concat': 连接TCR和肽特征
                - 'add': 相加融合
                - 'multiply': 相乘融合
            dropout: Dropout率
        """
        super().__init__()

        self.hidden_dim = hidden_dim
        self.pooling_strategy = pooling_strategy
        self.fusion_method = fusion_method

        # 根据融合方法确定分类器输入维度
        if fusion_method == "concat":
            classifier_input_dim = hidden_dim * 2  # TCR + 肽
        else:  # 'add' 或 'multiply'
            classifier_input_dim = hidden_dim

        logger.info(f"Initializing classifier: {pooling_strategy} pooling + {fusion_method} fusion")

        # 分类网络：两层全连接网络 + Dropout
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(classifier_input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes),
        )

    def pool_embeddings(
        self, embeddings: torch.Tensor, attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        将变长序列池化为固定长度表示

        参数:
            embeddings: 序列嵌入 [batch_size, seq_len, hidden_dim]
            attention_mask: 注意力掩码 [batch_size, seq_len]

        返回:
            池化后的表示 [batch_size, hidden_dim]
        """

        if self.pooling_strategy == "cls":
            # 使用[CLS] token（第一个位置的表示）
            # 这是BERT等预训练模型常用的做法
            return embeddings[:, 0, :]

        elif self.pooling_strategy == "mean":
            # 平均池化：计算有效位置的平均值
            if attention_mask is not None:
                # 只对非padding位置进行平均
                masked_embeddings = embeddings * attention_mask.unsqueeze(-1)
                pooled = masked_embeddings.sum(dim=1) / attention_mask.sum(dim=1, keepdim=True)
            else:
                # 简单平均
                pooled = embeddings.mean(dim=1)
            return pooled

        elif self.pooling_strategy == "max":
            # 最大池化：取每个维度的最大值
            if attention_mask is not None:
                # 将padding位置设为很小的值
                masked_embeddings = embeddings.clone()
                masked_embeddings[~attention_mask.bool()] = float("-inf")
                pooled, _ = masked_embeddings.max(dim=1)
            else:
                pooled, _ = embeddings.max(dim=1)
            return pooled

        else:
            raise ValueError(f"Unsupported pooling strategy: {self.pooling_strategy}")

    def forward(
        self,
        tcr_embeddings: torch.Tensor,
        peptide_embeddings: torch.Tensor,
        tcr_mask: Optional[torch.Tensor] = None,
        peptide_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        分类器前向传播

        参数:
            tcr_embeddings: TCR表示 [batch_size, tcr_len, hidden_dim]
            peptide_embeddings: 肽表示 [batch_size, pep_len, hidden_dim]
            tcr_mask: TCR掩码 [batch_size, tcr_len]
            peptide_mask: 肽掩码 [batch_size, pep_len]

        返回:
            分类logits [batch_size, num_classes]
        """
        # Step 1: 池化序列表示
        tcr_pooled = self.pool_embeddings(tcr_embeddings, tcr_mask)  # [batch_size, hidden_dim]
        peptide_pooled = self.pool_embeddings(
            peptide_embeddings, peptide_mask
        )  # [batch_size, hidden_dim]

        # Step 2: 融合TCR和肽特征
        if self.fusion_method == "concat":
            # 连接融合：[TCR特征; 肽特征]
            combined = torch.cat([tcr_pooled, peptide_pooled], dim=-1)
        elif self.fusion_method == "add":
            # 相加融合：TCR特征 + 肽特征
            combined = tcr_pooled + peptide_pooled
        elif self.fusion_method == "multiply":
            # 相乘融合：TCR特征 ⊙ 肽特征（Hadamard product）
            combined = tcr_pooled * peptide_pooled
        else:
            raise ValueError(f"Unsupported fusion method: {self.fusion_method}")

        # Step 3: 分类预测
        logits = self.classifier(combined)
        return logits


class EnhancedBindingClassifier(nn.Module):
    """
    增强版结合预测分类器

    在基础分类器基础上增加了高级特性：
    1. 自适应融合：自动学习最优的融合权重
    2. 多层感知机：更深的网络结构
    3. 批归一化：提高训练稳定性
    4. 注意力池化：更智能的序列池化方法
    """

    def __init__(
        self,
        hidden_dim: int,
        num_classes: int = 2,
        pooling_strategy: str = "cls",
        fusion_method: str = "adaptive",
        dropout: float = 0.1,
    ):
        """
        初始化增强版分类器

        参数:
            hidden_dim: 隐藏维度
            num_classes: 分类类别数
            pooling_strategy: 池化策略
            fusion_method: 融合方法（包括'adaptive'自适应融合）
            dropout: Dropout率
        """
        super().__init__()

        self.hidden_dim = hidden_dim
        self.pooling_strategy = pooling_strategy
        self.fusion_method = fusion_method

        logger.info(f"Initializing enhanced classifier: {fusion_method} fusion")

        # 自适应融合网络
        if fusion_method == "adaptive":
            self.fusion_net = nn.Sequential(
                nn.Linear(hidden_dim * 2, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, 3),  # 输出3个权重：TCR、肽、交互
                nn.Softmax(dim=-1),
            )
            classifier_input_dim = hidden_dim
        elif fusion_method == "concat":
            classifier_input_dim = hidden_dim * 2
        else:
            classifier_input_dim = hidden_dim

        # 注意力池化网络（如果需要）
        if pooling_strategy == "attention":
            self.attention_pooling = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2), nn.Tanh(), nn.Linear(hidden_dim // 2, 1)
            )

        # 增强的分类网络：更深、批归一化
        self.classifier = nn.Sequential(
            nn.Linear(classifier_input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.BatchNorm1d(hidden_dim // 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 4, num_classes),
        )

    def attention_pool_embeddings(
        self, embeddings: torch.Tensor, attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        注意力机制池化

        学习每个位置的重要性权重，进行加权平均
        """
        # 计算注意力权重
        attn_weights = self.attention_pooling(embeddings)  # [batch, seq_len, 1]

        if attention_mask is not None:
            # 将padding位置的权重设为很小的值
            attn_weights = attn_weights.masked_fill(~attention_mask.unsqueeze(-1), float("-inf"))

        # 软最大化得到归一化权重
        attn_weights = torch.softmax(attn_weights, dim=1)

        # 加权求和
        pooled = torch.sum(embeddings * attn_weights, dim=1)
        return pooled

    def pool_embeddings(
        self, embeddings: torch.Tensor, attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """增强版池化，支持注意力池化"""

        if self.pooling_strategy == "attention":
            return self.attention_pool_embeddings(embeddings, attention_mask)
        else:
            # 使用基础池化方法
            base_classifier = BindingClassifier(
                self.hidden_dim, pooling_strategy=self.pooling_strategy
            )
            return base_classifier.pool_embeddings(embeddings, attention_mask)

    def forward(
        self,
        fusion_outputs: Optional[Dict[str, torch.Tensor]] = None,
        tcr_embeddings: Optional[torch.Tensor] = None,
        peptide_embeddings: Optional[torch.Tensor] = None,
        tcr_mask: Optional[torch.Tensor] = None,
        peptide_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        增强版分类器前向传播

        参数:
            fusion_outputs: 来自增强版Cross Attention的输出
            tcr_embeddings: TCR表示（备用）
            peptide_embeddings: 肽表示（备用）
            tcr_mask: TCR掩码
            peptide_mask: 肽掩码

        返回:
            分类logits
        """
        # 获取融合后的表示
        if fusion_outputs is not None:
            tcr_fused = fusion_outputs.get("tcr_fused", tcr_embeddings)
            peptide_fused = fusion_outputs.get("peptide_fused", peptide_embeddings)
        else:
            tcr_fused = tcr_embeddings
            peptide_fused = peptide_embeddings

        # 池化
        tcr_pooled = self.pool_embeddings(tcr_fused, tcr_mask)
        peptide_pooled = self.pool_embeddings(peptide_fused, peptide_mask)

        # 特征融合
        if self.fusion_method == "adaptive":
            # 自适应融合：学习最优组合权重
            concat_features = torch.cat([tcr_pooled, peptide_pooled], dim=-1)
            fusion_weights = self.fusion_net(concat_features)  # [batch, 3]

            # 组合特征：w1*TCR + w2*肽 + w3*(TCR⊙肽)
            combined = (
                fusion_weights[:, 0:1] * tcr_pooled
                + fusion_weights[:, 1:2] * peptide_pooled
                + fusion_weights[:, 2:3] * (tcr_pooled * peptide_pooled)
            )

        elif self.fusion_method == "concat":
            combined = torch.cat([tcr_pooled, peptide_pooled], dim=-1)
        elif self.fusion_method == "add":
            combined = tcr_pooled + peptide_pooled
        elif self.fusion_method == "multiply":
            combined = tcr_pooled * peptide_pooled
        else:
            raise ValueError(f"不支持的融合方法: {self.fusion_method}")

        # 分类预测
        logits = self.classifier(combined)
        return logits
