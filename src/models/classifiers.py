#!/usr/bin/env python3


import torch
import torch.nn as nn
from typing import Dict, Optional
import logging

logger = logging.getLogger(__name__)


class BindingClassifier(nn.Module):
    """
    基础结合预测分类器

    将经过交叉注意力融合的TCR和肽表示进行分类预测。

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
            classifier_input_dim = hidden_dim * 2
        else:  # 'add' 或 'multiply'
            classifier_input_dim = hidden_dim

        logger.info(f"Initializing classifier: {pooling_strategy} pooling + {fusion_method} fusion")

        # 分类网络：两层全连接网络 + 丢弃
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
            return embeddings[:, 0, :]

        elif self.pooling_strategy == "mean":
            if attention_mask is not None:
                masked_embeddings = embeddings * attention_mask.unsqueeze(-1)
                pooled = masked_embeddings.sum(dim=1) / attention_mask.sum(dim=1, keepdim=True)
            else:
                pooled = embeddings.mean(dim=1)
            return pooled

        elif self.pooling_strategy == "max":
            if attention_mask is not None:
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
        # 池化序列表示
        tcr_pooled = self.pool_embeddings(tcr_embeddings, tcr_mask)  # [batch_size, hidden_dim]
        peptide_pooled = self.pool_embeddings(
            peptide_embeddings, peptide_mask
        )  # [batch_size, hidden_dim]

        #  融合TCR和肽特征
        if self.fusion_method == "concat":
            # 连接融合：[TCR特征; 肽特征]
            combined = torch.cat([tcr_pooled, peptide_pooled], dim=-1)
        elif self.fusion_method == "add":
            # 相加融合：TCR特征 + 肽特征
            combined = tcr_pooled + peptide_pooled
        elif self.fusion_method == "multiply":
            # 相乘融合：TCR特征 ⊙ 肽特征（哈达玛积）
            combined = tcr_pooled * peptide_pooled
        else:
            raise ValueError(f"Unsupported fusion method: {self.fusion_method}")

        # 分类预测
        logits = self.classifier(combined)
        return logits
