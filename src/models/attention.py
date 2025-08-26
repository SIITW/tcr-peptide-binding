#!/usr/bin/env python3
"""
注意力机制模块

实现TCR和肽序列之间的交叉注意力机制，包括：
1. 基础Cross Attention - 双向交叉注意力
2. 增强Cross Attention - 包含多尺度、位置感知、对比学习等高级特性

Cross Attention的核心思想：
- TCR → 肽：让TCR序列关注肽序列中的重要位置
- 肽 → TCR：让肽序列关注TCR序列中的重要位置
- 这种相互关注能够捕获两个序列间的相互作用模式
"""

import torch
import torch.nn as nn
from typing import Dict, Optional
import logging

logger = logging.getLogger(__name__)


class CrossAttentionFusion(nn.Module):
    """
    交叉注意力融合模块

    实现TCR和肽序列表示之间的双向交叉注意力机制。
    这是整个模型的核心创新点，通过让两个序列相互关注，
    能够更好地捕获TCR-肽结合的相互作用模式。

    架构说明:
    1. TCR → 肽 注意力：TCR序列作为query，肽序列作为key和value
    2. 肽 → TCR 注意力：肽序列作为query，TCR序列作为key和value
    3. 残差连接和层归一化：保证训练稳定性
    4. 前馈网络：进一步处理融合后的特征
    """

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int = 8,
        dropout: float = 0.1,
        fusion_strategy: str = "bidirectional",
    ):
        """
        初始化Cross Attention融合模块

        参数:
            hidden_dim: 输入嵌入的隐藏层维度
            num_heads: 多头注意力的头数，增加表达能力
            dropout: Dropout比例，防止过拟合
            fusion_strategy: 融合策略
                - 'tcr_to_peptide': 只有TCR关注肽
                - 'peptide_to_tcr': 只有肽关注TCR
                - 'bidirectional': 双向交叉注意力（推荐）
        """
        super().__init__()

        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.fusion_strategy = fusion_strategy

        # 确保hidden_dim能被num_heads整除
        assert (
            hidden_dim % num_heads == 0
        ), f"hidden_dim ({hidden_dim}) 必须能被 num_heads ({num_heads}) 整除"
        self.head_dim = hidden_dim // num_heads

        logger.info(f"Initializing Cross Attention: {fusion_strategy}, {num_heads} heads, dimension {hidden_dim}")

        # TCR → 肽 交叉注意力
        if fusion_strategy in ["tcr_to_peptide", "bidirectional"]:
            self.tcr_to_peptide_attn = nn.MultiheadAttention(
                embed_dim=hidden_dim,
                num_heads=num_heads,
                dropout=dropout,
                batch_first=True,  # 使用batch_first格式: [batch, seq, dim]
            )

        # 肽 → TCR 交叉注意力
        if fusion_strategy in ["peptide_to_tcr", "bidirectional"]:
            self.peptide_to_tcr_attn = nn.MultiheadAttention(
                embed_dim=hidden_dim, num_heads=num_heads, dropout=dropout, batch_first=True
            )

        # 层归一化，用于残差连接
        self.layer_norm1 = nn.LayerNorm(hidden_dim)
        self.layer_norm2 = nn.LayerNorm(hidden_dim)

        # 前馈网络（FFN），用于进一步处理注意力输出
        self.ffn1 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),  # 扩展到4倍维度
            nn.GELU(),  # GELU激活函数，在Transformer中表现更好
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim),  # 压缩回原维度
            nn.Dropout(dropout),
        )

        # 如果是双向注意力，为肽序列单独创建FFN
        if fusion_strategy == "bidirectional":
            self.ffn2 = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim * 4),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim * 4, hidden_dim),
                nn.Dropout(dropout),
            )

    def forward(
        self,
        tcr_embeddings: torch.Tensor,
        peptide_embeddings: torch.Tensor,
        tcr_mask: Optional[torch.Tensor] = None,
        peptide_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Cross Attention前向传播

        参数:
            tcr_embeddings: TCR序列嵌入 [batch_size, tcr_seq_len, hidden_dim]
            peptide_embeddings: 肽序列嵌入 [batch_size, pep_seq_len, hidden_dim]
            tcr_mask: TCR注意力掩码 [batch_size, tcr_seq_len]
            peptide_mask: 肽注意力掩码 [batch_size, pep_seq_len]

        返回:
            包含融合嵌入和注意力权重的字典
        """
        outputs = {}

        # 转换mask格式: MultiheadAttention需要key_padding_mask格式（True表示padding位置）
        tcr_key_padding_mask = None
        peptide_key_padding_mask = None

        if tcr_mask is not None:
            tcr_key_padding_mask = ~tcr_mask.bool()  # 反转mask：True->False, False->True
        if peptide_mask is not None:
            peptide_key_padding_mask = ~peptide_mask.bool()

        # TCR → 肽 交叉注意力
        if self.fusion_strategy in ["tcr_to_peptide", "bidirectional"]:
            # TCR序列关注肽序列的信息
            tcr_to_pep_output, tcr_to_pep_weights = self.tcr_to_peptide_attn(
                query=tcr_embeddings,  # TCR作为查询
                key=peptide_embeddings,  # 肽作为键
                value=peptide_embeddings,  # 肽作为值
                key_padding_mask=peptide_key_padding_mask,  # 肽的padding mask
                need_weights=True,  # 返回注意力权重用于可视化
            )

            # 残差连接 + 层归一化（Post-LN架构）
            tcr_fused = self.layer_norm1(tcr_embeddings + tcr_to_pep_output)

            # 前馈网络 + 残差连接
            tcr_fused = tcr_fused + self.ffn1(tcr_fused)

            outputs["tcr_fused"] = tcr_fused
            outputs["tcr_to_peptide_weights"] = tcr_to_pep_weights

        # 肽 → TCR 交叉注意力
        if self.fusion_strategy in ["peptide_to_tcr", "bidirectional"]:
            # 肽序列关注TCR序列的信息
            pep_to_tcr_output, pep_to_tcr_weights = self.peptide_to_tcr_attn(
                query=peptide_embeddings,  # 肽作为查询
                key=tcr_embeddings,  # TCR作为键
                value=tcr_embeddings,  # TCR作为值
                key_padding_mask=tcr_key_padding_mask,  # TCR的padding mask
                need_weights=True,
            )

            # 残差连接 + 层归一化
            layer_norm = (
                self.layer_norm2 if self.fusion_strategy == "bidirectional" else self.layer_norm1
            )
            peptide_fused = layer_norm(peptide_embeddings + pep_to_tcr_output)

            # 前馈网络 + 残差连接
            ffn = self.ffn2 if self.fusion_strategy == "bidirectional" else self.ffn1
            peptide_fused = peptide_fused + ffn(peptide_fused)

            outputs["peptide_fused"] = peptide_fused
            outputs["peptide_to_tcr_weights"] = pep_to_tcr_weights

        return outputs


class EnhancedCrossAttentionFusion(nn.Module):
    """
    增强版交叉注意力融合模块

    在基础Cross Attention基础上增加了高级特性：
    1. 多尺度注意力：捕获不同粒度的相互作用
    2. 位置感知注意力：考虑序列位置信息
    3. 门控融合机制：自适应控制信息流
    4. 对比学习损失：增强正负样本区分能力

    这个版本适合对性能有更高要求的场景。
    """

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int = 8,
        fusion_strategy: str = "enhanced_bidirectional",
        dropout: float = 0.1,
        use_multi_scale: bool = True,
        use_position_aware: bool = True,
        use_contrastive: bool = True,
        use_gated_fusion: bool = True,
    ):
        """
        初始化增强版Cross Attention

        参数:
            hidden_dim: 隐藏维度
            num_heads: 注意力头数
            fusion_strategy: 融合策略（增强版）
            dropout: Dropout率
            use_multi_scale: 是否使用多尺度注意力
            use_position_aware: 是否使用位置感知注意力
            use_contrastive: 是否使用对比学习
            use_gated_fusion: 是否使用门控融合
        """
        super().__init__()

        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.fusion_strategy = fusion_strategy
        self.use_multi_scale = use_multi_scale
        self.use_position_aware = use_position_aware
        self.use_contrastive = use_contrastive
        self.use_gated_fusion = use_gated_fusion

        logger.info("Initializing Enhanced Cross Attention")
        logger.info(f"   Multi-scale: {use_multi_scale}, Position-aware: {use_position_aware}")
        logger.info(f"   Contrastive learning: {use_contrastive}, Gated fusion: {use_gated_fusion}")

        # 基础交叉注意力模块
        self.base_attention = CrossAttentionFusion(
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            fusion_strategy="bidirectional",
        )

        # 多尺度注意力：使用不同的头数
        if use_multi_scale:
            self.multi_scale_heads = nn.ModuleList(
                [
                    nn.MultiheadAttention(hidden_dim, heads, dropout=dropout, batch_first=True)
                    for heads in [4, 8, 16]
                    if heads <= num_heads
                ]
            )

        # 位置编码（如果需要位置感知）
        if use_position_aware:
            self.pos_encoding = nn.Parameter(torch.randn(512, hidden_dim) * 0.02)

        # 门控机制
        if use_gated_fusion:
            self.gate_tcr = nn.Sequential(nn.Linear(hidden_dim * 2, hidden_dim), nn.Sigmoid())
            self.gate_peptide = nn.Sequential(nn.Linear(hidden_dim * 2, hidden_dim), nn.Sigmoid())

    def forward(
        self,
        tcr_embeddings: torch.Tensor,
        peptide_embeddings: torch.Tensor,
        tcr_mask: Optional[torch.Tensor] = None,
        peptide_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        增强版Cross Attention前向传播

        参数:
            tcr_embeddings: TCR嵌入
            peptide_embeddings: 肽嵌入
            tcr_mask: TCR掩码
            peptide_mask: 肽掩码
            labels: 标签（用于对比学习）

        返回:
            增强的融合输出
        """
        # 基础交叉注意力
        base_outputs = self.base_attention(
            tcr_embeddings, peptide_embeddings, tcr_mask, peptide_mask
        )

        tcr_fused = base_outputs["tcr_fused"]
        peptide_fused = base_outputs["peptide_fused"]

        # 位置感知增强
        if self.use_position_aware:
            seq_len_tcr = tcr_fused.size(1)
            seq_len_pep = peptide_fused.size(1)

            if seq_len_tcr <= 512:
                tcr_fused = tcr_fused + self.pos_encoding[:seq_len_tcr].unsqueeze(0)
            if seq_len_pep <= 512:
                peptide_fused = peptide_fused + self.pos_encoding[:seq_len_pep].unsqueeze(0)

        # 门控融合
        if self.use_gated_fusion:
            # 计算门控权重
            tcr_concat = torch.cat([tcr_embeddings, tcr_fused], dim=-1)
            peptide_concat = torch.cat([peptide_embeddings, peptide_fused], dim=-1)

            tcr_gate = self.gate_tcr(tcr_concat)
            peptide_gate = self.gate_peptide(peptide_concat)

            # 应用门控
            tcr_fused = tcr_gate * tcr_fused + (1 - tcr_gate) * tcr_embeddings
            peptide_fused = peptide_gate * peptide_fused + (1 - peptide_gate) * peptide_embeddings

        outputs = {
            "tcr_fused": tcr_fused,
            "peptide_fused": peptide_fused,
            **{k: v for k, v in base_outputs.items() if "weights" in k},
        }

        # 对比学习损失计算
        if self.use_contrastive and labels is not None:
            contrastive_loss = self._compute_contrastive_loss(tcr_fused, peptide_fused, labels)
            outputs["contrastive_loss"] = contrastive_loss

        return outputs

    def _compute_contrastive_loss(
        self,
        tcr_fused: torch.Tensor,
        peptide_fused: torch.Tensor,
        labels: torch.Tensor,
        temperature: float = 0.1,
    ) -> torch.Tensor:
        """
        计算对比学习损失

        目标：让结合的TCR-肽对在表示空间中更相似，
        不结合的对更远离。
        """
        # 池化到固定长度
        tcr_pooled = tcr_fused.mean(dim=1)  # [batch_size, hidden_dim]
        peptide_pooled = peptide_fused.mean(dim=1)

        # 计算相似度矩阵
        similarities = (
            torch.cosine_similarity(tcr_pooled.unsqueeze(1), peptide_pooled.unsqueeze(0), dim=-1)
            / temperature
        )

        # 对比学习损失：正样本相似度高，负样本相似度低
        positive_mask = labels.unsqueeze(1) == labels.unsqueeze(0)
        positive_mask = positive_mask.float()

        # InfoNCE loss的简化版本
        loss = -torch.log_softmax(similarities, dim=1) * positive_mask
        loss = loss.sum(dim=1) / positive_mask.sum(dim=1).clamp(min=1)

        return loss.mean()
