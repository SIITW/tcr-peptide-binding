#!/usr/bin/env python3
import torch
import torch.nn as nn
from typing import Dict, Optional
import logging

logger = logging.getLogger(__name__)


class CrossAttentionFusion(nn.Module):
    """
    交叉注意力融合模块
    实现TCR和肽序列表示之间的双向交叉注意力机制。


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
        初始化

        参数:
            hidden_dim: 输入嵌入的隐藏层维度
            num_heads: 多头注意力的头数，增加表达能力
            dropout: Dropout比例，防止过拟合
            fusion_strategy: 融合策略
                - 'tcr_to_peptide': 只有TCR关注肽
                - 'peptide_to_tcr': 只有肽关注TCR
                - 'bidirectional': 双向交叉注意力
        """
        super().__init__()

        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.fusion_strategy = fusion_strategy

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
                batch_first=True, 
            )

        # 肽 → TCR 交叉注意力
        if fusion_strategy in ["peptide_to_tcr", "bidirectional"]:
            self.peptide_to_tcr_attn = nn.MultiheadAttention(
                embed_dim=hidden_dim, num_heads=num_heads, dropout=dropout, batch_first=True
            )

        # 层归一化，用于残差连接
        self.layer_norm1 = nn.LayerNorm(hidden_dim)
        self.layer_norm2 = nn.LayerNorm(hidden_dim)

        # 进一步处理注意力输出
        self.ffn1 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4), 
            nn.GELU(),  
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim), 
            nn.Dropout(dropout),
        )

        # 双向注意力，为肽序列单独创建FFN
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
        交叉注意力前向传播

        参数:
            tcr_embeddings: TCR序列嵌入 [batch_size, tcr_seq_len, hidden_dim]
            peptide_embeddings: 肽序列嵌入 [batch_size, pep_seq_len, hidden_dim]
            tcr_mask: TCR注意力掩码 [batch_size, tcr_seq_len]
            peptide_mask: 肽注意力掩码 [batch_size, pep_seq_len]

        返回:
            包含融合嵌入和注意力权重的字典
        """
        outputs = {}

        tcr_key_padding_mask = None
        peptide_key_padding_mask = None

        if tcr_mask is not None:
            tcr_key_padding_mask = ~tcr_mask.bool() 
        if peptide_mask is not None:
            peptide_key_padding_mask = ~peptide_mask.bool()

        # TCR → 肽 交叉注意力
        if self.fusion_strategy in ["tcr_to_peptide", "bidirectional"]:
            tcr_to_pep_output, tcr_to_pep_weights = self.tcr_to_peptide_attn(
                query=tcr_embeddings,  # TCR作为查询
                key=peptide_embeddings,  # 肽作为键
                value=peptide_embeddings,  # 肽作为值
                key_padding_mask=peptide_key_padding_mask,  # 肽的填充掩码
                need_weights=True,  # 返回注意力权重用于可视化
            )

            # 残差连接 + 层归一化
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
                key_padding_mask=tcr_key_padding_mask,  # TCR的填充掩码
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
