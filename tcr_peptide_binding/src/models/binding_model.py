#!/usr/bin/env python3
"""
TCR-肽结合预测完整模型

组合所有模型组件构建完整的预测流程：
编码器 → Cross Attention → 分类器

模型架构：
1. TCR编码器 + 肽编码器：使用ESM++预训练模型进行序列编码
2. Cross Attention融合：捕获TCR-肽之间的相互作用
3. 结合分类器：预测是否结合

支持标准版和增强版两种配置。
"""

import torch
import torch.nn as nn
from typing import Dict, Any
import logging

from .encoders import TCREncoder, PeptideEncoder
from .attention import CrossAttentionFusion, EnhancedCrossAttentionFusion
from .classifiers import BindingClassifier, EnhancedBindingClassifier

logger = logging.getLogger(__name__)


class TCRPeptideBindingModel(nn.Module):
    """
    TCR-肽结合预测完整模型

    这是整个系统的核心模型，集成了：
    1. 序列编码（ESM++ + PEFT微调）
    2. 跨序列注意力（Cross Attention）
    3. 结合预测（分类器）

    模型支持两种配置：
    - 标准版：简单高效，适合大多数场景
    - 增强版：功能丰富，适合高精度需求
    """

    def __init__(self, config: Dict[str, Any]):
        """
        初始化完整的结合预测模型

        参数:
            config: 完整的配置字典，包含所有组件的配置
        """
        super().__init__()

        self.config = config

        logger.info("Initializing TCR-peptide binding prediction model")

        # 创建编码器
        logger.info("Creating sequence encoders...")
        self.tcr_encoder = TCREncoder(config)
        self.peptide_encoder = PeptideEncoder(config)

        # 确保两个编码器具有相同的隐藏维度
        assert (
            self.tcr_encoder.hidden_dim == self.peptide_encoder.hidden_dim
        ), "TCR and peptide encoders must share the same hidden dimension"
        hidden_dim = self.tcr_encoder.hidden_dim

        # 根据配置选择融合类型
        fusion_config = config.get("fusion", {})
        fusion_type = fusion_config.get("type", "standard")

        logger.info(f"Creating fusion module: {fusion_type}")

        if fusion_type == "enhanced":
            # 增强版Cross Attention
            self.fusion = EnhancedCrossAttentionFusion(
                hidden_dim=hidden_dim,
                num_heads=fusion_config.get("num_heads", 8),
                fusion_strategy=fusion_config.get("strategy", "enhanced_bidirectional"),
                dropout=fusion_config.get("dropout", 0.1),
                use_multi_scale=fusion_config.get("use_multi_scale", True),
                use_position_aware=fusion_config.get("use_position_aware", True),
                use_contrastive=fusion_config.get("use_contrastive", True),
                use_gated_fusion=fusion_config.get("use_gated_fusion", True),
            )
        else:
            # 标准版Cross Attention
            self.fusion = CrossAttentionFusion(
                hidden_dim=hidden_dim,
                num_heads=fusion_config.get("num_heads", 8),
                dropout=fusion_config.get("dropout", 0.1),
                fusion_strategy=fusion_config.get("strategy", "bidirectional"),
            )

        # 根据配置选择分类器类型
        classifier_config = config.get("classifier", {})

        logger.info("Creating classifier...")

        if fusion_type == "enhanced":
            # 增强版分类器
            self.classifier = EnhancedBindingClassifier(
                hidden_dim=hidden_dim,
                num_classes=classifier_config.get("num_classes", 2),
                pooling_strategy=classifier_config.get("pooling_strategy", "cls"),
                fusion_method=classifier_config.get("fusion_method", "adaptive"),
                dropout=classifier_config.get("dropout", 0.1),
            )
        else:
            # 标准版分类器
            self.classifier = BindingClassifier(
                hidden_dim=hidden_dim,
                num_classes=classifier_config.get("num_classes", 2),
                pooling_strategy=classifier_config.get("pooling_strategy", "cls"),
                fusion_method=classifier_config.get("fusion_method", "concat"),
                dropout=classifier_config.get("dropout", 0.1),
            )

        self.fusion_type = fusion_type

        # 统计模型参数
        self._log_model_info()

        logger.info("TCR-peptide binding prediction model initialization completed")

    def _log_model_info(self):
        """记录模型信息（中文注释）"""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)

        logger.info("Model parameter statistics:")
        logger.info(f"   Total parameters: {total_params:,}")
        logger.info(f"   Trainable parameters: {trainable_params:,}")
        logger.info(f"   Trainable ratio: {100 * trainable_params / total_params:.2f}%")
        logger.info(f"   Fusion type: {self.fusion_type}")

    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        模型前向传播

        完整的预测流程：
        1. 序列编码：TCR和肽序列分别通过编码器
        2. Cross Attention：两个序列进行交叉注意力融合
        3. 分类预测：融合后的特征进行结合预测

        参数:
            batch: 批次数据字典，包含：
                - tcr_input_ids: TCR序列的token IDs
                - tcr_attention_mask: TCR注意力掩码
                - peptide_input_ids: 肽序列的token IDs
                - peptide_attention_mask: 肽注意力掩码
                - labels: 真实标签（可选，用于损失计算）

        返回:
            输出字典，包含logits和可选的loss
        """

        # Step 1: 序列编码
        logger.debug("Starting sequence encoding...")

        # TCR序列编码
        tcr_outputs = self.tcr_encoder(
            input_ids=batch["tcr_input_ids"],
            attention_mask=batch["tcr_attention_mask"],
            return_dict=True,
        )

        # 肽序列编码
        peptide_outputs = self.peptide_encoder(
            input_ids=batch["peptide_input_ids"],
            attention_mask=batch["peptide_attention_mask"],
            return_dict=True,
        )

        # 获取最后一层的隐藏状态
        tcr_embeddings = tcr_outputs.last_hidden_state  # [batch, tcr_len, hidden_dim]
        peptide_embeddings = peptide_outputs.last_hidden_state  # [batch, pep_len, hidden_dim]

        logger.debug(f"   TCR embedding shape: {tcr_embeddings.shape}")
        logger.debug(f"   Peptide embedding shape: {peptide_embeddings.shape}")

        # Step 2: Cross Attention融合
        logger.debug("Starting Cross Attention fusion...")

        fusion_kwargs = {
            "tcr_embeddings": tcr_embeddings,
            "peptide_embeddings": peptide_embeddings,
            "tcr_mask": batch["tcr_attention_mask"],
            "peptide_mask": batch["peptide_attention_mask"],
        }

        # 增强版融合需要标签进行对比学习
        if self.fusion_type == "enhanced" and "labels" in batch:
            fusion_kwargs["labels"] = batch["labels"]

        fusion_outputs = self.fusion(**fusion_kwargs)

        logger.debug("   Fusion completed")

        # Step 3: 分类预测
        logger.debug("Starting classification prediction...")

        if self.fusion_type == "enhanced":
            # 增强版分类器使用融合输出
            logits = self.classifier(
                fusion_outputs=fusion_outputs,
                tcr_mask=batch["tcr_attention_mask"],
                peptide_mask=batch["peptide_attention_mask"],
            )
        else:
            # 标准版分类器使用独立的嵌入
            tcr_final = fusion_outputs.get("tcr_fused", tcr_embeddings)
            peptide_final = fusion_outputs.get("peptide_fused", peptide_embeddings)

            logits = self.classifier(
                tcr_embeddings=tcr_final,
                peptide_embeddings=peptide_final,
                tcr_mask=batch["tcr_attention_mask"],
                peptide_mask=batch["peptide_attention_mask"],
            )

        logger.debug(f"   Logits shape: {logits.shape}")

        # 构建输出
        outputs = {"logits": logits}

        # Step 4: 计算损失（如果提供了标签）
        if "labels" in batch:
            logger.debug("Computing loss...")

            # 主分类损失
            loss_fn = nn.CrossEntropyLoss()
            main_loss = loss_fn(logits, batch["labels"])

            total_loss = main_loss

            # 添加对比学习损失（如果有）
            if "contrastive_loss" in fusion_outputs:
                contrastive_weight = 0.1  # 对比损失权重
                contrastive_loss = fusion_outputs["contrastive_loss"]
                total_loss = main_loss + contrastive_weight * contrastive_loss

                outputs["contrastive_loss"] = contrastive_loss
                outputs["main_loss"] = main_loss

                logger.debug(f"   Main loss: {main_loss.item():.4f}")
                logger.debug(f"   Contrastive loss: {contrastive_loss.item():.4f}")

            outputs["loss"] = total_loss
            logger.debug(f"   Total loss: {total_loss.item():.4f}")

        # 添加注意力权重（用于可视化分析）
        for key in fusion_outputs:
            if "weights" in key:
                outputs[key] = fusion_outputs[key]

        return outputs

    def predict(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        预测模式的前向传播（无梯度计算）

        参数:
            batch: 输入批次

        返回:
            预测结果，包含概率和预测类别
        """
        self.eval()
        with torch.no_grad():
            outputs = self.forward(batch)
            logits = outputs["logits"]

            # 计算概率和预测类别
            probs = torch.softmax(logits, dim=1)
            preds = torch.argmax(logits, dim=1)

            return {
                "predictions": preds,
                "probabilities": probs,
                "binding_probability": probs[:, 1],  # 结合概率（中文注释）
                "logits": logits,
            }


# 便捷函数
def create_binding_model(config: Dict[str, Any]) -> TCRPeptideBindingModel:
    """
    创建TCR-肽结合预测模型的便捷函数

    参数:
        config: 模型配置字典

    返回:
        初始化好的模型实例
    """
    return TCRPeptideBindingModel(config)
