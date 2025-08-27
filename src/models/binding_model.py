#!/usr/bin/env python3
import torch
import torch.nn as nn
from typing import Dict, Any
import logging

from .encoders import TCREncoder, PeptideEncoder
from .attention import CrossAttentionFusion
from .classifiers import BindingClassifier

logger = logging.getLogger(__name__)


class TCRPeptideBindingModel(nn.Module):
    """
    TCR-肽结合预测完整模型

    这是整个系统的核心模型，集成了：
    1. 序列编码（ESM++ + PEFT微调）
    2. 跨序列注意力（交叉注意力）
    3. 结合预测（分类器）
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


        self.fusion = CrossAttentionFusion(
            hidden_dim=hidden_dim,
            num_heads=fusion_config.get("num_heads", 8),
            dropout=fusion_config.get("dropout", 0.1),
            fusion_strategy=fusion_config.get("strategy", "bidirectional")
        )
        # 根据配置选择分类器类型
        classifier_config = config.get("classifier", {})

        logger.info("Creating classifier...")

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
        """记录模型信息"""
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
        2. 交叉注意力：两个序列进行交叉注意力融合
        3. 分类预测：融合后的特征进行结合预测

        参数:
            batch: 批次数据字典，包含：
                - tcr_input_ids: TCR序列的token IDs
                - tcr_attention_mask: TCR注意力掩码
                - peptide_input_ids: 肽序列的token IDs
                - peptide_attention_mask: 肽注意力掩码
                - labels: 真实标签

        返回:
            输出字典，包含logits和可选的loss
        """

        # 步骤1: 序列编码
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

        # 步骤2: 跨序列注意力融合
        logger.debug("Starting Cross Attention fusion...")

        tcr_mask = getattr(tcr_outputs, 'updated_attention_mask', batch["tcr_attention_mask"])
        peptide_mask = getattr(peptide_outputs, 'updated_attention_mask', batch["peptide_attention_mask"])

        fusion_kwargs = {
            "tcr_embeddings": tcr_embeddings,
            "peptide_embeddings": peptide_embeddings,
            "tcr_mask": tcr_mask,
            "peptide_mask": peptide_mask,
        }

        if self.fusion_type == "enhanced" and "labels" in batch:
            fusion_kwargs["labels"] = batch["labels"]

        fusion_outputs = self.fusion(**fusion_kwargs)

        logger.debug("   Fusion completed")

        logger.debug("Starting classification prediction...")

        tcr_final = fusion_outputs.get("tcr_fused", tcr_embeddings)
        peptide_final = fusion_outputs.get("peptide_fused", peptide_embeddings)

        logits = self.classifier(
            tcr_embeddings=tcr_final,
            peptide_embeddings=peptide_final,
            tcr_mask=batch["tcr_attention_mask"],
            peptide_mask=batch["peptide_attention_mask"],
        )

        logger.debug(f"   Logits shape: {logits.shape}")

        outputs = {"logits": logits}

        # 步骤4: 计算损失
        if "labels" in batch:
            logger.debug("Computing loss...")
            loss_fn = nn.CrossEntropyLoss()
            main_loss = loss_fn(logits, batch["labels"])

            total_loss = main_loss
            outputs["loss"] = total_loss
            logger.debug(f"   Total loss: {total_loss.item():.4f}")

        # 添加注意力权重
        for key in fusion_outputs:
            if "weights" in key:
                outputs[key] = fusion_outputs[key]

        return outputs

    def predict(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        预测模式的前向传播

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
                "binding_probability": probs[:, 1],  # 结合概率
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
