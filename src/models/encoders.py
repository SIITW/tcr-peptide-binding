#!/usr/bin/env python3
"""
序列编码器模块

包含TCR和肽序列的编码器实现，支持多种PEFT微调方法：
- LoRA (Low-Rank Adaptation)
- AdaLoRA (Adaptive LoRA)
- VeRA (Vector-based Random Matrix Adaptation)
- BOFT (Butterfly Factorization)
- FourierFT (Fourier Transform based Fine-tuning)
- OFT (Orthogonal Fine-tuning)
- IA3 (Infused Adapter by Inhibiting and Amplifying Inner Activations)
"""

import torch
import torch.nn as nn
from transformers import AutoModel, AutoConfig
from peft import (
    LoraConfig,
    AdaLoraConfig,
    VeraConfig,
    BOFTConfig,
    FourierFTConfig,
    OFTConfig,
    IA3Config,
    PrefixTuningConfig,
    get_peft_model,
    TaskType,
    PeftModel,
)
from typing import Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)


class BaseSequenceEncoder(nn.Module):
    """
    基础序列编码器类

    提供序列编码的通用功能，包括：
    1. ESM++预训练模型加载
    2. PEFT方法应用
    3. 参数冻结管理
    """

    def __init__(
        self,
        model_name: str = "Synthyra/ESMplusplus_large",
        peft_config: Optional[Dict[str, Any]] = None,
        freeze_base_model: bool = True,
    ):
        """
        初始化基础编码器

        参数:
            model_name: 微调模型的hugging face路径
            peft_config: PEFT配置字典
            freeze_base_model: 是否冻结基础模型参数
        """
        super().__init__()

        self.model_name = model_name
        self.peft_config = peft_config

        # 加载预训练ESM++模型
        try:
            logger.info(f"Loading pretrained model: {model_name}")
            self.base_model = AutoModel.from_pretrained(
                model_name, trust_remote_code=True, torch_dtype=torch.float32
            )
            self.config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
            logger.info("Pretrained model loaded successfully")
        except Exception as e:
            logger.error(f"Pretrained model loading failed {model_name}: {e}")
            raise

        # 冻结基础模型参数（只训练PEFT参数）
        if freeze_base_model:
            for param in self.base_model.parameters():
                param.requires_grad = False
            logger.info("Base model parameters frozen, only training PEFT parameters")

        # 应用PEFT方法
        self.peft_model = None
        if peft_config:
            self.peft_model = self._setup_peft(peft_config)

        # 获取隐藏维度供下游任务使用
        self.hidden_dim = self.config.hidden_size

    def _setup_peft(self, peft_config: Dict[str, Any]) -> PeftModel:
        """
        根据配置设置PEFT方法

        参数:
            peft_config: PEFT配置字典

        返回:
            配置好的PEFT模型
        """
        peft_method = peft_config.get("method", "lora").lower()
        logger.info(f"Setting up PEFT method: {peft_method.upper()}")

        # LoRA - 最稳定可靠的方法（推荐）
        if peft_method == "lora":
            config = LoraConfig(
                task_type=TaskType.FEATURE_EXTRACTION,
                r=peft_config.get("lora_r", 8),  # 低秩矩阵的秩
                lora_alpha=peft_config.get("lora_alpha", 16),  # LoRA缩放参数
                lora_dropout=peft_config.get("lora_dropout", 0.05),  # LoRA dropout
                target_modules=peft_config.get(
                    "target_modules", ["attn.layernorm_qkv.1", "attn.out_proj"]
                ),  # 目标模块
                bias=peft_config.get("bias", "none"),  # bias处理方式
            )

        # AdaLoRA - 自适应LoRA，动态调整秩
        elif peft_method == "adalora":
            config = AdaLoraConfig(
                task_type=TaskType.FEATURE_EXTRACTION,
                init_r=peft_config.get("init_r", 12),  # 初始秩
                r=peft_config.get("lora_r", 8),  # 目标秩
                lora_alpha=peft_config.get("lora_alpha", 16),
                lora_dropout=peft_config.get("lora_dropout", 0.05),
                target_modules=peft_config.get(
                    "target_modules", ["attn.layernorm_qkv.1", "attn.out_proj"]
                ),
                # AdaLoRA特有参数
                tinit=peft_config.get("tinit", 0),  # 热身步数
                tfinal=peft_config.get("tfinal", 200),  # 最终调整步数
                deltaT=peft_config.get("deltaT", 10),  # 调整间隔
                beta1=peft_config.get("beta1", 0.85),  # 敏感度参数
                beta2=peft_config.get("beta2", 0.85),  # 不确定性参数
                orth_reg_weight=peft_config.get("orth_reg_weight", 0.5),  # 正交正则化权重
            )

        # VeRA - 向量化随机矩阵适应，内存效率更高
        elif peft_method == "vera":
            config = VeraConfig(
                task_type=TaskType.FEATURE_EXTRACTION,
                r=peft_config.get("vera_r", 256),  # VeRA秩
                target_modules=peft_config.get(
                    "target_modules", ["attn.layernorm_qkv.1", "attn.out_proj"]
                ),
                vera_dropout=peft_config.get("vera_dropout", 0.0),  # VeRA dropout
                d_initial=peft_config.get("d_initial", 0.1),  # 初始lambda值
                projection_prng_key=peft_config.get("projection_prng_key", 0),  # 随机投影种子
            )

        # BOFT - 蝶式因子分解，新兴方法
        elif peft_method == "boft":
            config = BOFTConfig(
                task_type=TaskType.FEATURE_EXTRACTION,
                target_modules=peft_config.get(
                    "target_modules", ["attn.layernorm_qkv.1", "attn.out_proj"]
                ),
                boft_block_size=peft_config.get("boft_block_size", 4),  # 块大小
                boft_block_num=peft_config.get("boft_block_num", 0),  # 块数量
                boft_dropout=peft_config.get("boft_dropout", 0.0),  # BOFT dropout
            )

        # FourierFT - 基于傅里叶变换的微调
        elif peft_method == "fourierft":
            config = FourierFTConfig(
                task_type=TaskType.FEATURE_EXTRACTION,
                target_modules=peft_config.get(
                    "target_modules", ["attn.layernorm_qkv.1", "attn.out_proj"]
                ),
                n_frequency=peft_config.get("n_frequency", 1000),  # 频率数量
                scaling=peft_config.get("scaling", 300.0),  # 缩放因子
            )

        # OFT - 正交微调，保持模型几何性质
        elif peft_method == "oft":
            config = OFTConfig(
                task_type=TaskType.FEATURE_EXTRACTION,
                target_modules=peft_config.get(
                    "target_modules", ["attn.layernorm_qkv.1", "attn.out_proj"]
                ),
                r=peft_config.get("oft_r", 8),  # OFT秩
                oft_dropout=peft_config.get("oft_dropout", 0.0),  # OFT dropout
                coft=peft_config.get("coft", False),  # 约束OFT
                eps=peft_config.get("eps", 6e-5),  # 数值稳定性参数
                block_share=peft_config.get("block_share", False),  # 块共享
            )

        # IA3 - 参数最少的方法，适合极度资源受限场景
        elif peft_method == "ia3":
            config = IA3Config(
                task_type=TaskType.FEATURE_EXTRACTION,
                target_modules=peft_config.get("target_modules", ["k_proj", "v_proj"]),
                feedforward_modules=peft_config.get("feedforward_modules", ["down_proj"]),
            )

        # Prefix Tuning - 前缀微调
        elif peft_method == "prefix":
            config = PrefixTuningConfig(
                task_type=TaskType.FEATURE_EXTRACTION,
                num_virtual_tokens=peft_config.get("num_virtual_tokens", 20),
                prefix_projection=peft_config.get("prefix_projection", True),
                encoder_hidden_size=self.config.hidden_size,
            )

        else:
            raise ValueError(f"Unsupported PEFT method: {peft_method}")

        # 应用PEFT配置
        try:
            peft_model = get_peft_model(self.base_model, config)
            logger.info(f"{peft_method.upper()} configuration applied successfully")

            # 打印可训练参数信息
            total_params = sum(p.numel() for p in peft_model.parameters())
            trainable_params = sum(p.numel() for p in peft_model.parameters() if p.requires_grad)
            logger.info(
                f"Parameters: total={total_params:,}, trainable={trainable_params:,} "
                f"({100 * trainable_params / total_params:.2f}%)"
            )

            return peft_model
        except Exception as e:
            logger.error(f"PEFT setup failed: {e}")
            raise

    def forward(self, input_ids, attention_mask, return_dict=True):
        """
        前向传播

        参数:
            input_ids: 输入序列的token IDs
            attention_mask: 注意力掩码
            return_dict: 是否返回字典格式

        返回:
            模型输出，包含hidden_states等
        """
        model = self.peft_model if self.peft_model else self.base_model
        return model(input_ids=input_ids, attention_mask=attention_mask, return_dict=return_dict)


class TCREncoder(BaseSequenceEncoder):
    """
    TCR序列编码器

    专门用于编码T细胞受体(TCR)序列，继承自BaseSequenceEncoder，
    具有完整的PEFT微调能力。
    """

    def __init__(self, config_dict: Dict[str, Any]):
        """
        初始化TCR编码器

        参数:
            config_dict: 包含模型配置的字典
        """
        tcr_config = config_dict.get("tcr_encoder", {})
        peft_config = (
            config_dict.get("peft", {})
            if config_dict.get("peft", {}).get("enabled", True)
            else None
        )

        super().__init__(
            model_name=tcr_config.get("model_name", "Synthyra/ESMplusplus_large"),
            peft_config=peft_config,
            freeze_base_model=tcr_config.get("freeze_base_model", True),
        )

        logger.info("TCR encoder initialization completed")


class PeptideEncoder(BaseSequenceEncoder):
    """
    肽序列编码器

    专门用于编码肽(Peptide)序列，继承自BaseSequenceEncoder，
    具有完整的PEFT微调能力。
    """

    def __init__(self, config_dict: Dict[str, Any]):
        """
        初始化肽编码器

        参数:
            config_dict: 包含模型配置的字典
        """
        peptide_config = config_dict.get("peptide_encoder", {})
        peft_config = (
            config_dict.get("peft", {})
            if config_dict.get("peft", {}).get("enabled", True)
            else None
        )

        super().__init__(
            model_name=peptide_config.get("model_name", "Synthyra/ESMplusplus_large"),
            peft_config=peft_config,
            freeze_base_model=peptide_config.get("freeze_base_model", True),
        )

        logger.info("Peptide encoder initialization completed")


# 便捷函数，用于创建编码器
def create_tcr_encoder(config: Dict[str, Any]) -> TCREncoder:
    """创建TCR编码器"""
    return TCREncoder(config)


def create_peptide_encoder(config: Dict[str, Any]) -> PeptideEncoder:
    """创建肽编码器"""
    return PeptideEncoder(config)
