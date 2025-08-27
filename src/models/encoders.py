#!/usr/bin/env python3

import torch
import torch.nn as nn
from transformers import AutoModel, AutoConfig
from peft import (
    LoraConfig,
    AdaLoraConfig,
    VeraConfig,
    BOFTConfig,
    OFTConfig,
    IA3Config,
    PrefixTuningConfig,
    PromptTuningConfig,
    AdaptionPromptConfig,
    get_peft_model,
    TaskType,
    PeftModel,
)
from .adapters import create_adapter
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
            model_name: 模型仓库路径（如 Hugging Face 名称或本地路径）
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
        self.custom_adapter = None
        self.custom_tuning = None
        if peft_config:
            peft_method = peft_config.get("method", "lora").lower()
            if peft_method in ["token_adapter", "pfeiffer_adapter", "houlsby_adapter"]:
                self.custom_adapter = self._setup_custom_adapter(peft_config)
            elif peft_method in ["prefix", "prompt"]:
                self.custom_tuning = self._setup_custom_tuning(peft_method, peft_config)
            else:
                self.peft_model = self._setup_peft(peft_config)

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

        # LoRA 方法
        if peft_method == "lora":
            config = LoraConfig(
                task_type=TaskType.FEATURE_EXTRACTION,
                r=peft_config.get("lora_r", 8),  # 低秩矩阵的秩
                lora_alpha=peft_config.get("lora_alpha", 16),  # LoRA缩放参数
                lora_dropout=peft_config.get("lora_dropout", 0.05),  # LoRA丢弃率
                target_modules=peft_config.get(
                    "target_modules", ["attn.layernorm_qkv.1", "attn.out_proj"]
                ), 
                bias=peft_config.get("bias", "none"), 
            )

        # AdaLoRA方法
        elif peft_method == "adalora":
            total_steps = peft_config.get("total_step", 3500) 
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
                total_step=total_steps,  # 总训练步数，AdaLoRA必需参数
            )

        # VeRA 方法
        elif peft_method == "vera":
            config = VeraConfig(
                task_type=TaskType.FEATURE_EXTRACTION,
                r=peft_config.get("vera_r", 256),  # VeRA秩
                target_modules=peft_config.get(
                    "target_modules", ["attn.layernorm_qkv.1", "attn.out_proj"]
                ),
                vera_dropout=peft_config.get("vera_dropout", 0.0),  # VeRA丢弃率
                d_initial=peft_config.get("d_initial", 0.1),  # 初始lambda值
                projection_prng_key=peft_config.get("projection_prng_key", 0),  # 随机投影种子
            )

        # BOFT 方法
        elif peft_method == "boft":
            config = BOFTConfig(
                task_type=TaskType.FEATURE_EXTRACTION,
                target_modules=peft_config.get(
                    "target_modules", ["attn.layernorm_qkv.1", "attn.out_proj"]
                ),
                boft_block_size=peft_config.get("boft_block_size", 4),  # 块大小
                boft_block_num=peft_config.get("boft_block_num", 0),  # 块数量
                boft_dropout=peft_config.get("boft_dropout", 0.0),  # BOFT 丢弃率
            )


        # OFT 方法
        elif peft_method == "oft":
            config = OFTConfig(
                task_type=TaskType.FEATURE_EXTRACTION,
                target_modules=peft_config.get(
                    "target_modules", ["attn.layernorm_qkv.1", "attn.out_proj"]
                ),
                # 只设置 oft_block_size，不设置 r 参数以避免冲突
                oft_block_size=peft_config.get("oft_block_size", 4),  # OFT块大小
                coft=peft_config.get("coft", False),  # 约束OFT
                eps=peft_config.get("eps", 6e-5),  # 数值稳定性参数
                block_share=peft_config.get("block_share", False),  # 块共享
            )

        # IA3 方法
        elif peft_method == "ia3":
            # IA3需要feedforward_modules是target_modules的子集
            feedforward_modules = peft_config.get("feedforward_modules", ["ffn.3"])
            target_modules = peft_config.get(
                "target_modules", ["attn.layernorm_qkv.1", "attn.out_proj"] + feedforward_modules
            )
            # 确保feedforward_modules是target_modules的子集
            for ff_module in feedforward_modules:
                if ff_module not in target_modules:
                    target_modules.append(ff_module)
            
            config = IA3Config(
                task_type=TaskType.FEATURE_EXTRACTION,
                target_modules=target_modules,
                feedforward_modules=feedforward_modules,
            )

        # 前缀微调（Prefix Tuning）
        elif peft_method == "prefix":
            config = PrefixTuningConfig(
                task_type=TaskType.FEATURE_EXTRACTION,
                num_virtual_tokens=peft_config.get("num_virtual_tokens", 20),
                prefix_projection=peft_config.get("prefix_projection", True),
                encoder_hidden_size=self.config.hidden_size,
            )

        # 提示微调（Prompt Tuning）
        elif peft_method == "prompt":
            config = PromptTuningConfig(
                task_type=TaskType.FEATURE_EXTRACTION,
                num_virtual_tokens=peft_config.get("num_virtual_tokens", 20),
                prompt_tuning_init=peft_config.get("prompt_tuning_init", "RANDOM"),  # RANDOM, TEXT
                prompt_tuning_init_text=peft_config.get("prompt_tuning_init_text", "Classify if TCR binds to peptide"),
                tokenizer_name_or_path=peft_config.get("tokenizer_name_or_path", self.model_name),
            )


        # 自适应提示
        elif peft_method == "adaption_prompt":
            config = AdaptionPromptConfig(
                adapter_len=peft_config.get("adapter_len", 10),  # 适配器长度
                adapter_layers=peft_config.get("adapter_layers", 30),  # 适配器层数
                task_type=TaskType.FEATURE_EXTRACTION,
            )

        # 自定义Adapter方法
        elif peft_method in ["token_adapter", "pfeiffer_adapter", "houlsby_adapter"]:
            return None
            
        else:
            raise ValueError(f"Unsupported PEFT method: {peft_method}")

        try:
            peft_model = get_peft_model(self.base_model, config)
            logger.info(f"{peft_method.upper()} configuration applied successfully")

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

    def _setup_custom_adapter(self, peft_config: Dict[str, Any]) -> nn.Module:
        """
        自定义Adapter方法
        
        参数:
            peft_config: PEFT配置字典
            
        返回:
            配置好的Adapter
        """
        peft_method = peft_config.get("method", "lora").lower()
        logger.info(f"Setting up custom adapter method: {peft_method.upper()}")
        
        try:
            # 获取隐藏层维度
            hidden_size = self.config.hidden_size
            
            # 创建自定义adapter
            adapter = create_adapter(peft_method, peft_config, hidden_size, self.base_model)
            
            # 打印可训练参数信息
            base_total_params = sum(p.numel() for p in self.base_model.parameters())
            base_trainable_params = sum(p.numel() for p in self.base_model.parameters() if p.requires_grad)
            adapter_params = sum(p.numel() for p in adapter.parameters())
            total_params = base_total_params + adapter_params
            trainable_params = base_trainable_params + adapter_params
            
            logger.info(f"{peft_method.upper()} adapter created successfully")
            logger.info(
                f"Parameters: total={total_params:,}, trainable={trainable_params:,} "
                f"({100 * trainable_params / total_params:.2f}%)"
            )
            logger.info(f"Adapter parameters: {adapter_params:,}")
            
            return adapter
            
        except Exception as e:
            logger.error(f"Custom adapter setup failed: {e}")
            raise

    def _setup_custom_tuning(self, tuning_type: str, peft_config: Dict[str, Any]):
        """
        自定义的prefix/prompt tuning
        
        参数:
            tuning_type: tuning类型 ("prefix" 或 "prompt")
            peft_config: PEFT配置
            
        返回:
            CustomTuningWrapper实例
        """
        try:
            from .custom_tuning import CustomTuningWrapper
            
            # 准备配置
            tuning_config = {
                "hidden_size": self.config.hidden_size,
                "num_virtual_tokens": peft_config.get("num_virtual_tokens", 20),
            }
            
            if tuning_type == "prefix":
                tuning_config.update({
                    "prefix_projection": peft_config.get("prefix_projection", True),
                    "prefix_hidden_size": peft_config.get("prefix_hidden_size", None),
                })
            elif tuning_type == "prompt":
                tuning_config.update({
                    "prompt_init": peft_config.get("prompt_tuning_init", "random"),
                    "prompt_init_text": peft_config.get("prompt_tuning_init_text", "Classify if TCR binds to peptide"),
                    "tokenizer": None,  # 待实现：如果需要基于文本初始化
                })
            
            # 创建tuning wrapper
            custom_tuning = CustomTuningWrapper(tuning_type, tuning_config)
            
            logger.info(f"Custom {tuning_type} tuning setup completed")
            
            # 打印参数统计
            trainable_params = sum(p.numel() for p in custom_tuning.parameters() if p.requires_grad)
            total_params = sum(p.numel() for p in self.base_model.parameters())
            
            logger.info(
                f"Parameters: total={total_params:,}, trainable={trainable_params:,} "
                f"({100 * trainable_params / total_params:.2f}%)"
            )
            
            return custom_tuning
            
        except Exception as e:
            logger.error(f"Custom {tuning_type} tuning setup failed: {e}")
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
        if self.custom_adapter is not None:
            if self.custom_tuning is not None:
                # 自定义adapter + tuning
                outputs = self.base_model(input_ids=input_ids, attention_mask=attention_mask, return_dict=return_dict)
                hidden_states = outputs.last_hidden_state if return_dict else outputs[0]
                
                # 应用自定义tuning
                updated_hidden_states, updated_attention_mask = self.custom_tuning(hidden_states, attention_mask)
                
                if return_dict:
                    outputs.last_hidden_state = updated_hidden_states
                    outputs.updated_attention_mask = updated_attention_mask
                else:
                    outputs = (updated_hidden_states,) + outputs[1:]
            elif self.peft_model is not None:
                outputs = self.peft_model(input_ids=input_ids, attention_mask=attention_mask, return_dict=return_dict)
            else:
                outputs = self.base_model(input_ids=input_ids, attention_mask=attention_mask, return_dict=return_dict)

        else:
            if self.custom_tuning is not None:
                outputs = self.base_model(input_ids=input_ids, attention_mask=attention_mask, return_dict=return_dict)
                hidden_states = outputs.last_hidden_state if return_dict else outputs[0]
                
                updated_hidden_states, updated_attention_mask = self.custom_tuning(hidden_states, attention_mask)
                
                if return_dict:
                    outputs.last_hidden_state = updated_hidden_states
                    outputs.updated_attention_mask = updated_attention_mask
                else:
                    outputs = (updated_hidden_states,) + outputs[1:]
                    
            elif self.peft_model is not None:
                outputs = self.peft_model(input_ids=input_ids, attention_mask=attention_mask, return_dict=return_dict)
                
            else:
                outputs = self.base_model(input_ids=input_ids, attention_mask=attention_mask, return_dict=return_dict)
        
        return outputs


class TCREncoder(BaseSequenceEncoder):
    """
    TCR序列编码器
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
