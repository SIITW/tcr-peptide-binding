#!/usr/bin/env python3

import torch
import torch.nn as nn
from typing import Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class CustomPrefixTuning(nn.Module):
    """
    Prefix Tuning实现
    在输入序列前面添加可学习的prefix tokens
    """
    
    def __init__(
        self,
        hidden_size: int = 1152,  # ESM++的隐藏维度
        num_virtual_tokens: int = 20,
        prefix_projection: bool = True,
        prefix_hidden_size: Optional[int] = None,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_virtual_tokens = num_virtual_tokens
        self.prefix_projection = prefix_projection
        
        if prefix_projection:
            # 使用投影层，可以减少参数量
            self.prefix_hidden_size = prefix_hidden_size or hidden_size // 2
            # 可学习的前缀参数
            self.prefix_tokens = nn.Parameter(
                torch.randn(num_virtual_tokens, self.prefix_hidden_size)
            )
            # 投影到实际的隐藏维度
            self.prefix_projection_layer = nn.Linear(self.prefix_hidden_size, hidden_size)
        else:
            # 直接学习全尺寸的前缀向量
            self.prefix_tokens = nn.Parameter(
                torch.randn(num_virtual_tokens, hidden_size)
            )
            self.prefix_projection_layer = None
        
        # 初始化
        self._init_weights()
        
        logger.info(f"CustomPrefixTuning initialized: {num_virtual_tokens} virtual tokens, "
                   f"hidden_size={hidden_size}, projection={prefix_projection}")
    
    def _init_weights(self):
        """初始化权重"""
        if self.prefix_projection:
            nn.init.normal_(self.prefix_tokens, std=0.02)
            nn.init.normal_(self.prefix_projection_layer.weight, std=0.02)
            nn.init.zeros_(self.prefix_projection_layer.bias)
        else:
            nn.init.normal_(self.prefix_tokens, std=0.02)
    
    def get_prefix_embeddings(self, batch_size: int) -> torch.Tensor:
        """
        获取prefix embeddings
        
        参数:
            batch_size: 批次大小
            
        返回:
            shape为[batch_size, num_virtual_tokens, hidden_size]的prefix embeddings
        """
        if self.prefix_projection:
            prefix_embeds = self.prefix_projection_layer(self.prefix_tokens)
        else:
            prefix_embeds = self.prefix_tokens
        
        # 扩展到batch维度
        prefix_embeds = prefix_embeds.unsqueeze(0).expand(batch_size, -1, -1)
        return prefix_embeds
    
    def forward(
        self, 
        hidden_states: torch.Tensor, 
        attention_mask: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        前向传播
        
        参数:
            hidden_states: 原始的hidden states [batch_size, seq_len, hidden_size]
            attention_mask: 原始的attention mask [batch_size, seq_len]
            
        返回:
            updated_hidden_states: 添加了prefix的hidden states
            updated_attention_mask: 更新后的attention mask
        """
        batch_size, seq_len, hidden_size = hidden_states.shape
        
        # 获取prefix embeddings
        prefix_embeds = self.get_prefix_embeddings(batch_size)
        
        # 拼接prefix和原始hidden states
        updated_hidden_states = torch.cat([prefix_embeds, hidden_states], dim=1)
        
        # 更新attention_mask
        device = attention_mask.device
        dtype = attention_mask.dtype
        prefix_mask = torch.ones(batch_size, self.num_virtual_tokens, device=device, dtype=dtype)
        updated_attention_mask = torch.cat([prefix_mask, attention_mask], dim=1)
        
        logger.debug(f"Prefix tuning: {hidden_states.shape} -> {updated_hidden_states.shape}")
        
        return updated_hidden_states, updated_attention_mask


class CustomPromptTuning(nn.Module):
    """
    Prompt Tuning实现
    """
    
    def __init__(
        self,
        hidden_size: int = 1152,  # ESM++的隐藏维度
        num_virtual_tokens: int = 20,
        prompt_init: str = "random", 
        prompt_init_text: str = "Classify if TCR binds to peptide",
        tokenizer = None,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_virtual_tokens = num_virtual_tokens
        self.prompt_init = prompt_init
        
        # 可学习的提示向量
        self.prompt_embeddings = nn.Parameter(
            torch.randn(num_virtual_tokens, hidden_size)
        )
        
        # 初始化
        self._init_weights(prompt_init_text, tokenizer)
        
        logger.info(f"CustomPromptTuning initialized: {num_virtual_tokens} virtual tokens, "
                   f"hidden_size={hidden_size}, init={prompt_init}")
    
    def _init_weights(self, prompt_init_text: str = "", tokenizer = None):
        """初始化权重"""
        if self.prompt_init == "random":
            nn.init.normal_(self.prompt_embeddings, std=0.02)
        else:
            nn.init.normal_(self.prompt_embeddings, std=0.02)
    
    def forward(
        self, 
        hidden_states: torch.Tensor, 
        attention_mask: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        前向传播
        
        参数:
            hidden_states: 原始的hidden states [batch_size, seq_len, hidden_size]
            attention_mask: 原始的attention mask [batch_size, seq_len]
            
        返回:
            updated_hidden_states: 添加了prompt的hidden states
            updated_attention_mask: 更新后的attention mask
        """
        batch_size, seq_len, hidden_size = hidden_states.shape
        
        # 将提示向量扩展到批次维度
        prompt_embeds = self.prompt_embeddings.unsqueeze(0).expand(batch_size, -1, -1)
        
        # 拼接提示向量与原始隐藏状态
        updated_hidden_states = torch.cat([prompt_embeds, hidden_states], dim=1)
        
        # 更新注意力掩码
        device = attention_mask.device
        dtype = attention_mask.dtype
        prompt_mask = torch.ones(batch_size, self.num_virtual_tokens, device=device, dtype=dtype)
        updated_attention_mask = torch.cat([prompt_mask, attention_mask], dim=1)
        
        logger.debug(f"Prompt tuning: {hidden_states.shape} -> {updated_hidden_states.shape}")
        
        return updated_hidden_states, updated_attention_mask


class CustomTuningWrapper(nn.Module):
    """
    Tuning的包装器
    """
    
    def __init__(self, tuning_type: str, tuning_config: dict):
        super().__init__()
        self.tuning_type = tuning_type
        
        if tuning_type == "prefix":
            self.tuning_module = CustomPrefixTuning(**tuning_config)
        elif tuning_type == "prompt":
            self.tuning_module = CustomPromptTuning(**tuning_config)
        else:
            raise ValueError(f"Unsupported tuning type: {tuning_type}")
        
        # 计算参数数量
        total_params = sum(p.numel() for p in self.tuning_module.parameters())
        logger.info(f"Custom {tuning_type} tuning parameters: {total_params:,}")
    
    def forward(self, hidden_states: torch.Tensor, attention_mask: torch.Tensor):
        return self.tuning_module(hidden_states, attention_mask)
    
    def get_trainable_parameters(self):
        """获取可训练参数"""
        return [p for p in self.tuning_module.parameters() if p.requires_grad]
