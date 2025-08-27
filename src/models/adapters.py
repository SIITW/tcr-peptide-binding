#!/usr/bin/env python3
import torch
import torch.nn as nn
from typing import Dict, Any, Optional, List
import logging

logger = logging.getLogger(__name__)


class TokenAdapter(nn.Module):
    """
    Token Adapter实现
    
    真正在token embeddings级别进行适配，通过hook劫持embedding输出。
    在embedding层输出后立即进行任务特定的变换，然后再进入Transformer。
    """
    
    def __init__(
        self,
        hidden_size: int,
        adapter_size: int = 64,
        dropout: float = 0.1,
        activation: str = "relu",
        init_weights: bool = True
    ):
        """
        初始化Token Adapter
        
        参数:
            hidden_size: 隐藏层维度
            adapter_size: 适配器瓶颈维度
            dropout: dropout率
            activation: 激活函数类型
            init_weights: 是否初始化权重
        """
        super().__init__()
        
        self.hidden_size = hidden_size
        self.adapter_size = adapter_size
        
        # 下投影：隐藏维度 → 适配器维度
        self.down_proj = nn.Linear(hidden_size, adapter_size)
        
        # 上投影：适配器维度 → 隐藏维度  
        self.up_proj = nn.Linear(adapter_size, hidden_size)
        
        # 激活函数
        if activation.lower() == "relu":
            self.activation = nn.ReLU()
        elif activation.lower() == "gelu":
            self.activation = nn.GELU()
        elif activation.lower() == "swish":
            self.activation = nn.SiLU()
        else:
            self.activation = nn.ReLU()
            
        # 丢弃层
        self.dropout = nn.Dropout(dropout)
        
        # 层归一化
        self.layer_norm = nn.LayerNorm(hidden_size)
        
        if init_weights:
            self._init_weights()
    
    def _init_weights(self):
        """初始化权重"""
        # 下投影使用标准初始化
        nn.init.kaiming_uniform_(self.down_proj.weight, a=0, mode='fan_in')
        nn.init.zeros_(self.down_proj.bias)
        
        # 上投影零初始化确保初始时adapter不影响原模型
        nn.init.zeros_(self.up_proj.weight)
        nn.init.zeros_(self.up_proj.bias)
    
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        参数:
            hidden_states: [batch_size, seq_len, hidden_size]
            
        返回:
            torch.Tensor: 适配后的隐藏状态
        """
        # 保存输入用于残差连接
        residual = hidden_states
        
        # 1. 下投影 
        down_projected = self.down_proj(hidden_states)  # 形状 [..., 适配器维度]
        
        # 2. 激活函数
        activated = self.activation(down_projected)
        
        # 3. 丢弃
        dropped = self.dropout(activated)
        
        # 4. 上投影
        up_projected = self.up_proj(dropped)  # 形状 [..., 隐藏维度]
        
        # 5. 残差连接
        adapted_states = residual + up_projected
        
        # 6. 层归一化
        output = self.layer_norm(adapted_states)
        
        return output


class PfeifferAdapter(nn.Module):
    """
    Pfeiffer Adapter实现
    在Transformer层内部插入的串行适配器，通过forward hook在attention和FFN之间插入。
    """
    
    def __init__(
        self,
        hidden_size: int,
        adapter_size: int = 64,
        dropout: float = 0.1,
        activation: str = "relu",
        init_weights: bool = True
    ):
        """
        初始化Pfeiffer Adapter
        
        参数:
            hidden_size: 隐藏层维度
            adapter_size: 适配器瓶颈维度
            dropout: dropout率
            activation: 激活函数类型
            init_weights: 是否初始化权重
        """
        super().__init__()
        
        self.hidden_size = hidden_size
        self.adapter_size = adapter_size
        
        # 下投影：隐藏维度 → 适配器维度
        self.down_proj = nn.Linear(hidden_size, adapter_size)
        
        # 上投影：适配器维度 → 隐藏维度
        self.up_proj = nn.Linear(adapter_size, hidden_size)
        
        # 激活函数
        if activation.lower() == "relu":
            self.activation = nn.ReLU()
        elif activation.lower() == "gelu":
            self.activation = nn.GELU()
        elif activation.lower() == "swish":
            self.activation = nn.SiLU()
        else:
            self.activation = nn.ReLU()
            
        # 丢弃层
        self.dropout = nn.Dropout(dropout)
        
        # 层归一化
        self.layer_norm = nn.LayerNorm(hidden_size)
        
        if init_weights:
            self._init_weights()
    
    def _init_weights(self):
        """初始化权重"""
        # 下投影使用标准初始化
        nn.init.kaiming_uniform_(self.down_proj.weight, a=0, mode='fan_in')
        nn.init.zeros_(self.down_proj.bias)
        
        # 上投影使用零初始化，确保初始时adapter不影响原模型
        nn.init.zeros_(self.up_proj.weight)
        nn.init.zeros_(self.up_proj.bias)
    
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        参数:
            hidden_states: [batch_size, seq_len, hidden_size]
            
        返回:
            torch.Tensor: 适配后的隐藏状态
        """
        # 保存输入用于残差连接
        residual = hidden_states
        
        # 1. 下投影
        down_projected = self.down_proj(hidden_states)  # 形状 [..., 适配器维度]
        
        # 2. 激活函数
        activated = self.activation(down_projected)
        
        # 3. 丢弃
        dropped = self.dropout(activated)
        
        # 4. 上投影
        up_projected = self.up_proj(dropped)  # 形状 [..., 隐藏维度]
        
        # 5. 残差连接
        output = residual + up_projected
        
        # 6. 层归一化
        output = self.layer_norm(output)
        
        return output


class TokenAdapterManager(nn.Module):
    """
    Token Adapter管理器
    通过hook劫持embedding层输出，在token embeddings上应用adapter
    """
    
    def __init__(
        self,
        base_model: nn.Module,
        hidden_size: int,
        adapter_size: int = 64,
        dropout: float = 0.1,
        activation: str = "relu"
    ):
        """
        初始化Token Adapter管理器
        
        参数:
            base_model: 基础模型
            hidden_size: 隐藏层维度
            adapter_size: 适配器瓶颈维度
            dropout: dropout率
            activation: 激活函数类型
        """
        super().__init__()
        
        self.base_model = base_model
        self.token_adapter = TokenAdapter(
            hidden_size=hidden_size,
            adapter_size=adapter_size,
            dropout=dropout,
            activation=activation,
            init_weights=True
        )
        
        # 定位嵌入层并注册hook
        self.hook = None
        if hasattr(base_model, 'embed'):
            # ESM++ 的嵌入层
            self.hook = base_model.embed.register_forward_hook(self._embedding_hook)
            logger.info("Token Adapter hook registered on embedding layer")
        else:
            logger.warning("Cannot find embedding layer, Token Adapter will not work correctly")
    
    def _embedding_hook(self, module, input_tuple, output):
        """Embedding层的forward hook"""
        # output 为嵌入层输出，形状 [batch_size, seq_len, hidden_size]
        logger.debug(f"Token Adapter Hook called! Input shape: {output.shape}")
        
        # 保存原始输出用于对比
        original_mean = output.mean().item()
        
        adapted_embeddings = self.token_adapter(output)
        
        # 验证adapter确实改变了embeddings
        adapted_mean = adapted_embeddings.mean().item()
        logger.debug(f"Token Adapter effect: original_mean={original_mean:.4f}, adapted_mean={adapted_mean:.4f}")
        
        return adapted_embeddings
    
    def remove_hook(self):
        """移除hook"""
        if self.hook is not None:
            self.hook.remove()
            self.hook = None
    
    def forward(self, *args, **kwargs):
        """前向传播，直接调用base_model"""
        return self.base_model(*args, **kwargs)


class PfeifferAdapterManager(nn.Module):
    """
    Pfeiffer Adapter管理器
    负责将Pfeiffer Adapter通过forward hook插入到指定的Transformer层中
    """
    
    def __init__(
        self, 
        base_model: nn.Module,
        hidden_size: int,
        adapter_size: int = 64,
        dropout: float = 0.1,
        activation: str = "relu",
        target_layers: Optional[List[int]] = None
    ):
        """
        初始化Pfeiffer Adapter管理器
        
        参数:
            base_model: 基础Transformer模型
            hidden_size: 隐藏层维度
            adapter_size: 适配器瓶颈维度
            dropout: dropout率
            activation: 激活函数类型
            target_layers: 要插入adapter的层索引列表，None表示所有层
        """
        super().__init__()
        
        self.base_model = base_model
        self.hidden_size = hidden_size
        self.adapters = nn.ModuleDict()
        self.hooks = []
        
        # 找到所有的Transformer层 (ESM++使用transformer.blocks.X格式)
        transformer_layers = []
        for name, module in base_model.named_modules():
            if name.startswith('transformer.blocks.') and name.count('.') == 2:
                layer_idx = int(name.split('.')[-1])
                transformer_layers.append((layer_idx, name, module))
        
        transformer_layers.sort(key=lambda x: x[0])
        
        # 确定要插入adapter的层
        if target_layers is None:
            target_layers = list(range(len(transformer_layers)))
        
        # 为每个目标层创建adapter并注册hook
        for layer_idx in target_layers:
            if layer_idx < len(transformer_layers):
                layer_name = f"layer_{layer_idx}"
                adapter = PfeifferAdapter(
                    hidden_size=hidden_size,
                    adapter_size=adapter_size,
                    dropout=dropout,
                    activation=activation,
                    init_weights=True
                )
                self.adapters[layer_name] = adapter
                
                # 注册forward hook到对应的层
                layer_module = transformer_layers[layer_idx][2]
                hook = layer_module.register_forward_hook(
                    self._create_hook_fn(layer_name)
                )
                self.hooks.append(hook)
        
        logger.info(f"Pfeiffer Adapters inserted into {len(self.adapters)} layers")
    
    def _create_hook_fn(self, layer_name: str):
        """为特定层创建hook函数"""
        def hook_fn(module, input_tuple, output):
            # 获取hidden states (通常是output的第一个元素)
            if isinstance(output, tuple):
                hidden_states = output[0]
                other_outputs = output[1:]
            else:
                hidden_states = output
                other_outputs = ()

            adapter = self.adapters[layer_name]
            adapted_hidden_states = adapter(hidden_states)

            adapted_mean = adapted_hidden_states.mean().item()
            if other_outputs:
                return (adapted_hidden_states,) + other_outputs
            else:
                return adapted_hidden_states
        
        return hook_fn
    
    def remove_hooks(self):
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()
    
    def forward(self, *args, **kwargs):
        return self.base_model(*args, **kwargs)


class HoulsbyAdapter(nn.Module):
    """
    Houlsby Adapter实现
    并行适配器结构，同时在注意力机制和FFN中插入适配器。
    """
    
    def __init__(
        self,
        hidden_size: int,
        adapter_size: int = 64,
        dropout: float = 0.1,
        activation: str = "relu",
        use_attention_adapter: bool = True,
        use_ffn_adapter: bool = True,
        init_weights: bool = True
    ):
        """
        初始化Houlsby Adapter
        
        参数:
            hidden_size: 隐藏层维度
            adapter_size: 适配器瓶颈维度
            dropout: dropout率
            activation: 激活函数类型
            use_attention_adapter: 是否在注意力层使用adapter
            use_ffn_adapter: 是否在FFN层使用adapter
            init_weights: 是否初始化权重
        """
        super().__init__()
        
        self.hidden_size = hidden_size
        self.adapter_size = adapter_size
        self.use_attention_adapter = use_attention_adapter
        self.use_ffn_adapter = use_ffn_adapter
        
        # 注意力适配器
        if use_attention_adapter:
            self.attention_adapter = self._create_adapter_block(
                hidden_size, adapter_size, dropout, activation
            )
        
        # FFN适配器
        if use_ffn_adapter:
            self.ffn_adapter = self._create_adapter_block(
                hidden_size, adapter_size, dropout, activation
            )
        
        if init_weights:
            self._init_weights()
    
    def _create_adapter_block(
        self, 
        hidden_size: int, 
        adapter_size: int, 
        dropout: float, 
        activation: str
    ) -> nn.Module:
        """创建适配器块"""
        # 激活函数选择
        if activation.lower() == "relu":
            act_fn = nn.ReLU()
        elif activation.lower() == "gelu":
            act_fn = nn.GELU()
        elif activation.lower() == "swish":
            act_fn = nn.SiLU()
        else:
            act_fn = nn.ReLU()
        
        return nn.Sequential(
            nn.Linear(hidden_size, adapter_size),    # 下投影
            act_fn,                                  # 激活函数
            nn.Dropout(dropout),                     # 丢弃
            nn.Linear(adapter_size, hidden_size),    # 上投影
            nn.LayerNorm(hidden_size)               # 层归一化
        )
    
    def _init_weights(self):
        """初始化权重"""
        for adapter in [self.attention_adapter, self.ffn_adapter]:
            if adapter is not None:
                # 下投影标准初始化
                nn.init.kaiming_uniform_(adapter[0].weight, a=0, mode='fan_in')
                nn.init.zeros_(adapter[0].bias)
                
                # 上投影零初始化
                nn.init.zeros_(adapter[3].weight)
                nn.init.zeros_(adapter[3].bias)
    
    def forward_attention_adapter(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        前向传播 - 注意力适配器部分
        
        参数:
            hidden_states: [batch_size, seq_len, hidden_size]
            
        返回:
            torch.Tensor: 注意力适配后的隐藏状态
        """
        if not self.use_attention_adapter:
            return hidden_states
        
        # 残差连接
        residual = hidden_states
        adapter_output = self.attention_adapter(hidden_states)
        
        return residual + adapter_output
    
    def forward_ffn_adapter(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        前向传播 - FFN适配器部分
        
        参数:
            hidden_states: [batch_size, seq_len, hidden_size]
            
        返回:
            torch.Tensor: FFN适配后的隐藏状态
        """
        if not self.use_ffn_adapter:
            return hidden_states
        
        # 残差连接
        residual = hidden_states
        adapter_output = self.ffn_adapter(hidden_states)
        
        return residual + adapter_output
    
    def forward(self, hidden_states: torch.Tensor, adapter_type: str = "ffn") -> torch.Tensor:
        """
        前向传播
        
        参数:
            hidden_states: [batch_size, seq_len, hidden_size]
            adapter_type: 适配器类型 ("attention" 或 "ffn")
            
        返回:
            torch.Tensor: 适配后的隐藏状态
        """
        if adapter_type == "attention":
            return self.forward_attention_adapter(hidden_states)
        elif adapter_type == "ffn":
            return self.forward_ffn_adapter(hidden_states)
        else:
            # 默认使用FFN适配器
            return self.forward_ffn_adapter(hidden_states)


def create_adapter(adapter_type: str, config: Dict[str, Any], hidden_size: int, base_model: nn.Module = None) -> nn.Module:
    """
    创建适配器的工厂函数
    
    参数:
        adapter_type: 适配器类型
        config: 配置字典
        hidden_size: 隐藏层维度
        base_model: 基础模型（Pfeiffer Adapter需要）
        
    返回:
        nn.Module: 创建的适配器
    """
    adapter_type = adapter_type.lower()
    
    if adapter_type == "token_adapter":
        if base_model is None:
            raise ValueError("Token Adapter requires base_model parameter")
        
        return TokenAdapterManager(
            base_model=base_model,
            hidden_size=hidden_size,
            adapter_size=config.get("adapter_size", 64),
            dropout=config.get("dropout", 0.1),
            activation=config.get("activation", "relu")
        )
     
    elif adapter_type == "pfeiffer_adapter":
        if base_model is None:
            raise ValueError("Pfeiffer Adapter requires base_model parameter")
        
        return PfeifferAdapterManager(
            base_model=base_model,
            hidden_size=hidden_size,
            adapter_size=config.get("adapter_size", 64),
            dropout=config.get("dropout", 0.1),
            activation=config.get("activation", "relu"),
            target_layers=config.get("target_layers", None)  # None表示所有层
        )
    
    elif adapter_type == "houlsby_adapter":
        return HoulsbyAdapter(
            hidden_size=hidden_size,
            adapter_size=config.get("adapter_size", 64),
            dropout=config.get("dropout", 0.1),
            activation=config.get("activation", "relu"),
            use_attention_adapter=config.get("use_attention_adapter", True),
            use_ffn_adapter=config.get("use_ffn_adapter", True),
            init_weights=config.get("init_weights", True)
        )
    
    else:
        raise ValueError(f"Unsupported adapter type: {adapter_type}")
