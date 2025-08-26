"""
模型模块

包含所有与模型相关的组件：
- 序列编码器（TCR和肽）
- 注意力机制（Cross Attention）
- 分类器
- 完整的绑定预测模型
"""

from .encoders import TCREncoder, PeptideEncoder
from .attention import CrossAttentionFusion, EnhancedCrossAttentionFusion
from .classifiers import BindingClassifier, EnhancedBindingClassifier
from .binding_model import TCRPeptideBindingModel

__all__ = [
    "TCREncoder",
    "PeptideEncoder",
    "CrossAttentionFusion",
    "EnhancedCrossAttentionFusion",
    "BindingClassifier",
    "EnhancedBindingClassifier",
    "TCRPeptideBindingModel",
]
