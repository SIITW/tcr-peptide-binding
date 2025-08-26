"""
训练模块

包含PyTorch Lightning训练相关组件：
- Lightning模块：集成模型、优化器、训练步骤
- 训练工具：学习率调度、回调函数等
"""

from .lightning_module import TCRPeptideBindingLightningModule

__all__ = ["TCRPeptideBindingLightningModule"]
