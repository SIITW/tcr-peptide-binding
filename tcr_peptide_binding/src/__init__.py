"""
TCR-Peptide Binding Prediction Package

项目说明（中文注释）：
- 优化的分词器管理（避免重复加载ESM++模型）
- 完整的数据验证系统
- 规范的错误处理和类型注解
- 单元测试覆盖

主要特性（中文注释）：
- 支持多种PEFT微调方法 (LoRA, AdaLoRA, 等)
- Cross-attention 融合机制
- 基于 PyTorch Lightning 的训练框架
- 内存优化的分词器缓存
"""

__version__ = "1.0.0"

# 延迟导入避免依赖问题（中文注释）
# 只有在实际需要时才导入重量级模块
