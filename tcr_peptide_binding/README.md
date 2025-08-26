# TCR-肽结合预测系统

一个结构化、模块化的TCR（T细胞受体）与肽结合预测深度学习系统，基于ESM++预训练蛋白质语言模型、PEFT微调技术和Cross Attention机制。

## 🎯 项目特色

### 核心技术栈
- **预训练模型**: ESM++ (Evolutionary Scale Modeling++)
- **微调技术**: 支持7种PEFT方法 (LoRA, AdaLoRA, VeRA, BOFT, FourierFT, OFT, IA3)
- **注意力机制**: 双向Cross Attention融合
- **训练框架**: PyTorch Lightning
- **配置管理**: YAML配置文件 + 命令行参数覆盖

### 模型架构
```
TCR序列 ──┐
          ├─→ ESM++ 编码器 ──┐
肽序列 ───┘                  ├─→ Cross Attention ──→ 分类器 ──→ 结合预测
                            ┘
            ↓ PEFT微调        ↓ 双向注意力      ↓ 池化+融合
          (LoRA/AdaLoRA/...)  (TCR↔肽相互关注)  (连接/自适应)
```

## 📁 项目结构

```
tcr_peptide_binding/
├── configs/                    # 配置文件
│   └── default_config.yaml    # 默认配置（详细注释）
├── src/                       # 源代码
│   ├── models/               # 模型组件
│   │   ├── encoders.py       # TCR和肽编码器
│   │   ├── attention.py      # Cross Attention机制
│   │   ├── classifiers.py    # 结合预测分类器
│   │   └── binding_model.py  # 完整模型组装
│   ├── data/                 # 数据处理
│   │   ├── dataset.py        # 数据集和数据加载器
│   │   └── preprocessing.py  # 数据预处理工具
│   ├── training/             # 训练模块
│   │   └── lightning_module.py # PyTorch Lightning模块
│   └── utils/                # 工具函数
│       ├── config.py         # 配置管理
│       ├── logging_setup.py  # 日志配置
│       ├── paths.py          # 路径管理
│       ├── reproducibility.py # 随机种子设置
│       └── metrics.py        # 评估指标
├── scripts/                  # 执行脚本
│   └── train.py             # 主训练脚本
├── docs/                    # 文档
├── tests/                   # 测试
└── outputs/                 # 输出目录
    ├── checkpoints/         # 模型检查点
    ├── logs/               # 训练日志
    ├── results/            # 评估结果
    └── plots/              # 可视化图表
```

## 🚀 快速开始



### 1. 准备数据

数据格式为CSV文件，包含以下列：
```csv
TCR,Peptide,Label
CASSRTGDNEQFF,KLGGALQAK,1
CASSLGDSEQFF,VLQAGQVVL,0
...
```

### 2. 开始训练

```bash
# 使用默认配置训练
python scripts/train.py --data_path data/your_data.csv

# 自定义参数训练
python scripts/train.py \
    --data_path data/your_data.csv \
    --batch_size 8 \
    --learning_rate 1e-5 \
    --epochs 20 \
    --peft_method lora

# 使用自定义配置文件
python scripts/train.py \
    --config configs/my_config.yaml \
    --data_path data/your_data.csv
```

## ⚙️ 配置说明



### Cross Attention策略

- 双向交叉注意力
- 简单有效，计算开销小

### 硬件配置建议

## 📊 训练监控

### TensorBoard
```bash
tensorboard --logdir outputs/logs/tensorboard
```

### 训练指标
- **损失**: `train_loss`, `val_loss`
- **准确率**: `train_acc`, `val_acc`
- **F1分数**: `val_f1`
- **AUC指标**: `val_auroc`, `val_auprc`
- **学习率**: `learning_rate`

## 🔧 高级配置

### 自定义配置文件

复制 `configs/default_config.yaml` 并修改所需参数：

```yaml
# 模型配置
model:
  tokenizer_name: "Synthyra/ESMplusplus_large"

# PEFT配置
peft:
  method: "adalora"  # 使用AdaLoRA
  lora_r: 16         # 增加LoRA秩
  lora_alpha: 32

# 训练配置  
training:
  batch_size: 16
  learning_rate: 1e-5
  epochs: 30

# 融合配置
fusion:
  type: "enhanced"   # 使用增强版融合
  use_contrastive: true
```

### 命令行参数覆盖

```bash
python scripts/train.py \
    --config configs/my_config.yaml \
    --batch_size 8 \              # 覆盖配置文件中的batch_size
    --peft_method vera \          # 覆盖PEFT方法
    --devices 2                   # 使用2个GPU
```

## 📈 结果分析

训练完成后，系统会自动生成：

1. **模型检查点**: `outputs/checkpoints/`
2. **训练日志**: `outputs/logs/`
3. **评估报告**: `outputs/results/`
4. **可视化图表**: `outputs/plots/`

## 📚 技术细节

### Cross Attention机制

```python
# TCR关注肽的重要信息
tcr_attended = MultiheadAttention(
    query=tcr_embeddings,      # TCR作为查询
    key=peptide_embeddings,    # 肽作为键
    value=peptide_embeddings   # 肽作为值
)

# 肽关注TCR的重要信息  
peptide_attended = MultiheadAttention(
    query=peptide_embeddings,  # 肽作为查询
    key=tcr_embeddings,        # TCR作为键
    value=tcr_embeddings       # TCR作为值
)
```
