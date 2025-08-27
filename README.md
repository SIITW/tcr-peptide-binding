# TCR-肽结合预测系统

一个结构化、模块化的TCR（T细胞受体）与肽结合预测深度学习系统，基于ESM++预训练蛋白质语言模型、PEFT微调技术和Cross Attention机制。

## 🎯 项目特色

### 核心技术栈
- **预训练模型**: ESM++ (Evolutionary Scale Modeling++)
- **微调技术**: 多种 PEFT：LoRA、AdaLoRA、VeRA、BOFT、OFT、IA3、Prefix、Prompt、Token Adapter、Pfeiffer Adapter、Houlsby Adapter。
- **注意力机制**: 双向Cross Attention融合
- **训练框架**: PyTorch Lightning
- **配置管理**: YAML配置文件 + 命令行参数覆盖

### 模型架构
```
TCR序列 ──┐
          ├─→ ESM++ 编码器 ──┐
肽序列 ───┘                  ├─→ Cross Attention ──→ 分类器 ──→ 结合预测
                            ┘
            ↓       								↓ 双向注意力      	↓ 池化+融合
           PEFT微调   						(TCR↔肽相互关注)  		(连接/自适应)
```

## 📁 项目结构

```
tcr_peptide_binding_v2/
├── configs/
│   └── default_config.yaml        # 默认配置（可被命令行覆盖）
├── data/
│   └── tcr_peptide.csv            # 示例数据 (TCR, Peptide, Label)
├── scripts/
│   └── train.py                   # 训练入口脚本
├── src/
│   ├── data/
│   │   ├── dataset.py             # 数据集 + DataLoader (分词、padding、mask)
│   │   └── preprocessing.py       # 预处理与数据质量分析
│   ├── models/
│   │   ├── attention.py           # Cross Attention 融合（标准/增强）
│   │   ├── binding_model.py       # 编码器+融合+分类器组装
│   │   ├── classifiers.py         # 分类头（多种池化/融合）
│   │   ├── custom_tuning.py       # 自定义 Prefix/Prompt Tuning
│   │   ├── encoders.py            # ESM++ 编码器 + PEFT 选择/装配
│   │   └── adapters.py            # Token/Pfeiffer/Houlsby Adapter（hook）
│   ├── training/
│   │   └── lightning_module.py    # Lightning 模块（train/val/test/predict）
│   └── utils/
│       ├── config.py              # 配置加载/合并/命令行覆盖/保存
│       ├── logging_setup.py       # 日志初始化（控制台/文件/TensorBoard）
│       ├── paths.py               # 输出目录管理
│       ├── reproducibility.py     # 随机种子与确定性
│       └── tokenizer_manager.py   # 分词器缓存（ESM++ 模型 → tokenizer）
├── LICENSE
├── README.md
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
python scripts/train.py \
  --data_path data/tcr_peptide.csv \
  --peft_method lora \
  --batch_size 8 \
  --epochs 20
```

## 常用命令行参数

- 基础：`--data_path`（必填）、`--batch_size`、`--learning_rate/--lr`、`--epochs`、`--output_dir`
- PEFT：`--peft_method`，可选
  - `lora`/`adalora`/`vera`/`boft`/`oft`/`ia3`/`prefix`/`prompt`/`token_adapter`/`pfeiffer_adapter`/`houlsby_adapter`
- 数据：`--max_tcr_length`、`--max_peptide_length`、`--test_size`、`--val_size`
- 硬件：`--devices/--gpus`、`--precision`（32/16-mixed/bf16-mixed）、`--accumulate_grad_batches`
- 调试：`--fast_dev_run`
- 其它：`--output_dir`（默认 `outputs/`）

命令行会覆盖 YAML 中对应字段（映射见 `src/utils/config.py`）。最终生效配置会保存至 `outputs/<exp>/final_config.yaml`。



## ⚙️ 配置说明配置详解（configs/default_config.yaml）

- `tcr_encoder` / `peptide_encoder`：ESM++ 模型路径、是否冻结
- `peft`：方法选择与各自超参（见下一节）
- `fusion`：Cross Attention 融合
- `classifier`：池化（`cls|mean|max|attention`）、融合（`concat|add|multiply|adaptive`）
- `data`：最大长度、划分比例、基础预处理
- `training`：优化器（AdamW/Adam）、调度（cosine_with_warmup|cosine|none）、早停、梯度裁剪
- `hardware`：加速器（auto/gpu/cpu）、设备数、精度、梯度累积
- `logging`：控制台/文件/TensorBoard 配置
- `checkpointing`：保存/监控指标

## 支持的 PEFT 方法与实现要点

- LoRA (`lora`)、AdaLoRA (`adalora`)、VeRA (`vera`)、BOFT (`boft`)、OFT (`oft`)、IA3 (`ia3`)：
  - 通过 PEFT 库标准装配（`src/models/encoders.py::_setup_peft`）。
  - 可配置目标模块、秩、dropout 等。
- Prefix Tuning (`prefix`)：
  - 自定义实现（`src/models/custom_tuning.py`）。
  - 在编码器输出前拼接可学习前缀；支持 `prefix_projection`（低维→线性投影到 hidden_size）。
- Prompt Tuning (`prompt`)：
  - 自定义实现（`src/models/custom_tuning.py`）。
  - 拼接全维可学习 prompt
- Token Adapter (`token_adapter`)：
  - 在 embedding 输出处通过瓶颈层调制 token 表示，不改变下游序列长度（`src/models/adapters.py::TokenAdapterManager`）。
- Pfeiffer Adapter (`pfeiffer_adapter`)：
  - 在 Transformer 层内部（attention/FFN 之间）插入瓶颈结构，hook 到多层（`src/models/adapters.py::PfeifferAdapterManager`）。
- Houlsby Adapter (`houlsby_adapter`)：
  - 并行适配（注意力与 FFN 两支），可分别开关（`use_attention_adapter|use_ffn_adapter`）。

## Cross Attention 融合

### Cross Attention策略

- 双向交叉注意力
- 简单有效，计算开销小
- 双向交叉注意力（TCR→Peptide 与 Peptide→TCR），残差 + 层归一化 + FFN。

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
