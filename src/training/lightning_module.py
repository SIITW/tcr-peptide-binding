#!/usr/bin/env python3


import torch
import pytorch_lightning as pl
from torch.optim import AdamW, Adam
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
import torchmetrics
from typing import Dict, Any
import logging

from ..models.binding_model import create_binding_model

logger = logging.getLogger(__name__)


class TCRPeptideBindingLightningModule(pl.LightningModule):
    """
    TCR-肽结合预测的PyTorch Lightning训练模块

    集成了完整的训练流程：
    1. 模型初始化和配置
    2. 训练、验证、测试步骤
    3. 优化器和调度器配置
    4. 评估指标计算
    """

    def __init__(self, config: Dict[str, Any]):
        """
        初始化Lightning模块

        参数:
            config: 完整的配置字典
        """
        super().__init__()

        # 保存超参数
        self.save_hyperparameters(config)

        self.config = config
        training_config = config.get("training", {})

        lr_value = training_config.get("learning_rate", 2e-5)
        wd_value = training_config.get("weight_decay", 0.01)

        try:
            self.learning_rate = float(lr_value)
        except (ValueError, TypeError) as e:
            logger.warning(f"Learning rate parse error {lr_value}: {e}, using default 2e-5")
            self.learning_rate = 2e-5

        try:
            self.weight_decay = float(wd_value)
        except (ValueError, TypeError) as e:
            logger.warning(f"Weight decay parse error {wd_value}: {e}, using default 0.01")
            self.weight_decay = 0.01

        logger.info(f"Training parameters: lr={self.learning_rate}, weight_decay={self.weight_decay}")

        logger.info("Initializing Lightning module...")

        self.model = create_binding_model(config)

        # 评估指标
        self._setup_metrics()

        logger.info("Lightning module initialization completed")

    def _setup_metrics(self) -> None:
        """设置评估指标"""

        logger.info("Initializing evaluation metrics...")

        # 训练指标
        self.train_acc = torchmetrics.Accuracy(task="binary")

        # 验证指标
        self.val_acc = torchmetrics.Accuracy(task="binary")
        self.val_f1 = torchmetrics.F1Score(task="binary")
        self.val_precision = torchmetrics.Precision(task="binary")
        self.val_recall = torchmetrics.Recall(task="binary")
        self.val_auroc = torchmetrics.AUROC(task="binary")
        self.val_auprc = torchmetrics.AveragePrecision(task="binary")

        # 测试指标
        self.test_acc = torchmetrics.Accuracy(task="binary")
        self.test_f1 = torchmetrics.F1Score(task="binary")
        self.test_precision = torchmetrics.Precision(task="binary")
        self.test_recall = torchmetrics.Recall(task="binary")
        self.test_auroc = torchmetrics.AUROC(task="binary")
        self.test_auprc = torchmetrics.AveragePrecision(task="binary")

    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        前向传播

        参数:
            batch: 输入批次数据

        返回:
            模型输出，包含logits和loss（如果有标签）
        """
        return self.model(batch)

    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """
        训练步骤

        在每个训练批次中：
        1. 前向传播得到预测结果
        2. 计算损失和准确率
        3. 记录训练指标
        """
        outputs = self(batch)
        loss = outputs["loss"]
        logits = outputs["logits"]
        labels = batch["labels"]

        # 计算预测结果
        preds = torch.argmax(logits, dim=1)

        # 更新训练指标
        self.train_acc(preds, labels)

        # 记录指标
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log(
            "train_acc", self.train_acc, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True
        )
        if "main_loss" in outputs:
            self.log(
                "train_main_loss", outputs["main_loss"], on_step=True, on_epoch=True, sync_dist=True
            )

        return loss

    def validation_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """
        验证步骤

        在每个验证批次中：
        1. 前向传播得到预测结果
        2. 计算多种评估指标
        3. 记录验证指标
        """
        outputs = self(batch)
        loss = outputs["loss"]
        logits = outputs["logits"]
        labels = batch["labels"]

        # 计算预测结果和概率
        preds = torch.argmax(logits, dim=1)
        probs = torch.softmax(logits, dim=1)[:, 1]  # 正类概率

        # 更新验证指标
        self.val_acc(preds, labels)
        self.val_f1(preds, labels)
        self.val_precision(preds, labels)
        self.val_recall(preds, labels)
        self.val_auroc(probs, labels)
        self.val_auprc(probs, labels)

        # 记录指标
        self.log("val_loss", loss, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("val_acc", self.val_acc, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("val_f1", self.val_f1, on_epoch=True, sync_dist=True)
        self.log("val_precision", self.val_precision, on_epoch=True, sync_dist=True)
        self.log("val_recall", self.val_recall, on_epoch=True, sync_dist=True)
        self.log("val_auroc", self.val_auroc, on_epoch=True, sync_dist=True)
        self.log("val_auprc", self.val_auprc, on_epoch=True, sync_dist=True)

        return loss

    def test_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """
        测试步骤

        在每个测试批次中计算测试指标
        """
        # 前向传播
        outputs = self(batch)
        loss = outputs["loss"]
        logits = outputs["logits"]
        labels = batch["labels"]

        # 计算预测结果和概率
        preds = torch.argmax(logits, dim=1)
        probs = torch.softmax(logits, dim=1)[:, 1]  # 正类概率

        # 更新测试指标
        self.test_acc(preds, labels)
        self.test_f1(preds, labels)
        self.test_precision(preds, labels)
        self.test_recall(preds, labels)
        self.test_auroc(probs, labels)
        self.test_auprc(probs, labels)

        # 记录指标
        self.log("test_loss", loss, on_epoch=True, sync_dist=True)
        self.log("test_acc", self.test_acc, on_epoch=True, sync_dist=True)
        self.log("test_f1", self.test_f1, on_epoch=True, sync_dist=True)
        self.log("test_precision", self.test_precision, on_epoch=True, sync_dist=True)
        self.log("test_recall", self.test_recall, on_epoch=True, sync_dist=True)
        self.log("test_auroc", self.test_auroc, on_epoch=True, sync_dist=True)
        self.log("test_auprc", self.test_auprc, on_epoch=True, sync_dist=True)

        return loss

    def configure_optimizers(self):
        """
        配置优化器和学习率调度器

        返回:
            优化器和调度器配置
        """
        training_config = self.config.get("training", {})

        # 优化器配置
        optimizer_type = training_config.get("optimizer", "adamw").lower()

        optimizer_kwargs = {
            "lr": self.learning_rate,
            "weight_decay": self.weight_decay,
            "eps": float(training_config.get("eps", 1e-8)),
        }

        if optimizer_type == "adamw":
            optimizer_kwargs["betas"] = training_config.get("betas", [0.9, 0.999])
            optimizer = AdamW(self.parameters(), **optimizer_kwargs)
        elif optimizer_type == "adam":
            optimizer_kwargs["betas"] = training_config.get("betas", [0.9, 0.999])
            optimizer = Adam(self.parameters(), **optimizer_kwargs)
        else:
            raise ValueError(f"Unsupported optimizer type: {optimizer_type}")

        # 学习率调度器配置
        scheduler_type = training_config.get("scheduler", "cosine_with_warmup")

        if scheduler_type == "none":
            return optimizer

        # 计算调度参数
        warmup_ratio = training_config.get("warmup_ratio", 0.1)
        min_lr_ratio = training_config.get("min_lr_ratio", 0.01)

        # 估计总步数
        if (
            hasattr(self.trainer, "estimated_stepping_batches")
            and self.trainer.estimated_stepping_batches
        ):
            total_steps = self.trainer.estimated_stepping_batches
        else:
            total_steps = 1000 
        warmup_steps = int(total_steps * warmup_ratio)

        if scheduler_type == "cosine_with_warmup":
            # 线性热身 + 余弦退火
            warmup_scheduler = LinearLR(optimizer, start_factor=0.1, total_iters=warmup_steps)

            cosine_scheduler = CosineAnnealingLR(
                optimizer,
                T_max=total_steps - warmup_steps,
                eta_min=self.learning_rate * min_lr_ratio,
            )

            scheduler = SequentialLR(
                optimizer,
                schedulers=[warmup_scheduler, cosine_scheduler],
                milestones=[warmup_steps],
            )

        elif scheduler_type == "cosine":
            # 纯余弦退火
            scheduler = CosineAnnealingLR(
                optimizer, T_max=total_steps, eta_min=self.learning_rate * min_lr_ratio
            )

        else:
            raise ValueError(f"Unsupported scheduler type: {scheduler_type}")

        logger.info(f"Optimizer configuration: {optimizer_type}")
        logger.info(f"Scheduler configuration: {scheduler_type}")
        logger.info(f"Warmup steps: {warmup_steps} / {total_steps}")

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step", 
                "frequency": 1,
                "name": "learning_rate",
            },
        }

    def on_train_epoch_end(self):
        # 记录当前学习率
        current_lr = self.optimizers().param_groups[0]["lr"]
        self.log("learning_rate", current_lr, sync_dist=True)

    def on_validation_epoch_end(self):
        # 打印主要指标
        if self.trainer.current_epoch % 5 == 0: 
            logger.info(
                f"Epoch {self.trainer.current_epoch}: "
                f"val_loss={self.trainer.logged_metrics.get('val_loss', 0):.4f}, "
                f"val_acc={self.trainer.logged_metrics.get('val_acc', 0):.4f}, "
                f"val_f1={self.trainer.logged_metrics.get('val_f1', 0):.4f}"
            )

    def predict_step(
        self, batch: Dict[str, torch.Tensor], batch_idx: int
    ) -> Dict[str, torch.Tensor]:
        """
        预测步骤

        返回:
            预测结果，包含概率和预测类别
        """
        return self.model.predict(batch)
