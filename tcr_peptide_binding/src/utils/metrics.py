#!/usr/bin/env python3
"""
评估指标计算工具

提供各种评估指标的计算和可视化功能。
"""

import torch
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    roc_curve,
    precision_recall_curve,
)
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional, Tuple, Union
import logging

logger = logging.getLogger(__name__)


class MetricsCalculator:
    """
    评估指标计算器

    计算各种二分类评估指标并提供可视化功能。
    """

    def __init__(self, device: Optional[torch.device] = None):
        """
        初始化指标计算器

        参数:
            device: 计算设备
        """
        self.device = device or torch.device("cpu")
        self.reset()

    def reset(self):
        """重置收集的数据"""
        self.predictions = []
        self.probabilities = []
        self.labels = []

    def update(
        self,
        preds: Union[torch.Tensor, np.ndarray, List],
        probs: Union[torch.Tensor, np.ndarray, List],
        labels: Union[torch.Tensor, np.ndarray, List],
    ):
        """
        更新预测结果

        参数:
            preds: 预测类别
            probs: 预测概率
            labels: 真实标签
        """
        # 转换为numpy数组
        if isinstance(preds, torch.Tensor):
            preds = preds.cpu().numpy()
        if isinstance(probs, torch.Tensor):
            probs = probs.cpu().numpy()
        if isinstance(labels, torch.Tensor):
            labels = labels.cpu().numpy()

        self.predictions.extend(preds)
        self.probabilities.extend(probs)
        self.labels.extend(labels)

    def compute_metrics(self) -> Dict[str, float]:
        """
        计算所有评估指标

        返回:
            指标字典
        """
        if not self.predictions:
            logger.warning("No prediction data available, cannot compute metrics")
            return {}

        preds = np.array(self.predictions)
        probs = np.array(self.probabilities)
        labels = np.array(self.labels)

        metrics = {}

        try:
            # 基础分类指标
            metrics["accuracy"] = accuracy_score(labels, preds)
            metrics["precision"] = precision_score(labels, preds, average="binary")
            metrics["recall"] = recall_score(labels, preds, average="binary")
            metrics["f1"] = f1_score(labels, preds, average="binary")

            # AUC指标
            metrics["roc_auc"] = roc_auc_score(labels, probs)
            metrics["pr_auc"] = average_precision_score(labels, probs)

            # 混淆矩阵元素
            cm = confusion_matrix(labels, preds)
            tn, fp, fn, tp = cm.ravel()

            metrics["true_negatives"] = tn
            metrics["false_positives"] = fp
            metrics["false_negatives"] = fn
            metrics["true_positives"] = tp

            # 特异性（真负例率）
            metrics["specificity"] = tn / (tn + fp) if (tn + fp) > 0 else 0

            # Matthews相关系数
            mcc_denom = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
            if mcc_denom != 0:
                metrics["mcc"] = (tp * tn - fp * fn) / mcc_denom
            else:
                metrics["mcc"] = 0

        except Exception as e:
            logger.error(f"Error computing metrics: {e}")
            return {}

        return metrics

    def print_metrics_report(self):
        """打印详细的指标报告"""

        metrics = self.compute_metrics()
        if not metrics:
            return

        print("\n" + "=" * 50)
        print("Evaluation Metrics Report")
        print("=" * 50)

        print(f"Accuracy:     {metrics['accuracy']:.4f}")
        print(f"Precision:    {metrics['precision']:.4f}")
        print(f"Recall:       {metrics['recall']:.4f}")
        print(f"F1-Score:     {metrics['f1']:.4f}")
        print(f"Specificity:  {metrics['specificity']:.4f}")
        print(f"MCC:          {metrics['mcc']:.4f}")
        print(f"ROC-AUC:      {metrics['roc_auc']:.4f}")
        print(f"PR-AUC:       {metrics['pr_auc']:.4f}")

        print("\nConfusion Matrix:")
        print(f"   True Negative (TN): {metrics['true_negatives']}")
        print(f"   False Positive (FP): {metrics['false_positives']}")
        print(f"   False Negative (FN): {metrics['false_negatives']}")
        print(f"   True Positive (TP): {metrics['true_positives']}")

        print("=" * 50 + "\n")

    def plot_confusion_matrix(
        self, save_path: Optional[str] = None, figsize: Tuple[int, int] = (8, 6)
    ) -> plt.Figure:
        """
        绘制混淆矩阵

        参数:
            save_path: 保存路径
            figsize: 图片大小

        返回:
            matplotlib图形对象
        """
        if not self.predictions:
            logger.warning("No prediction data available, cannot plot confusion matrix")
            return None

        preds = np.array(self.predictions)
        labels = np.array(self.labels)

        # 计算混淆矩阵
        cm = confusion_matrix(labels, preds)

        # 创建图形
        fig, ax = plt.subplots(figsize=figsize)

        # 绘制热图
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            ax=ax,
            xticklabels=["Non-binding", "Binding"],
            yticklabels=["Non-binding", "Binding"],
        )

        ax.set_xlabel("Predicted class")
        ax.set_ylabel("True class")
        ax.set_title("Confusion Matrix")

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            logger.info(f"Confusion matrix saved: {save_path}")

        return fig

    def plot_roc_curve(
        self, save_path: Optional[str] = None, figsize: Tuple[int, int] = (8, 6)
    ) -> plt.Figure:
        """
        绘制ROC曲线

        参数:
            save_path: 保存路径
            figsize: 图片大小

        返回:
            matplotlib图形对象
        """
        if not self.predictions:
            logger.warning("No prediction data available, cannot plot ROC curve")
            return None

        probs = np.array(self.probabilities)
        labels = np.array(self.labels)

        # 计算ROC曲线
        fpr, tpr, _ = roc_curve(labels, probs)
        roc_auc = roc_auc_score(labels, probs)

        # 创建图形
        fig, ax = plt.subplots(figsize=figsize)

        # 绘制ROC曲线
        ax.plot(fpr, tpr, color="darkorange", lw=2, label=f"ROC curve (AUC = {roc_auc:.4f})")
        ax.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--", label="Random classifier")

        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel("False Positive Rate (FPR)")
        ax.set_ylabel("True Positive Rate (TPR)")
        ax.set_title("ROC Curve")
        ax.legend(loc="lower right")
        ax.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            logger.info(f"ROC curve saved: {save_path}")

        return fig

    def plot_precision_recall_curve(
        self, save_path: Optional[str] = None, figsize: Tuple[int, int] = (8, 6)
    ) -> plt.Figure:
        """
        绘制精确率-召回率曲线

        参数:
            save_path: 保存路径
            figsize: 图片大小

        返回:
            matplotlib图形对象
        """
        if not self.predictions:
            logger.warning("No prediction data available, cannot plot PR curve")
            return None

        probs = np.array(self.probabilities)
        labels = np.array(self.labels)

        # 计算PR曲线
        precision, recall, _ = precision_recall_curve(labels, probs)
        pr_auc = average_precision_score(labels, probs)

        # 创建图形
        fig, ax = plt.subplots(figsize=figsize)

        # 绘制PR曲线
        ax.plot(recall, precision, color="darkorange", lw=2, label=f"PR curve (AUC = {pr_auc:.4f})")

        # 基线（随机分类器）
        positive_ratio = np.sum(labels) / len(labels)
        ax.axhline(
            y=positive_ratio,
            color="navy",
            linestyle="--",
            label=f"Random classifier (AP = {positive_ratio:.4f})",
        )

        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel("Recall")
        ax.set_ylabel("Precision")
        ax.set_title("Precision–Recall Curve")
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            logger.info(f"PR curve saved: {save_path}")

        return fig

    def save_results(self, save_path: str):
        """
        保存预测结果到CSV文件

        参数:
            save_path: 保存路径
        """
        if not self.predictions:
            logger.warning("No prediction data available, cannot save results")
            return

        df = pd.DataFrame(
            {
                "true_label": self.labels,
                "predicted_label": self.predictions,
                "binding_probability": self.probabilities,
                "correct": np.array(self.labels) == np.array(self.predictions),
            }
        )

        df.to_csv(save_path, index=False)
        logger.info(f"Prediction results saved: {save_path}")
