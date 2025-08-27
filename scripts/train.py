#!/usr/bin/env python3
import sys
import os
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import argparse
import logging
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger

from src.utils.config import ConfigManager, create_config_from_args
from src.utils.logging_setup import setup_logging
from src.utils.paths import PathManager
from src.utils.reproducibility import set_seed, set_deterministic
from src.data.dataset import prepare_data_splits, create_dataloaders
from src.data.preprocessing import analyze_data_quality
from src.training.lightning_module import TCRPeptideBindingLightningModule
from datetime import datetime

logger = logging.getLogger(__name__)


def parse_arguments():
    parser = argparse.ArgumentParser(
        description='Train a TCR–peptide binding prediction model',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument('--config', type=str, default='configs/default_config.yaml',
                       help='Path to configuration file')
    parser.add_argument('--data_path', type=str, required=True,
                       help='Training data path (CSV)')

    parser.add_argument('--batch_size', type=int,
                       help='Batch size')
    parser.add_argument('--learning_rate', '--lr', type=float,
                       help='Learning rate')
    parser.add_argument('--epochs', type=int,
                       help='Number of epochs')
    parser.add_argument('--peft_method', type=str,
                       choices=['lora', 'adalora', 'vera', 'boft', 'oft', 'ia3', 'prefix',
                               'prompt',
                               'token_adapter', 'pfeiffer_adapter', 'houlsby_adapter'],
                       help='PEFT finetuning method')
    parser.add_argument('--max_tcr_length', type=int,
                       help='Max TCR sequence length')
    parser.add_argument('--max_peptide_length', type=int,
                       help='Max peptide sequence length')
    parser.add_argument('--test_size', type=float,
                       help='Test split ratio')
    parser.add_argument('--val_size', type=float,
                       help='Validation split ratio')
    parser.add_argument('--devices', '--gpus', type=int, dest='devices',
                       help='Number of GPUs to use')
    parser.add_argument('--precision', type=str,
                       choices=['32', '16-mixed', 'bf16-mixed'],
                       help='Training precision')
    parser.add_argument('--accumulate_grad_batches', type=int,
                       help='Gradient accumulation steps')
    parser.add_argument('--output_dir', type=str, default='outputs',
                       help='Output directory')
    parser.add_argument('--fast_dev_run', action='store_true',
                       help='Run in fast development mode (only a few batches)')

    return parser.parse_args()


def setup_experiment(config: dict, args: argparse.Namespace) -> PathManager:
    """设置实验环境"""

    # 解析 peft 方法
    peft_method = (getattr(args, 'peft_method', None)
                   or config.get('training', {}).get('peft_method')
                   or 'baseline')
    peft_method = str(peft_method).lower()
    # 生成时间
    ts = datetime.now().strftime('%Y%m%d-%H%M%S')

    # 实验名
    experiment_name = f"{peft_method}_{ts}"

    # 创建路径管理器并建目录
    path_manager = PathManager(
        base_dir=args.output_dir,
        experiment_name=experiment_name
    )
    path_manager.setup_directories()

    # 设置日志
    log_config = dict(config.get('logging', {})) 
    if 'file' not in log_config or not isinstance(log_config['file'], dict):
        log_config['file'] = {}
    log_config['file']['path'] = str(path_manager.logs_dir / 'training.log')
    setup_logging(log_config)

    logger.info("Starting TCR-peptide binding prediction model training")
    logger.info(f"Experiment directory: {path_manager.base_dir}")
    logger.info(f"Experiment name: {experiment_name}")
    logger.info(f"PEFT method: {peft_method}")

    # 随机种子
    seed = config.get('experiment', {}).get('seed', 42)
    set_seed(seed)
    deterministic = config.get('experiment', {}).get('deterministic', True)
    if deterministic:
        set_deterministic()
        logger.info(f"Set random seed: {seed} (deterministic training)")
    else:
        logger.info(f"Set random seed: {seed}")

    # 将最终实验名写回 config
    config.setdefault('experiment', {})
    config['experiment']['name'] = experiment_name

    # 保存最终配置到实验目录
    config_manager = ConfigManager()
    config_manager.config = config
    config_manager.save_config(path_manager.base_dir / 'final_config.yaml')

    return path_manager



def prepare_data(config: dict, args: argparse.Namespace):
    """准备训练数据"""
    
    logger.info("Preparing training data...")
    
    data_path = Path(args.data_path)
    
    if not data_path.exists():
        raise FileNotFoundError(f"Data file not found: {data_path}")
    
    # 数据分割
    data_config = config.get('data', {})
    train_df, val_df, test_df = prepare_data_splits(
        data_path=str(data_path),
        test_size=data_config.get('test_size', 0.2),
        val_size=data_config.get('val_size', 0.1),
        random_state=config.get('experiment', {}).get('seed', 42)
    )
    
    # 数据分析
    logger.info("Analyzing training data quality...")
    quality_report = analyze_data_quality(train_df)
    
    # 数据加载器
    tokenizer_name = config.get('model', {}).get('tokenizer_name', 'Synthyra/ESMplusplus_large')
    training_config = config.get('training', {})
    
    train_loader, val_loader, test_loader = create_dataloaders(
        train_df=train_df,
        val_df=val_df if len(val_df) > 0 else None,
        test_df=test_df,
        tokenizer_name=tokenizer_name,
        batch_size=training_config.get('batch_size', 4),
        max_tcr_length=data_config.get('max_tcr_length', 128),
        max_peptide_length=data_config.get('max_peptide_length', 64),
        num_workers=training_config.get('num_workers', 2)
    )
    
    return train_loader, val_loader, test_loader


def create_model(config: dict) -> TCRPeptideBindingLightningModule:
    """创建Lightning模型"""
    
    logger.info("Creating model...")
    
    # 创建Lightning模块
    model = TCRPeptideBindingLightningModule(config)
    
    logger.info("Model creation completed")
    
    return model


def setup_callbacks(config: dict, path_manager: PathManager):
    """设置训练回调"""
    
    callbacks = []
    
    # 模型检查点
    checkpoint_config = config.get('checkpointing', {})
    checkpoint_callback = ModelCheckpoint(
        dirpath=path_manager.checkpoints_dir,
        filename=checkpoint_config.get('filename', 'epoch={epoch:02d}-val_loss={val_loss:.4f}'),
        monitor=checkpoint_config.get('monitor', 'val_loss'),
        mode=checkpoint_config.get('mode', 'min'),
        save_top_k=checkpoint_config.get('save_top_k', 1),
        save_last=checkpoint_config.get('save_last', True),
        auto_insert_metric_name=checkpoint_config.get('auto_insert_metric_name', False)
    )
    callbacks.append(checkpoint_callback)
    
    # 早停
    early_stopping_config = config.get('training', {}).get('early_stopping', {})
    if early_stopping_config.get('enabled', True):
        early_stopping = EarlyStopping(
            monitor=early_stopping_config.get('monitor', 'val_loss'),
            patience=early_stopping_config.get('patience', 5),
            mode=early_stopping_config.get('mode', 'min'),
            min_delta=early_stopping_config.get('min_delta', 0.001)
        )
        callbacks.append(early_stopping)
    
    # 学习率监控
    lr_monitor = LearningRateMonitor(logging_interval='step')
    callbacks.append(lr_monitor)
    
    logger.info(f"Setup {len(callbacks)} training callbacks")
    
    return callbacks


def setup_logger_and_trainer(config: dict, path_manager: PathManager, callbacks: list, args: argparse.Namespace):
    """设置日志记录器和训练器"""
    
    # TensorBoard日志记录器
    tb_config = config.get('logging', {}).get('tensorboard', {})
    if tb_config.get('enabled', True):
        tb_logger = TensorBoardLogger(
            save_dir=path_manager.logs_dir,
            name='tensorboard',
            log_graph=True
        )
        logger.info(f"TensorBoard logging: {tb_logger.log_dir}")
    else:
        tb_logger = None
    
    # 训练器配置
    training_config = config.get('training', {})
    hardware_config = config.get('hardware', {})
    debug_config = config.get('debug', {})
    
    trainer_kwargs = {
        'max_epochs': training_config.get('epochs', 20),
        'callbacks': callbacks,
        'logger': tb_logger,
        'enable_progress_bar': True,
        'log_every_n_steps': tb_config.get('log_every_n_steps', 10),
        'enable_model_summary': True,
        'deterministic': config.get('experiment', {}).get('deterministic', True)
    }
    
    # 硬件配置
    accelerator = hardware_config.get('accelerator', 'auto')
    devices = hardware_config.get('devices', 1)
    precision = hardware_config.get('precision', '16-mixed')
    
    if accelerator == 'auto':
        if torch.cuda.is_available():
            trainer_kwargs['accelerator'] = 'gpu'
            trainer_kwargs['devices'] = devices
        else:
            trainer_kwargs['accelerator'] = 'cpu'
            logger.info("No GPU detected, using CPU training")
    else:
        trainer_kwargs['accelerator'] = accelerator
        if accelerator == 'gpu':
            trainer_kwargs['devices'] = devices
    
    trainer_kwargs['precision'] = precision
    
    # 梯度累积
    accumulate_grad_batches = hardware_config.get('accumulate_grad_batches', 1)
    if accumulate_grad_batches > 1:
        trainer_kwargs['accumulate_grad_batches'] = accumulate_grad_batches
    
    # 梯度裁剪
    grad_clip_config = training_config.get('gradient_clipping', {})
    if grad_clip_config.get('enabled', True):
        trainer_kwargs['gradient_clip_val'] = grad_clip_config.get('max_norm', 1.0)
    
    # 调试选项
    if debug_config.get('fast_dev_run', False) or getattr(args, 'fast_dev_run', False):
        trainer_kwargs['fast_dev_run'] = True
        logger.info("Fast development run mode enabled")
    
    if debug_config.get('overfit_batches', 0) > 0:
        trainer_kwargs['overfit_batches'] = debug_config['overfit_batches']
    
    # 创建训练器
    trainer = pl.Trainer(**trainer_kwargs)
    
    logger.info("Trainer configuration:")
    logger.info(f"   Accelerator: {trainer_kwargs.get('accelerator', 'auto')}")
    logger.info(f"   Devices: {trainer_kwargs.get('devices', 'auto')}")
    logger.info(f"   Precision: {precision}")
    logger.info(f"   Max epochs: {trainer_kwargs['max_epochs']}")
    
    return trainer


def main():
    """主函数"""
    
    try:
        # 解析参数
        args = parse_arguments()
        
        # 创建配置
        config = create_config_from_args(args)
        
        # 设置实验环境
        path_manager = setup_experiment(config, args)
        
        # 准备数据
        train_loader, val_loader, test_loader = prepare_data(config, args)
        
        # 创建模型
        model = create_model(config)
        
        # 设置回调
        callbacks = setup_callbacks(config, path_manager)
        
        # 设置训练器
        trainer = setup_logger_and_trainer(config, path_manager, callbacks, args)
        
        # 开始训练
        logger.info("Starting model training...")
        trainer.fit(model, train_loader, val_loader)
        
        # 测试模型
        if test_loader is not None:
            logger.info("Starting model testing...")
            trainer.test(model, test_loader)
        
        logger.info("Training completed!")
        logger.info(f"Results saved in: {path_manager.base_dir}")
        
        # 打印最佳检查点路径
        if trainer.checkpoint_callback and trainer.checkpoint_callback.best_model_path:
            logger.info(f"Best model: {trainer.checkpoint_callback.best_model_path}")
        
    except KeyboardInterrupt:
        logger.info("Training interrupted by user")
    except Exception as e:
        logger.error(f"Training failed: {e}", exc_info=True)
        raise


if __name__ == '__main__':
    main()
