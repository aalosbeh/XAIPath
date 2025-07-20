"""
Training script for XAIPath framework.

This module provides:
- Complete training pipeline with validation
- Model checkpointing and early stopping
- Comprehensive evaluation metrics
- Learning rate scheduling
- Tensorboard logging
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import pandas as pd
import os
import json
import time
from datetime import datetime
from typing import Dict, List, Tuple, Optional
import argparse
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

from xaipath_model import XAIPathModel
from dataset import BacterialDataset, create_data_loaders, create_stratified_splits, create_synthetic_metadata
from utils import EarlyStopping, ModelCheckpoint, MetricsTracker


class XAIPathTrainer:
    """
    Trainer class for XAIPath model with comprehensive training and evaluation.
    """
    
    def __init__(self, 
                 model: XAIPathModel,
                 train_loader: torch.utils.data.DataLoader,
                 val_loader: torch.utils.data.DataLoader,
                 test_loader: torch.utils.data.DataLoader,
                 device: torch.device,
                 config: Dict):
        """
        Initialize trainer.
        
        Args:
            model: XAIPath model
            train_loader: Training data loader
            val_loader: Validation data loader
            test_loader: Test data loader
            device: Computing device
            config: Training configuration
        """
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.device = device
        self.config = config
        
        # Initialize optimizer
        self.optimizer = self._create_optimizer()
        
        # Initialize scheduler
        self.scheduler = self._create_scheduler()
        
        # Initialize loss function with class weights
        self.criterion = self._create_criterion()
        
        # Initialize tracking utilities
        self.metrics_tracker = MetricsTracker()
        self.early_stopping = EarlyStopping(
            patience=config.get('early_stopping_patience', 15),
            min_delta=config.get('early_stopping_min_delta', 0.001)
        )
        self.model_checkpoint = ModelCheckpoint(
            save_dir=config.get('checkpoint_dir', 'checkpoints'),
            save_best_only=True
        )
        
        # Initialize tensorboard writer
        self.writer = SummaryWriter(
            log_dir=os.path.join(config.get('log_dir', 'logs'), 
                               datetime.now().strftime('%Y%m%d_%H%M%S'))
        )
        
        # Training state
        self.current_epoch = 0
        self.best_val_f1 = 0.0
        self.training_history = {
            'train_loss': [], 'train_acc': [], 'train_f1': [],
            'val_loss': [], 'val_acc': [], 'val_f1': [],
            'learning_rate': []
        }
    
    def _create_optimizer(self) -> optim.Optimizer:
        """Create optimizer based on configuration."""
        optimizer_name = self.config.get('optimizer', 'adam').lower()
        lr = self.config.get('learning_rate', 0.001)
        weight_decay = self.config.get('weight_decay', 1e-4)
        
        if optimizer_name == 'adam':
            return optim.Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        elif optimizer_name == 'sgd':
            momentum = self.config.get('momentum', 0.9)
            return optim.SGD(self.model.parameters(), lr=lr, 
                           momentum=momentum, weight_decay=weight_decay)
        elif optimizer_name == 'adamw':
            return optim.AdamW(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        else:
            raise ValueError(f"Unsupported optimizer: {optimizer_name}")
    
    def _create_scheduler(self) -> Optional[optim.lr_scheduler._LRScheduler]:
        """Create learning rate scheduler."""
        scheduler_name = self.config.get('scheduler', 'step').lower()
        
        if scheduler_name == 'step':
            step_size = self.config.get('scheduler_step_size', 30)
            gamma = self.config.get('scheduler_gamma', 0.1)
            return optim.lr_scheduler.StepLR(self.optimizer, step_size=step_size, gamma=gamma)
        elif scheduler_name == 'cosine':
            T_max = self.config.get('num_epochs', 100)
            return optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=T_max)
        elif scheduler_name == 'plateau':
            return optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer, mode='max', factor=0.5, patience=10, verbose=True
            )
        elif scheduler_name == 'none':
            return None
        else:
            raise ValueError(f"Unsupported scheduler: {scheduler_name}")
    
    def _create_criterion(self) -> nn.Module:
        """Create loss function with class weights."""
        # Get class weights from training dataset
        if hasattr(self.train_loader.dataset, 'get_class_weights'):
            class_weights = self.train_loader.dataset.get_class_weights()
            class_weights = class_weights.to(self.device)
        else:
            class_weights = None
        
        return nn.CrossEntropyLoss(weight=class_weights)
    
    def train_epoch(self) -> Dict[str, float]:
        """
        Train for one epoch.
        
        Returns:
            Dictionary with training metrics
        """
        self.model.train()
        
        total_loss = 0.0
        total_cls_loss = 0.0
        total_temp_loss = 0.0
        total_env_loss = 0.0
        all_predictions = []
        all_targets = []
        
        for batch_idx, batch in enumerate(self.train_loader):
            # Move data to device
            images = batch['image'].to(self.device)
            labels = batch['label'].to(self.device)
            time = batch['time'].to(self.device)
            env_condition = batch['env_condition'].to(self.device)
            
            # Zero gradients
            self.optimizer.zero_grad()
            
            # Forward pass
            outputs = self.model(images, time, env_condition)
            
            # Compute loss
            losses = self.model.compute_loss(outputs, labels, time, env_condition)
            
            # Backward pass
            losses['total_loss'].backward()
            
            # Gradient clipping
            if self.config.get('gradient_clipping', False):
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), 
                    self.config.get('max_grad_norm', 1.0)
                )
            
            # Update parameters
            self.optimizer.step()
            
            # Accumulate losses
            total_loss += losses['total_loss'].item()
            total_cls_loss += losses['cls_loss'].item()
            total_temp_loss += losses['temp_loss'].item()
            total_env_loss += losses['env_loss'].item()
            
            # Accumulate predictions
            predictions = torch.argmax(outputs['logits'], dim=1)
            all_predictions.extend(predictions.cpu().numpy())
            all_targets.extend(labels.cpu().numpy())
            
            # Log batch metrics
            if batch_idx % self.config.get('log_interval', 100) == 0:
                print(f'Epoch {self.current_epoch}, Batch {batch_idx}/{len(self.train_loader)}, '
                      f'Loss: {losses["total_loss"].item():.4f}')
        
        # Compute epoch metrics
        avg_loss = total_loss / len(self.train_loader)
        avg_cls_loss = total_cls_loss / len(self.train_loader)
        avg_temp_loss = total_temp_loss / len(self.train_loader)
        avg_env_loss = total_env_loss / len(self.train_loader)
        
        accuracy = np.mean(np.array(all_predictions) == np.array(all_targets))
        precision = precision_score(all_targets, all_predictions, average='weighted', zero_division=0)
        recall = recall_score(all_targets, all_predictions, average='weighted', zero_division=0)
        f1 = f1_score(all_targets, all_predictions, average='weighted', zero_division=0)
        
        return {
            'loss': avg_loss,
            'cls_loss': avg_cls_loss,
            'temp_loss': avg_temp_loss,
            'env_loss': avg_env_loss,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1
        }
    
    def validate_epoch(self) -> Dict[str, float]:
        """
        Validate for one epoch.
        
        Returns:
            Dictionary with validation metrics
        """
        self.model.eval()
        
        total_loss = 0.0
        all_predictions = []
        all_targets = []
        all_attention_maps = []
        
        with torch.no_grad():
            for batch in self.val_loader:
                # Move data to device
                images = batch['image'].to(self.device)
                labels = batch['label'].to(self.device)
                time = batch['time'].to(self.device)
                env_condition = batch['env_condition'].to(self.device)
                
                # Forward pass
                outputs = self.model(images, time, env_condition)
                
                # Compute loss
                losses = self.model.compute_loss(outputs, labels, time, env_condition)
                
                # Accumulate losses
                total_loss += losses['total_loss'].item()
                
                # Accumulate predictions
                predictions = torch.argmax(outputs['logits'], dim=1)
                all_predictions.extend(predictions.cpu().numpy())
                all_targets.extend(labels.cpu().numpy())
                
                # Store attention maps for analysis
                all_attention_maps.extend(outputs['attention_maps'].cpu().numpy())
        
        # Compute epoch metrics
        avg_loss = total_loss / len(self.val_loader)
        accuracy = np.mean(np.array(all_predictions) == np.array(all_targets))
        precision = precision_score(all_targets, all_predictions, average='weighted', zero_division=0)
        recall = recall_score(all_targets, all_predictions, average='weighted', zero_division=0)
        f1 = f1_score(all_targets, all_predictions, average='weighted', zero_division=0)
        
        # Compute confusion matrix
        cm = confusion_matrix(all_targets, all_predictions)
        
        return {
            'loss': avg_loss,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'confusion_matrix': cm,
            'predictions': all_predictions,
            'targets': all_targets,
            'attention_maps': all_attention_maps
        }
    
    def train(self) -> Dict[str, List]:
        """
        Complete training loop.
        
        Returns:
            Training history dictionary
        """
        print(f"Starting training for {self.config.get('num_epochs', 100)} epochs...")
        print(f"Device: {self.device}")
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        
        start_time = time.time()
        
        for epoch in range(self.config.get('num_epochs', 100)):
            self.current_epoch = epoch
            
            # Training phase
            train_metrics = self.train_epoch()
            
            # Validation phase
            val_metrics = self.validate_epoch()
            
            # Update learning rate
            if self.scheduler is not None:
                if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_metrics['f1'])
                else:
                    self.scheduler.step()
            
            # Get current learning rate
            current_lr = self.optimizer.param_groups[0]['lr']
            
            # Update training history
            self.training_history['train_loss'].append(train_metrics['loss'])
            self.training_history['train_acc'].append(train_metrics['accuracy'])
            self.training_history['train_f1'].append(train_metrics['f1'])
            self.training_history['val_loss'].append(val_metrics['loss'])
            self.training_history['val_acc'].append(val_metrics['accuracy'])
            self.training_history['val_f1'].append(val_metrics['f1'])
            self.training_history['learning_rate'].append(current_lr)
            
            # Log to tensorboard
            self.writer.add_scalar('Loss/Train', train_metrics['loss'], epoch)
            self.writer.add_scalar('Loss/Validation', val_metrics['loss'], epoch)
            self.writer.add_scalar('Accuracy/Train', train_metrics['accuracy'], epoch)
            self.writer.add_scalar('Accuracy/Validation', val_metrics['accuracy'], epoch)
            self.writer.add_scalar('F1/Train', train_metrics['f1'], epoch)
            self.writer.add_scalar('F1/Validation', val_metrics['f1'], epoch)
            self.writer.add_scalar('Learning_Rate', current_lr, epoch)
            
            # Print epoch results
            print(f"Epoch {epoch+1}/{self.config.get('num_epochs', 100)}:")
            print(f"  Train - Loss: {train_metrics['loss']:.4f}, "
                  f"Acc: {train_metrics['accuracy']:.4f}, F1: {train_metrics['f1']:.4f}")
            print(f"  Val   - Loss: {val_metrics['loss']:.4f}, "
                  f"Acc: {val_metrics['accuracy']:.4f}, F1: {val_metrics['f1']:.4f}")
            print(f"  LR: {current_lr:.6f}")
            
            # Model checkpointing
            is_best = val_metrics['f1'] > self.best_val_f1
            if is_best:
                self.best_val_f1 = val_metrics['f1']
            
            self.model_checkpoint.save(
                model=self.model,
                optimizer=self.optimizer,
                scheduler=self.scheduler,
                epoch=epoch,
                metrics=val_metrics,
                is_best=is_best
            )
            
            # Early stopping
            if self.early_stopping.should_stop(val_metrics['f1']):
                print(f"Early stopping triggered at epoch {epoch+1}")
                break
        
        training_time = time.time() - start_time
        print(f"Training completed in {training_time:.2f} seconds")
        print(f"Best validation F1: {self.best_val_f1:.4f}")
        
        # Close tensorboard writer
        self.writer.close()
        
        return self.training_history
    
    def evaluate(self, data_loader: torch.utils.data.DataLoader, 
                split_name: str = 'test') -> Dict[str, float]:
        """
        Evaluate model on given dataset.
        
        Args:
            data_loader: Data loader for evaluation
            split_name: Name of the split being evaluated
            
        Returns:
            Dictionary with evaluation metrics
        """
        print(f"Evaluating on {split_name} set...")
        
        self.model.eval()
        
        all_predictions = []
        all_targets = []
        all_probabilities = []
        all_attention_maps = []
        
        with torch.no_grad():
            for batch in data_loader:
                # Move data to device
                images = batch['image'].to(self.device)
                labels = batch['label'].to(self.device)
                time = batch['time'].to(self.device)
                env_condition = batch['env_condition'].to(self.device)
                
                # Forward pass
                outputs = self.model(images, time, env_condition)
                
                # Get predictions and probabilities
                probabilities = torch.softmax(outputs['logits'], dim=1)
                predictions = torch.argmax(outputs['logits'], dim=1)
                
                # Accumulate results
                all_predictions.extend(predictions.cpu().numpy())
                all_targets.extend(labels.cpu().numpy())
                all_probabilities.extend(probabilities.cpu().numpy())
                all_attention_maps.extend(outputs['attention_maps'].cpu().numpy())
        
        # Compute metrics
        accuracy = np.mean(np.array(all_predictions) == np.array(all_targets))
        precision = precision_score(all_targets, all_predictions, average='weighted', zero_division=0)
        recall = recall_score(all_targets, all_predictions, average='weighted', zero_division=0)
        f1 = f1_score(all_targets, all_predictions, average='weighted', zero_division=0)
        
        # Per-class metrics
        precision_per_class = precision_score(all_targets, all_predictions, average=None, zero_division=0)
        recall_per_class = recall_score(all_targets, all_predictions, average=None, zero_division=0)
        f1_per_class = f1_score(all_targets, all_predictions, average=None, zero_division=0)
        
        # Confusion matrix
        cm = confusion_matrix(all_targets, all_predictions)
        
        print(f"{split_name.capitalize()} Results:")
        print(f"  Accuracy: {accuracy:.4f}")
        print(f"  Precision: {precision:.4f}")
        print(f"  Recall: {recall:.4f}")
        print(f"  F1-Score: {f1:.4f}")
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'precision_per_class': precision_per_class.tolist(),
            'recall_per_class': recall_per_class.tolist(),
            'f1_per_class': f1_per_class.tolist(),
            'confusion_matrix': cm.tolist(),
            'predictions': all_predictions,
            'targets': all_targets,
            'probabilities': all_probabilities,
            'attention_maps': all_attention_maps
        }
    
    def save_training_plots(self, save_dir: str = 'figures'):
        """
        Save training and validation curves.
        
        Args:
            save_dir: Directory to save plots
        """
        os.makedirs(save_dir, exist_ok=True)
        
        # Create training curves plot
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Loss curves
        axes[0, 0].plot(self.training_history['train_loss'], label='Train Loss', color='blue')
        axes[0, 0].plot(self.training_history['val_loss'], label='Validation Loss', color='red')
        axes[0, 0].set_title('Training and Validation Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # Accuracy curves
        axes[0, 1].plot(self.training_history['train_acc'], label='Train Accuracy', color='blue')
        axes[0, 1].plot(self.training_history['val_acc'], label='Validation Accuracy', color='red')
        axes[0, 1].set_title('Training and Validation Accuracy')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Accuracy')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        # F1 curves
        axes[1, 0].plot(self.training_history['train_f1'], label='Train F1', color='blue')
        axes[1, 0].plot(self.training_history['val_f1'], label='Validation F1', color='red')
        axes[1, 0].set_title('Training and Validation F1-Score')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('F1-Score')
        axes[1, 0].legend()
        axes[1, 0].grid(True)
        
        # Learning rate
        axes[1, 1].plot(self.training_history['learning_rate'], color='green')
        axes[1, 1].set_title('Learning Rate Schedule')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Learning Rate')
        axes[1, 1].set_yscale('log')
        axes[1, 1].grid(True)
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'training_curves.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Training curves saved to {save_dir}/training_curves.png")


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description='Train XAIPath model')
    parser.add_argument('--config', type=str, default='config.json',
                       help='Path to configuration file')
    parser.add_argument('--data_dir', type=str, default='data',
                       help='Path to data directory')
    parser.add_argument('--num_epochs', type=int, default=100,
                       help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                       help='Learning rate')
    parser.add_argument('--device', type=str, default='auto',
                       help='Device to use (cuda/cpu/auto)')
    
    args = parser.parse_args()
    
    # Load configuration
    if os.path.exists(args.config):
        with open(args.config, 'r') as f:
            config = json.load(f)
    else:
        config = {}
    
    # Override config with command line arguments
    config.update({
        'data_dir': args.data_dir,
        'num_epochs': args.num_epochs,
        'batch_size': args.batch_size,
        'learning_rate': args.learning_rate
    })
    
    # Set device
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    
    print(f"Using device: {device}")
    
    # Create synthetic data for demonstration
    print("Creating synthetic dataset...")
    metadata = create_synthetic_metadata(num_samples=2847, output_file='metadata.csv')
    splits = create_stratified_splits(metadata)
    
    # Create data loaders
    data_loaders = create_data_loaders(
        data_dir=args.data_dir,
        metadata_splits=splits,
        batch_size=config['batch_size'],
        num_workers=config.get('num_workers', 4)
    )
    
    # Create model
    model = XAIPathModel(
        num_classes=3,
        temporal_dim=config.get('temporal_dim', 128),
        env_dim=config.get('env_dim', 64),
        lambda_temp=config.get('lambda_temp', 0.1),
        lambda_env=config.get('lambda_env', 0.05)
    )
    
    # Create trainer
    trainer = XAIPathTrainer(
        model=model,
        train_loader=data_loaders['train'],
        val_loader=data_loaders['val'],
        test_loader=data_loaders['test'],
        device=device,
        config=config
    )
    
    # Train model
    training_history = trainer.train()
    
    # Evaluate on test set
    test_results = trainer.evaluate(data_loaders['test'], 'test')
    
    # Save training plots
    trainer.save_training_plots()
    
    # Save results
    results = {
        'training_history': training_history,
        'test_results': test_results,
        'config': config
    }
    
    with open('training_results.json', 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print("Training completed successfully!")


if __name__ == "__main__":
    main()

