"""
Utility functions for XAIPath training and evaluation.

This module provides:
- Early stopping mechanism
- Model checkpointing
- Metrics tracking
- Visualization utilities
- SHAP and LIME explanation generators
"""

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import json
from typing import Dict, List, Tuple, Optional, Any
from sklearn.metrics import confusion_matrix, classification_report
import cv2
from PIL import Image
import shap
from lime import lime_image
from lime.wrappers.scikit_image import SegmentationAlgorithm


class EarlyStopping:
    """
    Early stopping mechanism to prevent overfitting.
    """
    
    def __init__(self, patience: int = 10, min_delta: float = 0.001, mode: str = 'max'):
        """
        Initialize early stopping.
        
        Args:
            patience: Number of epochs to wait before stopping
            min_delta: Minimum change to qualify as improvement
            mode: 'max' for metrics to maximize, 'min' for metrics to minimize
        """
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        
    def should_stop(self, score: float) -> bool:
        """
        Check if training should stop.
        
        Args:
            score: Current metric score
            
        Returns:
            True if training should stop
        """
        if self.best_score is None:
            self.best_score = score
            return False
        
        if self.mode == 'max':
            improved = score > self.best_score + self.min_delta
        else:
            improved = score < self.best_score - self.min_delta
        
        if improved:
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
        
        return self.counter >= self.patience


class ModelCheckpoint:
    """
    Model checkpointing utility.
    """
    
    def __init__(self, save_dir: str = 'checkpoints', save_best_only: bool = True):
        """
        Initialize model checkpoint.
        
        Args:
            save_dir: Directory to save checkpoints
            save_best_only: Whether to save only the best model
        """
        self.save_dir = save_dir
        self.save_best_only = save_best_only
        self.best_score = None
        
        os.makedirs(save_dir, exist_ok=True)
    
    def save(self, model: nn.Module, optimizer: torch.optim.Optimizer,
             scheduler: Optional[torch.optim.lr_scheduler._LRScheduler],
             epoch: int, metrics: Dict[str, float], is_best: bool = False):
        """
        Save model checkpoint.
        
        Args:
            model: Model to save
            optimizer: Optimizer state
            scheduler: Scheduler state
            epoch: Current epoch
            metrics: Current metrics
            is_best: Whether this is the best model so far
        """
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
            'metrics': metrics
        }
        
        # Save latest checkpoint
        if not self.save_best_only:
            torch.save(checkpoint, os.path.join(self.save_dir, 'latest_checkpoint.pth'))
        
        # Save best checkpoint
        if is_best:
            torch.save(checkpoint, os.path.join(self.save_dir, 'best_checkpoint.pth'))
            self.best_score = metrics.get('f1', metrics.get('accuracy', 0))
            
            # Save model for inference
            torch.save(model.state_dict(), os.path.join(self.save_dir, 'best_model.pth'))
    
    def load(self, model: nn.Module, optimizer: torch.optim.Optimizer = None,
             scheduler: torch.optim.lr_scheduler._LRScheduler = None,
             checkpoint_path: str = None) -> Dict[str, Any]:
        """
        Load model checkpoint.
        
        Args:
            model: Model to load state into
            optimizer: Optimizer to load state into
            scheduler: Scheduler to load state into
            checkpoint_path: Path to checkpoint file
            
        Returns:
            Checkpoint dictionary
        """
        if checkpoint_path is None:
            checkpoint_path = os.path.join(self.save_dir, 'best_checkpoint.pth')
        
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        model.load_state_dict(checkpoint['model_state_dict'])
        
        if optimizer and 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if scheduler and 'scheduler_state_dict' in checkpoint and checkpoint['scheduler_state_dict']:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        return checkpoint


class MetricsTracker:
    """
    Utility for tracking and analyzing training metrics.
    """
    
    def __init__(self):
        self.metrics_history = {}
    
    def update(self, metrics: Dict[str, float], phase: str = 'train'):
        """
        Update metrics history.
        
        Args:
            metrics: Dictionary of metrics
            phase: Training phase (train/val/test)
        """
        if phase not in self.metrics_history:
            self.metrics_history[phase] = {}
        
        for key, value in metrics.items():
            if key not in self.metrics_history[phase]:
                self.metrics_history[phase][key] = []
            self.metrics_history[phase][key].append(value)
    
    def get_best_epoch(self, metric: str = 'f1', phase: str = 'val') -> int:
        """
        Get epoch with best metric value.
        
        Args:
            metric: Metric name
            phase: Training phase
            
        Returns:
            Best epoch number
        """
        if phase not in self.metrics_history or metric not in self.metrics_history[phase]:
            return 0
        
        values = self.metrics_history[phase][metric]
        return np.argmax(values)
    
    def plot_metrics(self, save_path: str = None):
        """
        Plot training metrics.
        
        Args:
            save_path: Path to save plot
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        metrics_to_plot = ['loss', 'accuracy', 'precision', 'f1']
        
        for i, metric in enumerate(metrics_to_plot):
            ax = axes[i // 2, i % 2]
            
            for phase in ['train', 'val']:
                if phase in self.metrics_history and metric in self.metrics_history[phase]:
                    values = self.metrics_history[phase][metric]
                    ax.plot(values, label=f'{phase.capitalize()} {metric.capitalize()}')
            
            ax.set_title(f'{metric.capitalize()} Over Time')
            ax.set_xlabel('Epoch')
            ax.set_ylabel(metric.capitalize())
            ax.legend()
            ax.grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        else:
            plt.show()
        
        plt.close()


class VisualizationUtils:
    """
    Utilities for creating visualizations and plots.
    """
    
    @staticmethod
    def plot_confusion_matrix(cm: np.ndarray, class_names: List[str],
                            title: str = 'Confusion Matrix', save_path: str = None):
        """
        Plot confusion matrix.
        
        Args:
            cm: Confusion matrix
            class_names: List of class names
            title: Plot title
            save_path: Path to save plot
        """
        plt.figure(figsize=(8, 6))
        
        # Normalize confusion matrix
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        
        # Create heatmap
        sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues',
                   xticklabels=class_names, yticklabels=class_names)
        
        plt.title(title)
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        else:
            plt.show()
        
        plt.close()
    
    @staticmethod
    def plot_attention_maps(images: np.ndarray, attention_maps: np.ndarray,
                          predictions: np.ndarray, targets: np.ndarray,
                          class_names: List[str], save_path: str = None):
        """
        Plot attention maps overlaid on images.
        
        Args:
            images: Original images
            attention_maps: Attention maps
            predictions: Model predictions
            targets: Ground truth labels
            class_names: List of class names
            save_path: Path to save plot
        """
        num_samples = min(8, len(images))
        fig, axes = plt.subplots(2, num_samples, figsize=(20, 8))
        
        for i in range(num_samples):
            # Original image
            img = images[i].transpose(1, 2, 0)
            img = (img - img.min()) / (img.max() - img.min())  # Normalize
            
            axes[0, i].imshow(img)
            axes[0, i].set_title(f'True: {class_names[targets[i]]}\nPred: {class_names[predictions[i]]}')
            axes[0, i].axis('off')
            
            # Attention map
            attention = attention_maps[i]
            attention_resized = cv2.resize(attention, (img.shape[1], img.shape[0]))
            
            axes[1, i].imshow(img, alpha=0.7)
            axes[1, i].imshow(attention_resized, alpha=0.5, cmap='jet')
            axes[1, i].set_title('Attention Map')
            axes[1, i].axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        else:
            plt.show()
        
        plt.close()
    
    @staticmethod
    def plot_temporal_analysis(results_by_time: Dict[float, Dict[str, float]],
                             save_path: str = None):
        """
        Plot performance analysis across temporal dimensions.
        
        Args:
            results_by_time: Dictionary mapping time points to metrics
            save_path: Path to save plot
        """
        time_points = sorted(results_by_time.keys())
        metrics = ['accuracy', 'precision', 'recall', 'f1']
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        for i, metric in enumerate(metrics):
            ax = axes[i // 2, i % 2]
            
            values = [results_by_time[t][metric] for t in time_points]
            ax.plot(time_points, values, marker='o', linewidth=2, markersize=6)
            
            ax.set_title(f'{metric.capitalize()} vs Time')
            ax.set_xlabel('Time (hours)')
            ax.set_ylabel(metric.capitalize())
            ax.grid(True, alpha=0.3)
            ax.set_ylim(0, 1)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        else:
            plt.show()
        
        plt.close()
    
    @staticmethod
    def plot_environmental_analysis(results_by_env: Dict[int, Dict[str, float]],
                                  save_path: str = None):
        """
        Plot performance analysis across environmental conditions.
        
        Args:
            results_by_env: Dictionary mapping environmental conditions to metrics
            save_path: Path to save plot
        """
        env_labels = ['Without Onion', 'With Onion']
        metrics = ['accuracy', 'precision', 'recall', 'f1']
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        x = np.arange(len(env_labels))
        width = 0.2
        
        for i, metric in enumerate(metrics):
            values = [results_by_env[env][metric] for env in sorted(results_by_env.keys())]
            ax.bar(x + i * width, values, width, label=metric.capitalize())
        
        ax.set_xlabel('Environmental Condition')
        ax.set_ylabel('Performance')
        ax.set_title('Performance Across Environmental Conditions')
        ax.set_xticks(x + width * 1.5)
        ax.set_xticklabels(env_labels)
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 1)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        else:
            plt.show()
        
        plt.close()


class SHAPExplainer:
    """
    SHAP-based explanation generator for XAIPath model.
    """
    
    def __init__(self, model: nn.Module, background_data: torch.Tensor):
        """
        Initialize SHAP explainer.
        
        Args:
            model: Trained model
            background_data: Background dataset for SHAP
        """
        self.model = model
        self.explainer = shap.DeepExplainer(model, background_data)
    
    def explain(self, images: torch.Tensor, time: torch.Tensor,
                env_condition: torch.Tensor) -> np.ndarray:
        """
        Generate SHAP explanations.
        
        Args:
            images: Input images
            time: Time values
            env_condition: Environmental conditions
            
        Returns:
            SHAP values
        """
        # Create wrapper function for SHAP
        def model_wrapper(x):
            with torch.no_grad():
                outputs = self.model(x, time, env_condition)
                return outputs['logits']
        
        # Compute SHAP values
        shap_values = self.explainer.shap_values(images)
        
        return shap_values


class LIMEExplainer:
    """
    LIME-based explanation generator for XAIPath model.
    """
    
    def __init__(self, model: nn.Module, class_names: List[str]):
        """
        Initialize LIME explainer.
        
        Args:
            model: Trained model
            class_names: List of class names
        """
        self.model = model
        self.class_names = class_names
        self.explainer = lime_image.LimeImageExplainer()
    
    def explain(self, image: np.ndarray, time: float, env_condition: int,
                num_samples: int = 1000) -> Any:
        """
        Generate LIME explanation for a single image.
        
        Args:
            image: Input image (H, W, C)
            time: Time value
            env_condition: Environmental condition
            num_samples: Number of samples for LIME
            
        Returns:
            LIME explanation
        """
        def predict_fn(images):
            """Prediction function for LIME."""
            batch_size = len(images)
            
            # Convert to tensor
            images_tensor = torch.tensor(images.transpose(0, 3, 1, 2), dtype=torch.float32)
            time_tensor = torch.tensor([time] * batch_size, dtype=torch.float32)
            env_tensor = torch.tensor([env_condition] * batch_size, dtype=torch.long)
            
            # Get predictions
            with torch.no_grad():
                outputs = self.model(images_tensor, time_tensor, env_tensor)
                probabilities = torch.softmax(outputs['logits'], dim=1)
            
            return probabilities.numpy()
        
        # Generate explanation
        explanation = self.explainer.explain_instance(
            image,
            predict_fn,
            top_labels=len(self.class_names),
            hide_color=0,
            num_samples=num_samples,
            segmentation_fn=SegmentationAlgorithm('quickshift', kernel_size=4, max_dist=200, ratio=0.2)
        )
        
        return explanation


def create_ablation_study_results():
    """
    Create synthetic ablation study results for demonstration.
    
    Returns:
        Dictionary with ablation study results
    """
    results = {
        'Full XAIPath': {
            'accuracy': 0.934,
            'precision': 0.947,
            'recall': 0.913,
            'f1': 0.929,
            'explanation_quality': 0.917
        },
        'w/o Temporal': {
            'accuracy': 0.866,
            'precision': 0.879,
            'recall': 0.851,
            'f1': 0.865,
            'explanation_quality': 0.844
        },
        'w/o Environmental': {
            'accuracy': 0.892,
            'precision': 0.905,
            'recall': 0.878,
            'f1': 0.891,
            'explanation_quality': 0.869
        },
        'w/o Both': {
            'accuracy': 0.815,
            'precision': 0.828,
            'recall': 0.801,
            'f1': 0.814,
            'explanation_quality': 0.784
        }
    }
    
    return results


def create_temporal_environmental_results():
    """
    Create synthetic temporal and environmental analysis results.
    
    Returns:
        Tuple of (temporal_results, environmental_results)
    """
    # Temporal results (by time point)
    temporal_results = {}
    time_points = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0]
    
    for t in time_points:
        # Simulate performance improvement over time
        base_acc = 0.85 + 0.08 * (t / 4.0)  # Accuracy improves with time
        temporal_results[t] = {
            'accuracy': base_acc + np.random.normal(0, 0.02),
            'precision': base_acc + 0.02 + np.random.normal(0, 0.02),
            'recall': base_acc - 0.01 + np.random.normal(0, 0.02),
            'f1': base_acc + 0.01 + np.random.normal(0, 0.02)
        }
    
    # Environmental results
    environmental_results = {
        0: {  # Without onion
            'accuracy': 0.934,
            'precision': 0.947,
            'recall': 0.913,
            'f1': 0.929
        },
        1: {  # With onion
            'accuracy': 0.921,
            'precision': 0.935,
            'recall': 0.908,
            'f1': 0.921
        }
    }
    
    return temporal_results, environmental_results


if __name__ == "__main__":
    # Test utility functions
    
    # Test early stopping
    early_stopping = EarlyStopping(patience=5, min_delta=0.001)
    
    scores = [0.8, 0.85, 0.87, 0.86, 0.86, 0.85, 0.84, 0.83]
    for i, score in enumerate(scores):
        should_stop = early_stopping.should_stop(score)
        print(f"Epoch {i+1}, Score: {score:.3f}, Should stop: {should_stop}")
        if should_stop:
            break
    
    # Test visualization utilities
    vis_utils = VisualizationUtils()
    
    # Create sample confusion matrix
    cm = np.array([[50, 2, 1], [3, 45, 2], [1, 1, 48]])
    class_names = ['Background', 'Salmonella', 'Mixed Culture']
    
    vis_utils.plot_confusion_matrix(cm, class_names, save_path='test_confusion_matrix.png')
    
    # Test temporal analysis
    temporal_results, environmental_results = create_temporal_environmental_results()
    vis_utils.plot_temporal_analysis(temporal_results, save_path='test_temporal_analysis.png')
    vis_utils.plot_environmental_analysis(environmental_results, save_path='test_environmental_analysis.png')
    
    print("Utility functions tested successfully!")

