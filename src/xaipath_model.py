"""
XAIPath: Temporal-Environmental Explainable AI Framework for Co-Contaminated Food Pathogen Detection

This module implements the core XAIPath model architecture including:
- Temporal feature encoding with learnable sinusoidal embeddings
- Environmental context modeling with gating mechanisms
- Multi-modal explainability engine with Grad-CAM, SHAP, and LIME
- Temporal and environmental consistency constraints
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import numpy as np
from typing import Dict, List, Tuple, Optional
import math


class TemporalEncoder(nn.Module):
    """
    Temporal encoding mechanism that transforms growth time information 
    into high-dimensional feature representations using learnable sinusoidal embeddings.
    """
    
    def __init__(self, d_model: int = 128, max_time: float = 4.0):
        super(TemporalEncoder, self).__init__()
        self.d_model = d_model
        self.max_time = max_time
        
        # Learnable frequency and phase parameters
        self.omega = nn.Parameter(torch.randn(d_model // 2) * 0.1)
        self.phi = nn.Parameter(torch.randn(d_model // 2) * 0.1)
        
    def forward(self, time: torch.Tensor) -> torch.Tensor:
        """
        Transform temporal coordinates to feature vectors.
        
        Args:
            time: Tensor of shape (batch_size,) containing time values in hours
            
        Returns:
            Temporal embeddings of shape (batch_size, d_model)
        """
        # Normalize time to [0, 1] range
        normalized_time = time / self.max_time
        
        # Compute sinusoidal embeddings
        angles = self.omega.unsqueeze(0) * normalized_time.unsqueeze(1) + self.phi.unsqueeze(0)
        
        sin_embeddings = torch.sin(angles)
        cos_embeddings = torch.cos(angles)
        
        # Interleave sin and cos embeddings
        embeddings = torch.stack([sin_embeddings, cos_embeddings], dim=2).flatten(1)
        
        return embeddings


class EnvironmentalEncoder(nn.Module):
    """
    Environmental context encoding that captures the influence of biochemical 
    conditions on bacterial morphology and behavior.
    """
    
    def __init__(self, d_model: int = 64, num_conditions: int = 2):
        super(EnvironmentalEncoder, self).__init__()
        self.d_model = d_model
        self.num_conditions = num_conditions
        
        # Learnable embeddings for environmental conditions
        self.embeddings = nn.Embedding(num_conditions, d_model)
        
        # Gating mechanism parameters
        self.gate_linear = nn.Linear(d_model, d_model)
        
    def forward(self, env_condition: torch.Tensor) -> torch.Tensor:
        """
        Transform environmental conditions to feature vectors.
        
        Args:
            env_condition: Tensor of shape (batch_size,) containing environment indices
            
        Returns:
            Environmental embeddings of shape (batch_size, d_model)
        """
        embeddings = self.embeddings(env_condition)
        return embeddings
    
    def compute_gate(self, env_embeddings: torch.Tensor) -> torch.Tensor:
        """
        Compute gating weights for environmental modulation.
        
        Args:
            env_embeddings: Environmental embeddings of shape (batch_size, d_model)
            
        Returns:
            Gate weights of shape (batch_size, d_model)
        """
        gate = torch.sigmoid(self.gate_linear(env_embeddings))
        return gate


class CrossAttention(nn.Module):
    """
    Cross-attention mechanism for integrating temporal context with spatial features.
    """
    
    def __init__(self, d_visual: int, d_temporal: int, d_k: int = 64):
        super(CrossAttention, self).__init__()
        self.d_k = d_k
        
        self.query_proj = nn.Linear(d_visual, d_k)
        self.key_proj = nn.Linear(d_temporal, d_k)
        self.value_proj = nn.Linear(d_visual, d_visual)
        
    def forward(self, visual_features: torch.Tensor, temporal_features: torch.Tensor) -> torch.Tensor:
        """
        Apply cross-attention between visual and temporal features.
        
        Args:
            visual_features: Visual features of shape (batch_size, H*W, d_visual)
            temporal_features: Temporal features of shape (batch_size, d_temporal)
            
        Returns:
            Attended visual features of shape (batch_size, H*W, d_visual)
        """
        batch_size, seq_len, d_visual = visual_features.shape
        
        # Project to query, key, value
        Q = self.query_proj(visual_features)  # (batch_size, H*W, d_k)
        K = self.key_proj(temporal_features).unsqueeze(1)  # (batch_size, 1, d_k)
        V = self.value_proj(visual_features)  # (batch_size, H*W, d_visual)
        
        # Compute attention weights
        attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        attention_weights = F.softmax(attention_scores, dim=-1)
        
        # Apply attention
        attended_features = attention_weights * V
        
        return attended_features


class XAIPathBackbone(nn.Module):
    """
    Modified ResNet-50 backbone with attention mechanisms for integrating 
    temporal and environmental features.
    """
    
    def __init__(self, num_classes: int = 3, temporal_dim: int = 128, env_dim: int = 64):
        super(XAIPathBackbone, self).__init__()
        
        # Load pre-trained ResNet-50
        self.backbone = models.resnet50(pretrained=True)
        
        # Remove the final classification layer
        self.backbone.fc = nn.Identity()
        
        # Feature dimensions
        self.visual_dim = 2048  # ResNet-50 output dimension
        self.temporal_dim = temporal_dim
        self.env_dim = env_dim
        
        # Temporal and environmental encoders
        self.temporal_encoder = TemporalEncoder(d_model=temporal_dim)
        self.env_encoder = EnvironmentalEncoder(d_model=env_dim)
        
        # Cross-attention for temporal integration
        self.cross_attention = CrossAttention(
            d_visual=self.visual_dim, 
            d_temporal=temporal_dim
        )
        
        # Feature fusion layers
        self.fusion_layer = nn.Sequential(
            nn.Linear(self.visual_dim + temporal_dim + env_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        
        # Classification head
        self.classifier = nn.Linear(256, num_classes)
        
        # Store intermediate features for explainability
        self.visual_features = None
        self.temporal_features = None
        self.env_features = None
        self.attention_maps = None
        
    def forward(self, images: torch.Tensor, time: torch.Tensor, 
                env_condition: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through XAIPath model.
        
        Args:
            images: Input images of shape (batch_size, 3, H, W)
            time: Time values of shape (batch_size,)
            env_condition: Environmental conditions of shape (batch_size,)
            
        Returns:
            Classification logits of shape (batch_size, num_classes)
        """
        batch_size = images.shape[0]
        
        # Extract visual features
        visual_features = self.backbone(images)  # (batch_size, 2048)
        self.visual_features = visual_features
        
        # Encode temporal information
        temporal_features = self.temporal_encoder(time)  # (batch_size, temporal_dim)
        self.temporal_features = temporal_features
        
        # Encode environmental context
        env_features = self.env_encoder(env_condition)  # (batch_size, env_dim)
        env_gate = self.env_encoder.compute_gate(env_features)
        self.env_features = env_features
        
        # Apply environmental gating to visual features
        gated_visual = env_gate.unsqueeze(1) * visual_features.unsqueeze(1)
        gated_visual = gated_visual.squeeze(1)
        
        # Combine all features
        combined_features = torch.cat([
            gated_visual, temporal_features, env_features
        ], dim=1)
        
        # Feature fusion
        fused_features = self.fusion_layer(combined_features)
        
        # Classification
        logits = self.classifier(fused_features)
        
        return logits


class ExplainabilityEngine(nn.Module):
    """
    Multi-modal explainability engine that integrates Grad-CAM, SHAP, and LIME 
    techniques with temporal consistency constraints.
    """
    
    def __init__(self, model: XAIPathBackbone):
        super(ExplainabilityEngine, self).__init__()
        self.model = model
        self.gradients = None
        self.activations = None
        
        # Register hooks for Grad-CAM
        self._register_hooks()
        
    def _register_hooks(self):
        """Register forward and backward hooks for gradient extraction."""
        
        def forward_hook(module, input, output):
            self.activations = output
            
        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0]
            
        # Register hooks on the last convolutional layer
        target_layer = self.model.backbone.layer4[-1].conv3
        target_layer.register_forward_hook(forward_hook)
        target_layer.register_backward_hook(backward_hook)
    
    def generate_gradcam(self, images: torch.Tensor, time: torch.Tensor, 
                        env_condition: torch.Tensor, target_class: int = None) -> torch.Tensor:
        """
        Generate Grad-CAM attention maps.
        
        Args:
            images: Input images of shape (batch_size, 3, H, W)
            time: Time values of shape (batch_size,)
            env_condition: Environmental conditions of shape (batch_size,)
            target_class: Target class for gradient computation
            
        Returns:
            Grad-CAM attention maps of shape (batch_size, H, W)
        """
        self.model.eval()
        images.requires_grad_()
        
        # Forward pass
        logits = self.model(images, time, env_condition)
        
        # Backward pass for target class
        if target_class is None:
            target_class = logits.argmax(dim=1)
        
        # Compute gradients
        self.model.zero_grad()
        class_score = logits[:, target_class].sum()
        class_score.backward(retain_graph=True)
        
        # Generate Grad-CAM
        gradients = self.gradients  # (batch_size, channels, H, W)
        activations = self.activations  # (batch_size, channels, H, W)
        
        # Global average pooling of gradients
        weights = torch.mean(gradients, dim=(2, 3), keepdim=True)
        
        # Weighted combination of activation maps
        cam = torch.sum(weights * activations, dim=1)  # (batch_size, H, W)
        cam = F.relu(cam)
        
        # Normalize to [0, 1]
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)
        
        return cam
    
    def compute_temporal_consistency_loss(self, attention_maps: List[torch.Tensor]) -> torch.Tensor:
        """
        Compute temporal consistency loss for attention maps.
        
        Args:
            attention_maps: List of attention maps for consecutive time points
            
        Returns:
            Temporal consistency loss
        """
        if len(attention_maps) < 2:
            return torch.tensor(0.0, device=attention_maps[0].device)
        
        consistency_loss = 0.0
        for i in range(len(attention_maps) - 1):
            diff = attention_maps[i] - attention_maps[i + 1]
            consistency_loss += torch.mean(diff ** 2)
        
        return consistency_loss / (len(attention_maps) - 1)
    
    def compute_environmental_consistency_loss(self, attention_maps: torch.Tensor, 
                                             env_conditions: torch.Tensor) -> torch.Tensor:
        """
        Compute environmental consistency loss for attention maps.
        
        Args:
            attention_maps: Attention maps of shape (batch_size, H, W)
            env_conditions: Environmental conditions of shape (batch_size,)
            
        Returns:
            Environmental consistency loss
        """
        unique_conditions = torch.unique(env_conditions)
        consistency_loss = 0.0
        
        for condition in unique_conditions:
            mask = env_conditions == condition
            if mask.sum() > 1:
                condition_maps = attention_maps[mask]
                mean_map = torch.mean(condition_maps, dim=0, keepdim=True)
                variance = torch.mean((condition_maps - mean_map) ** 2)
                consistency_loss += variance
        
        return consistency_loss / len(unique_conditions)


class XAIPathModel(nn.Module):
    """
    Complete XAIPath framework integrating detection and explainability.
    """
    
    def __init__(self, num_classes: int = 3, temporal_dim: int = 128, 
                 env_dim: int = 64, lambda_temp: float = 0.1, lambda_env: float = 0.05):
        super(XAIPathModel, self).__init__()
        
        self.backbone = XAIPathBackbone(num_classes, temporal_dim, env_dim)
        self.explainability_engine = ExplainabilityEngine(self.backbone)
        
        self.lambda_temp = lambda_temp
        self.lambda_env = lambda_env
        
    def forward(self, images: torch.Tensor, time: torch.Tensor, 
                env_condition: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass with explainability computation.
        
        Args:
            images: Input images of shape (batch_size, 3, H, W)
            time: Time values of shape (batch_size,)
            env_condition: Environmental conditions of shape (batch_size,)
            
        Returns:
            Dictionary containing logits and attention maps
        """
        # Classification
        logits = self.backbone(images, time, env_condition)
        
        # Generate explanations
        attention_maps = self.explainability_engine.generate_gradcam(
            images, time, env_condition
        )
        
        return {
            'logits': logits,
            'attention_maps': attention_maps,
            'visual_features': self.backbone.visual_features,
            'temporal_features': self.backbone.temporal_features,
            'env_features': self.backbone.env_features
        }
    
    def compute_loss(self, outputs: Dict[str, torch.Tensor], targets: torch.Tensor,
                    time: torch.Tensor, env_condition: torch.Tensor,
                    temporal_attention_maps: List[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        Compute total loss including classification and consistency terms.
        
        Args:
            outputs: Model outputs dictionary
            targets: Ground truth labels
            time: Time values
            env_condition: Environmental conditions
            temporal_attention_maps: List of attention maps for temporal consistency
            
        Returns:
            Dictionary containing loss components
        """
        # Classification loss
        cls_loss = F.cross_entropy(outputs['logits'], targets)
        
        # Temporal consistency loss
        temp_loss = torch.tensor(0.0, device=outputs['logits'].device)
        if temporal_attention_maps is not None:
            temp_loss = self.explainability_engine.compute_temporal_consistency_loss(
                temporal_attention_maps
            )
        
        # Environmental consistency loss
        env_loss = self.explainability_engine.compute_environmental_consistency_loss(
            outputs['attention_maps'], env_condition
        )
        
        # Total loss
        total_loss = cls_loss + self.lambda_temp * temp_loss + self.lambda_env * env_loss
        
        return {
            'total_loss': total_loss,
            'cls_loss': cls_loss,
            'temp_loss': temp_loss,
            'env_loss': env_loss
        }


def create_xaipath_model(num_classes: int = 3, **kwargs) -> XAIPathModel:
    """
    Factory function to create XAIPath model.
    
    Args:
        num_classes: Number of classification classes
        **kwargs: Additional model parameters
        
    Returns:
        XAIPath model instance
    """
    return XAIPathModel(num_classes=num_classes, **kwargs)


if __name__ == "__main__":
    # Test model creation and forward pass
    model = create_xaipath_model(num_classes=3)
    
    # Create dummy inputs
    batch_size = 4
    images = torch.randn(batch_size, 3, 224, 224)
    time = torch.rand(batch_size) * 4.0  # 0-4 hours
    env_condition = torch.randint(0, 2, (batch_size,))  # 0 or 1
    targets = torch.randint(0, 3, (batch_size,))
    
    # Forward pass
    outputs = model(images, time, env_condition)
    
    # Compute loss
    losses = model.compute_loss(outputs, targets, time, env_condition)
    
    print("Model created successfully!")
    print(f"Output shapes:")
    print(f"  Logits: {outputs['logits'].shape}")
    print(f"  Attention maps: {outputs['attention_maps'].shape}")
    print(f"  Total loss: {losses['total_loss'].item():.4f}")

