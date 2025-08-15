import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from PIL import Image
import pandas as pd
import numpy as np
import os
from typing import Dict, List, Tuple, Optional, Union
import json
from sklearn.model_selection import train_test_split, StratifiedKFold
import cv2


class BacterialDataset(Dataset):
    """
    Dataset class for bacterial microscopy images with temporal and environmental metadata.
    """
    
    def __init__(self, 
                 data_dir: str,
                 metadata_file: str,
                 transform: Optional[transforms.Compose] = None,
                 temporal_normalize: bool = True,
                 max_time: float = 4.0):
        """
        Initialize bacterial dataset.
        
        Args:
            data_dir: Directory containing image files
            metadata_file: CSV file with image metadata
            transform: Image transformations
            temporal_normalize: Whether to normalize temporal values
            max_time: Maximum time value for normalization
        """
        self.data_dir = data_dir
        self.transform = transform
        self.temporal_normalize = temporal_normalize
        self.max_time = max_time
        
        # Load metadata
        self.metadata = pd.read_csv(metadata_file)
        
        # Validate required columns
        required_cols = ['image_path', 'label', 'time_hours', 'env_condition']
        for col in required_cols:
            if col not in self.metadata.columns:
                raise ValueError(f"Missing required column: {col}")
        
        # Create label mapping
        self.label_mapping = {
            'background': 0,
            'salmonella': 1,
            'mixed_culture': 2
        }
        
        # Convert string labels to integers if necessary
        if self.metadata['label'].dtype == 'object':
            self.metadata['label'] = self.metadata['label'].map(self.label_mapping)
        
        # Filter out invalid entries
        self.metadata = self.metadata.dropna()
        self.metadata = self.metadata.reset_index(drop=True)
        
        print(f"Loaded {len(self.metadata)} samples")
        print(f"Label distribution: {self.metadata['label'].value_counts().to_dict()}")
        
    def __len__(self) -> int:
        return len(self.metadata)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a single sample from the dataset.
        
        Args:
            idx: Sample index
            
        Returns:
            Dictionary containing image, label, time, and environmental condition
        """
        row = self.metadata.iloc[idx]
        
        # Load image
        image_path = os.path.join(self.data_dir, row['image_path'])
        image = Image.open(image_path).convert('RGB')
        
        # Apply transformations
        if self.transform:
            image = self.transform(image)
        
        # Get temporal information
        time_hours = float(row['time_hours'])
        if self.temporal_normalize:
            time_hours = time_hours / self.max_time
        
        # Get environmental condition
        env_condition = int(row['env_condition'])
        
        # Get label
        label = int(row['label'])
        
        return {
            'image': image,
            'label': torch.tensor(label, dtype=torch.long),
            'time': torch.tensor(time_hours, dtype=torch.float32),
            'env_condition': torch.tensor(env_condition, dtype=torch.long),
            'image_path': row['image_path']
        }
    
    def get_class_weights(self) -> torch.Tensor:
        """
        Compute class weights for balanced training.
        
        Returns:
            Class weights tensor
        """
        label_counts = self.metadata['label'].value_counts().sort_index()
        total_samples = len(self.metadata)
        
        weights = []
        for i in range(len(label_counts)):
            weight = total_samples / (len(label_counts) * label_counts[i])
            weights.append(weight)
        
        return torch.tensor(weights, dtype=torch.float32)
    
    def get_temporal_distribution(self) -> Dict[str, List]:
        """
        Get temporal distribution statistics.
        
        Returns:
            Dictionary with temporal statistics
        """
        return {
            'time_points': self.metadata['time_hours'].unique().tolist(),
            'time_distribution': self.metadata['time_hours'].value_counts().to_dict(),
            'mean_time': self.metadata['time_hours'].mean(),
            'std_time': self.metadata['time_hours'].std()
        }
    
    def get_environmental_distribution(self) -> Dict[str, int]:
        """
        Get environmental condition distribution.
        
        Returns:
            Dictionary with environmental statistics
        """
        return self.metadata['env_condition'].value_counts().to_dict()


def create_transforms(image_size: Tuple[int, int] = (224, 224), 
                     augment: bool = True) -> Dict[str, transforms.Compose]:
    """
    Create image transformation pipelines for training and validation.
    
    Args:
        image_size: Target image size (height, width)
        augment: Whether to apply data augmentation
        
    Returns:
        Dictionary with train and validation transforms
    """
    # Base transforms
    base_transforms = [
        transforms.Resize(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ]
    
    # Training transforms with augmentation
    if augment:
        train_transforms = [
            transforms.Resize((int(image_size[0] * 1.1), int(image_size[1] * 1.1))),
            transforms.RandomCrop(image_size),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomRotation(degrees=15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ]
    else:
        train_transforms = base_transforms.copy()
    
    return {
        'train': transforms.Compose(train_transforms),
        'val': transforms.Compose(base_transforms),
        'test': transforms.Compose(base_transforms)
    }


def create_synthetic_metadata(num_samples: int = 2847, 
                            output_file: str = 'metadata.csv') -> pd.DataFrame:
    """
    Create synthetic metadata for demonstration purposes.
    
    Args:
        num_samples: Number of samples to generate
        output_file: Output CSV file path
        
    Returns:
        Generated metadata DataFrame
    """
    np.random.seed(42)
    
    # Define time points (8 temporal phases from 0.5 to 4 hours)
    time_points = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0]
    
    # Generate samples
    data = []
    for i in range(num_samples):
        # Random time point
        time_hours = np.random.choice(time_points)
        
        # Random environmental condition (0: without onion, 1: with onion)
        env_condition = np.random.choice([0, 1])
        
        # Label distribution: background (0), salmonella (1), mixed_culture (2)
        # Based on paper: 1,156 pure Salmonella, 1,691 mixed culture
        if i < 1156:
            label = 1  # salmonella
        elif i < 1156 + 1691:
            label = 2  # mixed_culture
        else:
            label = 0  # background (remaining samples)
        
        # Generate image path
        image_path = f"images/sample_{i:06d}.png"
        
        data.append({
            'image_path': image_path,
            'label': label,
            'time_hours': time_hours,
            'env_condition': env_condition,
            'sample_id': i
        })
    
    # Create DataFrame
    metadata = pd.DataFrame(data)
    
    # Shuffle the data
    metadata = metadata.sample(frac=1, random_state=42).reset_index(drop=True)
    
    # Save to CSV
    metadata.to_csv(output_file, index=False)
    
    print(f"Generated metadata for {num_samples} samples")
    print(f"Label distribution: {metadata['label'].value_counts().to_dict()}")
    print(f"Time distribution: {metadata['time_hours'].value_counts().to_dict()}")
    print(f"Environment distribution: {metadata['env_condition'].value_counts().to_dict()}")
    
    return metadata


def create_stratified_splits(metadata: pd.DataFrame, 
                           test_size: float = 0.2, 
                           val_size: float = 0.2,
                           random_state: int = 42) -> Dict[str, pd.DataFrame]:
    """
    Create stratified train/validation/test splits.
    
    Args:
        metadata: Metadata DataFrame
        test_size: Proportion of test set
        val_size: Proportion of validation set (from remaining data)
        random_state: Random seed
        
    Returns:
        Dictionary with train, validation, and test DataFrames
    """
    # Create stratification key combining label, time, and environment
    metadata['strat_key'] = (metadata['label'].astype(str) + '_' + 
                           metadata['time_hours'].astype(str) + '_' + 
                           metadata['env_condition'].astype(str))
    
    # First split: train+val vs test
    train_val_df, test_df = train_test_split(
        metadata, 
        test_size=test_size, 
        stratify=metadata['strat_key'],
        random_state=random_state
    )
    
    # Second split: train vs val
    train_df, val_df = train_test_split(
        train_val_df,
        test_size=val_size,
        stratify=train_val_df['strat_key'],
        random_state=random_state
    )
    
    # Remove stratification key
    for df in [train_df, val_df, test_df]:
        df.drop('strat_key', axis=1, inplace=True)
    
    print(f"Split sizes - Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")
    
    return {
        'train': train_df.reset_index(drop=True),
        'val': val_df.reset_index(drop=True),
        'test': test_df.reset_index(drop=True)
    }


def create_data_loaders(data_dir: str,
                       metadata_splits: Dict[str, pd.DataFrame],
                       batch_size: int = 32,
                       num_workers: int = 4,
                       image_size: Tuple[int, int] = (224, 224)) -> Dict[str, DataLoader]:
    """
    Create data loaders for train, validation, and test sets.
    
    Args:
        data_dir: Directory containing image files
        metadata_splits: Dictionary with train/val/test metadata
        batch_size: Batch size for data loaders
        num_workers: Number of worker processes
        image_size: Target image size
        
    Returns:
        Dictionary with train, validation, and test data loaders
    """
    transforms_dict = create_transforms(image_size=image_size, augment=True)
    
    data_loaders = {}
    
    for split_name, metadata_df in metadata_splits.items():
        # Save split metadata to temporary file
        temp_metadata_file = f"temp_{split_name}_metadata.csv"
        metadata_df.to_csv(temp_metadata_file, index=False)
        
        # Create dataset
        dataset = BacterialDataset(
            data_dir=data_dir,
            metadata_file=temp_metadata_file,
            transform=transforms_dict[split_name if split_name != 'test' else 'val']
        )
        
        # Create data loader
        shuffle = (split_name == 'train')
        data_loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=(split_name == 'train')
        )
        
        data_loaders[split_name] = data_loader
        
        # Clean up temporary file
        os.remove(temp_metadata_file)
    
    return data_loaders


class TemporalBatchSampler:
    """
    Custom batch sampler that ensures temporal consistency within batches.
    """
    
    def __init__(self, dataset: BacterialDataset, batch_size: int, 
                 temporal_window: float = 0.5):
        """
        Initialize temporal batch sampler.
        
        Args:
            dataset: Bacterial dataset
            batch_size: Batch size
            temporal_window: Time window for grouping samples
        """
        self.dataset = dataset
        self.batch_size = batch_size
        self.temporal_window = temporal_window
        
        # Group samples by temporal windows
        self.temporal_groups = self._create_temporal_groups()
        
    def _create_temporal_groups(self) -> Dict[float, List[int]]:
        """
        Group sample indices by temporal windows.
        
        Returns:
            Dictionary mapping time windows to sample indices
        """
        groups = {}
        
        for idx in range(len(self.dataset)):
            time_hours = self.dataset.metadata.iloc[idx]['time_hours']
            
            # Find appropriate time window
            window_key = round(time_hours / self.temporal_window) * self.temporal_window
            
            if window_key not in groups:
                groups[window_key] = []
            groups[window_key].append(idx)
        
        return groups
    
    def __iter__(self):
        """
        Generate batches with temporal consistency.
        
        Yields:
            Batches of sample indices
        """
        all_batches = []
        
        for time_window, indices in self.temporal_groups.items():
            # Shuffle indices within time window
            np.random.shuffle(indices)
            
            # Create batches from this time window
            for i in range(0, len(indices), self.batch_size):
                batch = indices[i:i + self.batch_size]
                if len(batch) == self.batch_size:  # Only full batches
                    all_batches.append(batch)
        
        # Shuffle batches
        np.random.shuffle(all_batches)
        
        for batch in all_batches:
            yield batch
    
    def __len__(self):
        """
        Get number of batches.
        
        Returns:
            Number of batches
        """
        total_full_batches = 0
        for indices in self.temporal_groups.values():
            total_full_batches += len(indices) // self.batch_size
        return total_full_batches


if __name__ == "__main__":
    # Test dataset creation and loading
    
    # Create synthetic metadata
    metadata = create_synthetic_metadata(num_samples=100, output_file='test_metadata.csv')
    
    # Create splits
    splits = create_stratified_splits(metadata)
    
    # Create transforms
    transforms_dict = create_transforms()
    
    # Test dataset loading
    dataset = BacterialDataset(
        data_dir='dummy_data',  # This would be the actual data directory
        metadata_file='test_metadata.csv',
        transform=transforms_dict['train']
    )
    
    print(f"Dataset size: {len(dataset)}")
    print(f"Class weights: {dataset.get_class_weights()}")
    print(f"Temporal distribution: {dataset.get_temporal_distribution()}")
    print(f"Environmental distribution: {dataset.get_environmental_distribution()}")
    
    # Clean up
    os.remove('test_metadata.csv')

