"""
Utility functions for data preparation.
"""
import os
import json
import random
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

def split_dataset(dataset_path, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1, seed=42):
    """
    Split a dataset into train, validation, and test sets.
    
    Args:
        dataset_path (str): Path to the dataset JSON file
        train_ratio (float): Proportion for training
        val_ratio (float): Proportion for validation
        test_ratio (float): Proportion for testing
        seed (int): Random seed for reproducibility
        
    Returns:
        tuple: Paths to the saved dataset splits
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "Ratios must sum to 1"
    
    logger.info(f"Splitting dataset: {dataset_path}")
    
    # Load dataset
    with open(dataset_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Shuffle with fixed seed for reproducibility
    random.seed(seed)
    random.shuffle(data)
    
    # Calculate split sizes
    n = len(data)
    train_size = int(n * train_ratio)
    val_size = int(n * val_ratio)
    
    # Split data
    train_data = data[:train_size]
    val_data = data[train_size:train_size + val_size]
    test_data = data[train_size + val_size:]
    
    # Generate output paths
    base_path = Path(dataset_path).stem
    dir_path = Path(dataset_path).parent
    
    train_path = dir_path / f"{base_path}_train.json"
    val_path = dir_path / f"{base_path}_val.json"
    test_path = dir_path / f"{base_path}_test.json"
    
    # Save splits
    with open(train_path, 'w', encoding='utf-8') as f:
        json.dump(train_data, f, indent=2, ensure_ascii=False)
    
    with open(val_path, 'w', encoding='utf-8') as f:
        json.dump(val_data, f, indent=2, ensure_ascii=False)
    
    with open(test_path, 'w', encoding='utf-8') as f:
        json.dump(test_data, f, indent=2, ensure_ascii=False)
    
    logger.info(f"Created splits: train={len(train_data)}, val={len(val_data)}, test={len(test_data)}")
    
    return str(train_path), str(val_path), str(test_path)