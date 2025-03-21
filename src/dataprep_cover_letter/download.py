"""
Module for downloading datasets from HuggingFace.
"""
import os
import logging
from datasets import load_dataset
from pathlib import Path

logger = logging.getLogger(__name__)

def download_hf_dataset(dataset_id, cache_dir=None):
    """
    Download a dataset from HuggingFace.
    
    Args:
        dataset_id (str): HuggingFace dataset ID
        cache_dir (str, optional): Directory to cache the dataset
        
    Returns:
        dataset: HuggingFace dataset object
    """
    logger.info(f"Downloading dataset: {dataset_id}")
    try:
        dataset = load_dataset(dataset_id, cache_dir=cache_dir)
        logger.info(f"Successfully downloaded dataset with {len(dataset)} splits")
        return dataset
    except Exception as e:
        logger.error(f"Error downloading dataset: {e}")
        raise e