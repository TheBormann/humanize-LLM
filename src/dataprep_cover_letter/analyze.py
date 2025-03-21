"""
Module for analyzing and visualizing datasets.
"""
import json
import logging
import pandas as pd
from collections import defaultdict

logger = logging.getLogger(__name__)

def dataset_statistics(dataset_path):
    """
    Generate statistics for a dataset.
    
    Args:
        dataset_path (str): Path to the dataset JSON file
        
    Returns:
        dict: Dictionary of statistics
    """
    logger.info(f"Analyzing dataset: {dataset_path}")
    
    try:
        with open(dataset_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        stats = {
            'sample_count': len(data),
            'avg_prompt_length': sum(len(item['prompt']) for item in data) / len(data),
            'avg_response_length': sum(len(item['response']) for item in data) / len(data),
            'min_prompt_length': min(len(item['prompt']) for item in data),
            'min_response_length': min(len(item['response']) for item in data),
            'max_prompt_length': max(len(item['prompt']) for item in data),
            'max_response_length': max(len(item['response']) for item in data),
        }
        
        return stats
    
    except Exception as e:
        logger.error(f"Error analyzing dataset: {e}")
        raise