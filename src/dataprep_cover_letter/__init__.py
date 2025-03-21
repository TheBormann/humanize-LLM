"""
Data preparation module for email style fine-tuning.
Provides utilities to download, convert and process datasets.
"""

from .download import download_hf_dataset
from .convert import convert_cover_letter_dataset
from .analyze import dataset_statistics

__all__ = [
    'download_hf_dataset',
    'convert_cover_letter_dataset',
    'dataset_statistics'
]