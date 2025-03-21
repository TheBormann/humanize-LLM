"""Utility functions for working with models."""
import os
import logging
from typing import Optional, Dict, Any

from ..models.local import LocalModel

logger = logging.getLogger(__name__)

def get_local_model(
    model_name: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0", 
    cache_dir: Optional[str] = None,
    quantize: bool = False,
    quantize_type: str = "4bit",
    device: Optional[str] = None,
    **kwargs
) -> LocalModel:
    """Load a model from Hugging Face."""
    logger.info(f"Loading model: {model_name}")
    
    # Check if running on Mac
    import platform
    is_mac = platform.system() == "Darwin"
    
    # Check if MPS (Apple Metal) is available
    import torch
    if device is None and torch.backends.mps.is_available():
        device = "mps"
        logger.info(f"Using MPS (Apple Metal) acceleration")
    
    # Force disable quantization on Mac
    if platform.system() == "Darwin" and quantize:
        logger.warning("Quantization not fully supported on Mac. Disabling quantization.")
        quantize = False
    
    # Disable quantization on Mac or adjust for compatibility
    if is_mac and quantize:
        if quantize_type == "4bit":
            logger.warning("4-bit quantization not fully supported on Mac. Falling back to 8-bit.")
            quantize_type = "8bit"
        
    # Create model instance
    model = LocalModel(device=device)
    
    # Prepare load_model arguments
    load_args = {
        'quantize': quantize,
        'quantize_type': quantize_type,
        **kwargs
    }
    
    if cache_dir:
        os.makedirs(cache_dir, exist_ok=True)
        load_args['cache_dir'] = cache_dir
    
    # Load the model
    model.load_model(model_name, **load_args)
    
    return model