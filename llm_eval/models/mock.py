"""Mock model implementation for testing."""
from typing import Dict, List, Optional

from .base import LLMModel


class MockModel(LLMModel):
    """A simple mock model for testing the pipeline without real LLMs.
    
    This model simply echoes the input with a prefix, useful for testing
    the pipeline functionality without requiring actual model inference.
    """
    
    def __init__(self, prefix: str = "Mock response: "):
        """Initialize the mock model.
        
        Args:
            prefix: Text to prepend to all responses
        """
        self.prefix = prefix
    
    def generate(self, prompt: str, **kwargs) -> str:
        """Generate a mock response by echoing the input with a prefix.
        
        Args:
            prompt: The input text
            **kwargs: Ignored in mock implementation
            
        Returns:
            A mock response
        """
        return f"{self.prefix}{prompt}"
    
    def batch_generate(self, prompts: List[str], **kwargs) -> List[str]:
        """Generate mock responses for multiple prompts.
        
        Args:
            prompts: List of input prompts
            **kwargs: Ignored in mock implementation
            
        Returns:
            List of mock responses
        """
        return [self.generate(prompt) for prompt in prompts]
    
    def get_model_info(self) -> Dict[str, str]:
        """Get information about the mock model.
        
        Returns:
            Dictionary with model information
        """
        info = super().get_model_info()
        info.update({
            "type": "mock",
            "description": "Mock model for testing"
        })
        return info