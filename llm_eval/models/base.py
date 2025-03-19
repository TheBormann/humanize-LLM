"""Base model interface for LLM implementations."""
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Union


class LLMModel(ABC):
    """Base interface for all LLM models.
    
    This abstract class defines the common interface that all model implementations
    must follow, whether they're API-based (like Hugging Face) or local models.
    """
    
    @abstractmethod
    def generate(self, prompt: str, **kwargs) -> str:
        """Generate a response for the given prompt.
        
        Args:
            prompt: The input text to generate a response for
            **kwargs: Additional model-specific parameters
            
        Returns:
            The generated text response
        """
        pass
    
    @abstractmethod
    def batch_generate(self, prompts: List[str], **kwargs) -> List[str]:
        """Generate responses for multiple prompts.
        
        Args:
            prompts: List of input prompts
            **kwargs: Additional model-specific parameters
            
        Returns:
            List of generated text responses
        """
        pass
    
    def get_model_info(self) -> Dict[str, str]:
        """Get information about the model.
        
        Returns:
            Dictionary with model information
        """
        return {
            "name": self.__class__.__name__,
            "type": "base"
        }