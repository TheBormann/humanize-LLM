"""Base model interface for LLM implementations."""
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Union
import torch
import transformers

IGNORE_INDEX = -100
EOT_TOKEN = "<|EOT|>"

@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="deepseek-ai/deepseek-coder-6.7b-instruct")

@dataclass
class DataArguments:
    data_path: str = field(default=None, metadata={"help": "Path to the training data."})


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    model_max_length: int = field(
        default=512,
        metadata={"help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."},
    )
    
@dataclass
class GenerationArguments:
    max_new_tokens: int = field(default=128)
    temperature: float = field(default=0.7)
    top_p: float = field(default=0.9)
    
@dataclass
class FinetuningArguments:
    """Arguments for finetuning the model."""
    train_data: List[Dict[str, Union[str, int]]] = field(default=None, metadata={"help": "Training data."})
    eval_data: List[Dict[str, Union[str, int]]] = field(default=None, metadata={"help": "Evaluation data."})
    epochs: int = field(default=3, metadata={"help": "Number of training epochs."})
    batch_size: int = field(default=1, metadata={"help": "Batch size for training."})
    learning_rate: float = field(default=2e-5, metadata={"help": "Learning rate for training."})

@dataclass
class PEFTArguments:
    method: str = field(default="lora", metadata={"help": "PEFT method (lora, qlora, etc.)"})
    rank: int = field(default=8)
    alpha: int = field(default=16)
    dropout: float = field(default=0.05)

class LLMModel(ABC):
    """Base interface for all LLM models.
    
    This abstract class defines the common interface that all model implementations
    must follow, whether they're API-based (like Hugging Face) or local models.
    """
    def __init__(self, device: str = "cuda" if torch.cuda.is_available() else "cpu"):
        self.device = device
        
    @abstractmethod
    def load_model(self, model_path: str, **kwargs) -> None:
        """Load the model from a given path."""
        pass
    
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
    
    @abstractmethod
    def finetune(self, training_args: FinetuningArguments, **kwargs) -> Dict:
        """Finetune the model with given arguments."""
        pass
    
    @abstractmethod
    def save_checkpoint(self, path: str) -> None:
        """Save model checkpoint."""
        pass

    @abstractmethod
    def load_checkpoint(self, path: str) -> None:
        """Load model from checkpoint."""
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