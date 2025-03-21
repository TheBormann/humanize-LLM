"""Core pipeline implementation for LLM evaluation."""
from typing import Dict, List, Optional, Union, Any, Callable
import json
import time

from .models.base import LLMModel


class LLMPipeline:
    """Main pipeline for running LLM evaluations.
    
    This class provides a simple interface for running prompts through
    different LLM implementations and collecting the results.
    """
    
    def __init__(self, model: LLMModel):
        """Initialize the pipeline with a model.
        
        Args:
            model: An instance of an LLMModel implementation
        """
        self.model = model
        self.results = []
    
    def run(self, prompt: str, **kwargs) -> str:
        """Run a single prompt through the pipeline.
        
        Args:
            prompt: The input text
            **kwargs: Additional parameters to pass to the model
            
        Returns:
            The generated response
        """
        start_time = time.time()
        response = self.model.generate(prompt, **kwargs)
        end_time = time.time()
        
        result = {
            "prompt": prompt,
            "response": response,
            "model_info": self.model.get_model_info(),
            "time_taken": end_time - start_time,
            "timestamp": time.time()
        }
        
        self.results.append(result)
        return response
    
    def run_batch(self, prompts: List[str], **kwargs) -> List[str]:
        """Run multiple prompts through the pipeline.
        
        Args:
            prompts: List of input prompts
            **kwargs: Additional parameters to pass to the model
            
        Returns:
            List of generated responses
        """
        start_time = time.time()
        responses = self.model.batch_generate(prompts, **kwargs)
        end_time = time.time()
        
        for prompt, response in zip(prompts, responses):
            result = {
                "prompt": prompt,
                "response": response,
                "model_info": self.model.get_model_info(),
                "time_taken": (end_time - start_time) / len(prompts),  # Average time per prompt
                "timestamp": time.time()
            }
            self.results.append(result)
        
        return responses
    
    def evaluate(self, prompts: List[str], evaluator: Callable[[str, str], float], **kwargs) -> Dict[str, Any]:
        """Run prompts and evaluate the responses.
        
        Args:
            prompts: List of input prompts
            evaluator: Function that takes (prompt, response) and returns a score
            **kwargs: Additional parameters to pass to the model
            
        Returns:
            Dictionary with evaluation results
        """
        responses = self.run_batch(prompts, **kwargs)
        scores = [evaluator(prompt, response) for prompt, response in zip(prompts, responses)]
        
        evaluation = {
            "model_info": self.model.get_model_info(),
            "num_prompts": len(prompts),
            "average_score": sum(scores) / len(scores) if scores else 0,
            "scores": scores,
            "timestamp": time.time()
        }
        
        return evaluation
    
    def save_results(self, file_path: str) -> None:
        """Save the current results to a JSON file.
        
        Args:
            file_path: Path to save the results
        """
        with open(file_path, 'w') as f:
            json.dump(self.results, f, indent=2)
    
    def clear_results(self) -> None:
        """Clear the stored results."""
        self.results = []