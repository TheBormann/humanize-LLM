"""Multi-stage pipeline implementation for chaining multiple LLM models."""
from typing import Dict, List, Optional, Union, Any, Callable
import json
import time

from .models.base import LLMModel
from .models.huggingface import HuggingFaceModel
from .models.local import LocalModel
from .models.evaluator import EvaluatorModel
from .pipeline import LLMPipeline


class MultiStagePipeline:
    """Pipeline for chaining multiple LLM models together.
    
    This class provides an interface for running prompts through a sequence of models,
    where the output of one model becomes the input to the next, and finally selecting
    the best response from multiple candidates.
    """
    
    def __init__(self, 
                 primary_model: HuggingFaceModel,
                 secondary_model: LocalModel,
                 evaluator_model: EvaluatorModel,
                 num_variations: int = 3):
        """Initialize the multi-stage pipeline with models.
        
        Args:
            primary_model: The first model in the chain (Hugging Face model)
            secondary_model: The second model in the chain (Local model)
            evaluator_model: The model used to evaluate and select the best response
            num_variations: Number of variations to generate with the secondary model
        """
        self.primary_model = primary_model
        self.secondary_model = secondary_model
        self.evaluator_model = evaluator_model
        self.num_variations = num_variations
        self.results = []
    
    def run(self, prompt: str, **kwargs) -> str:
        """Run a prompt through the multi-stage pipeline.
        
        The prompt is first processed by the primary model, then the output
        is used as input to the secondary model to generate multiple variations,
        and finally the evaluator model selects the best response.
        
        Args:
            prompt: The input text
            **kwargs: Additional parameters to pass to the models
            
        Returns:
            The best generated response
        """
        start_time = time.time()
        
        # Step 1: Process the prompt with the primary model (Hugging Face)
        primary_response = self.primary_model.generate(prompt, **kwargs)
        
        # Step 2: Generate multiple variations with the secondary model (Local)
        variations = []
        for _ in range(self.num_variations):
            # Use the primary model's output as input to the secondary model
            # We could add some randomness or different parameters here to get variations
            variation = self.secondary_model.generate(primary_response, **kwargs)
            variations.append(variation)
        
        # Step 3: Evaluate the variations and select the best one
        evaluation_result = self.evaluator_model.evaluate_responses(prompt, variations)
        best_response = evaluation_result["best_response"]
        best_index = evaluation_result["best_index"]
        scores = evaluation_result["scores"]
        
        end_time = time.time()
        
        # Record the result
        result = {
            "original_prompt": prompt,
            "primary_response": primary_response,
            "variations": variations,
            "best_response": best_response,
            "best_variation_index": best_index,
            "evaluation_scores": scores,
            "primary_model_info": self.primary_model.get_model_info(),
            "secondary_model_info": self.secondary_model.get_model_info(),
            "evaluator_model_info": self.evaluator_model.get_model_info(),
            "time_taken": end_time - start_time,
            "timestamp": time.time()
        }
        
        self.results.append(result)
        return best_response
    
    def run_batch(self, prompts: List[str], **kwargs) -> List[str]:
        """Run multiple prompts through the multi-stage pipeline.
        
        Args:
            prompts: List of input prompts
            **kwargs: Additional parameters to pass to the models
            
        Returns:
            List of best generated responses
        """
        return [self.run(prompt, **kwargs) for prompt in prompts]
    
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