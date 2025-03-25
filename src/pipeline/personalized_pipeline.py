"""Personalized LLM pipeline for style adaptation and response optimization."""
from typing import Dict, List, Optional, Union, Any, Callable
import json
import time

from ..models.base import LLMModel
from ..models.huggingface import HuggingFaceModel
from ..models.local import LocalModel
from ..models.evaluator import EvaluatorModel


class PersonalizedPipeline:
    """Pipeline for generating responses in a personalized style.
    
    This pipeline implements a three-stage process:
    1. Generate initial response with a large LLM (e.g., DeepSeek)
    2. Adapt the response to a personal style using a locally fine-tuned model
    3. Evaluate multiple variations and select the best one
    4. Support iterative improvement based on feedback
    """
    
    def __init__(self, 
                 initial_model: HuggingFaceModel,
                 style_model: LocalModel,
                 evaluator_model: EvaluatorModel,
                 num_variations: int = 3,
                 style_prompt_template: Optional[str] = None,
                 feedback_prompt_template: Optional[str] = None):
        """Initialize the personalized pipeline with models.
        
        Args:
            initial_model: The model for generating initial responses (e.g., DeepSeek via HuggingFace)
            style_model: The locally fine-tuned model for style adaptation
            evaluator_model: The model used to evaluate and select the best response
            num_variations: Number of style variations to generate
            style_prompt_template: Template for adapting style (uses {content} placeholder)
            feedback_prompt_template: Template for incorporating feedback (uses {prompt}, {response}, and {feedback} placeholders)
        """
        self.initial_model = initial_model
        self.style_model = style_model
        self.evaluator_model = evaluator_model
        self.num_variations = num_variations
        self.results = []
        
        # Default templates if none provided
        self.style_prompt_template = style_prompt_template or "Rewrite the following text in my personal style, keeping the same information but adapting the tone and phrasing: {content}"
        self.feedback_prompt_template = feedback_prompt_template or "Original prompt: {prompt}\n\nInitial response: {response}\n\nFeedback: {feedback}\n\nImproved response:"
    
    def run(self, prompt: str, auto_improve: bool = False, max_iterations: int = 2, 
            improvement_threshold: float = 0.1, **kwargs) -> Dict[str, Any]:
        """Run a prompt through the personalized pipeline.
        
        Args:
            prompt: The input text
            auto_improve: Whether to automatically apply evaluator feedback
            max_iterations: Maximum number of improvement iterations (if auto_improve is True)
            improvement_threshold: Minimum score improvement required to continue iterations
            **kwargs: Additional parameters to pass to the models
            
        Returns:
            Dictionary containing the best response and additional information
        """
        start_time = time.time()
        
        # Step 1: Generate initial response with large LLM
        initial_response = self.initial_model.generate(prompt, **kwargs)
        
        # Step 2: Generate multiple style variations
        variations = []
        for i in range(self.num_variations):
            # Add slight parameter variations to get diversity
            variation_kwargs = kwargs.copy()
            if "temperature" in variation_kwargs:
                variation_kwargs["temperature"] = min(1.0, variation_kwargs["temperature"] * (1.0 + (i * 0.1)))
            
            style_prompt = self.style_prompt_template.format(content=initial_response)
            styled_variation = self.style_model.generate(style_prompt, **variation_kwargs)
            variations.append(styled_variation)
        
        # Step 3: Evaluate variations and select the best
        evaluation_result = self.evaluator_model.evaluate_responses(prompt, variations)
        best_response = evaluation_result["best_response"]
        best_index = evaluation_result["best_index"]
        scores = evaluation_result["scores"]
        
        # Step 4: Apply automatic improvements if enabled
        improvement_history = []
        
        if auto_improve and "feedback" in evaluation_result:
            current_response = best_response
            current_score = scores[best_index] if isinstance(scores, list) else scores
            
            for iteration in range(max_iterations):
                # Get feedback for the current best response
                feedback = evaluation_result.get("feedback") or "Improve this response to make it more natural and in my personal style."
                
                # Apply the feedback
                improved_response = self.improve_with_feedback(prompt, current_response, feedback, **kwargs)
                
                # Evaluate the improved response
                new_eval_result = self.evaluator_model.evaluate_responses(prompt, [current_response, improved_response])
                new_score = new_eval_result["scores"][-1] if isinstance(new_eval_result["scores"], list) else new_eval_result["scores"]
                
                # Record the improvement
                improvement_info = {
                    "iteration": iteration + 1,
                    "previous_response": current_response,
                    "previous_score": current_score,
                    "feedback": feedback,
                    "improved_response": improved_response,
                    "improved_score": new_score,
                    "improvement": new_score - current_score
                }
                improvement_history.append(improvement_info)
                
                # Check if the improvement is significant
                if new_score - current_score < improvement_threshold:
                    break
                    
                # Update for next iteration
                current_response = improved_response
                current_score = new_score
                
            # Use the latest improved response
            best_response = current_response
        
        end_time = time.time()
        
        # Compile results
        result = {
            "original_prompt": prompt,
            "initial_response": initial_response,
            "styled_variations": variations,
            "best_response": best_response,
            "best_variation_index": best_index,
            "evaluation_scores": scores,
            "initial_model_info": self.initial_model.get_model_info(),
            "style_model_info": self.style_model.get_model_info(),
            "evaluator_model_info": self.evaluator_model.get_model_info(),
            "time_taken": end_time - start_time,
            "timestamp": time.time()
        }
        
        # Add improvement history if auto_improve was enabled
        if auto_improve and improvement_history:
            result["improvement_history"] = improvement_history
        
        self.results.append(result)
        return result
    
    def improve_with_feedback(self, prompt: str, response: str, feedback: str, **kwargs) -> str:
        """Improve a response based on feedback.
        
        Args:
            prompt: The original input prompt
            response: The response to improve
            feedback: Feedback on how to improve the response
            **kwargs: Additional parameters to pass to the models
            
        Returns:
            The improved response
        """
        feedback_prompt = self.feedback_prompt_template.format(
            prompt=prompt, 
            response=response,
            feedback=feedback
        )
        
        # Generate improved response using the style model
        improved_response = self.style_model.generate(feedback_prompt, **kwargs)
        
        # Record the improvement cycle
        self.results.append({
            "type": "improvement",
            "original_prompt": prompt,
            "original_response": response,
            "feedback": feedback,
            "improved_response": improved_response,
            "style_model_info": self.style_model.get_model_info(),
            "timestamp": time.time()
        })
        
        return improved_response
    
    def run_batch(self, prompts: List[str], **kwargs) -> List[Dict[str, Any]]:
        """Run multiple prompts through the pipeline.
        
        Args:
            prompts: List of input prompts
            **kwargs: Additional parameters to pass to the models
            
        Returns:
            List of result dictionaries with best responses and evaluation info
        """
        return [self.run(prompt, **kwargs) for prompt in prompts]
    
    def get_best_responses(self, batch_results: List[Dict[str, Any]]) -> List[str]:
        """Extract just the best responses from a batch run.
        
        Args:
            batch_results: Results from run_batch method
            
        Returns:
            List of best responses only
        """
        return [result["best_response"] for result in batch_results]
    
    def fine_tune_style_model(self, examples: List[Dict[str, str]], **training_kwargs) -> Dict[str, Any]:
        """Fine-tune the style model on additional examples.
        
        Args:
            examples: List of dictionaries with 'prompt' and 'response' keys
            **training_kwargs: Parameters for fine-tuning (epochs, batch_size, etc.)
            
        Returns:
            Training results
        """
        return self.style_model.fine_tune(examples, **training_kwargs)
    
    def save_results(self, file_path: str) -> None:
        """Save the current results to a JSON file.
        
        Args:
            file_path: Path to save the results
        """
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(self.results, f, indent=2, ensure_ascii=False)
    
    def clear_results(self) -> None:
        """Clear the stored results."""
        self.results = []
        
    def update_style_prompt_template(self, new_template: str) -> None:
        """Update the style prompt template.
        
        Args:
            new_template: New template string (must include {content} placeholder)
        """
        if "{content}" not in new_template:
            raise ValueError("Style prompt template must include {content} placeholder")
        self.style_prompt_template = new_template
        
    def update_feedback_prompt_template(self, new_template: str) -> None:
        """Update the feedback prompt template.
        
        Args:
            new_template: New template string (must include {prompt}, {response}, 
                         and {feedback} placeholders)
        """
        required = ["{prompt}", "{response}", "{feedback}"]
        for placeholder in required:
            if placeholder not in new_template:
                raise ValueError(f"Feedback prompt template must include {placeholder} placeholder")
        self.feedback_prompt_template = new_template