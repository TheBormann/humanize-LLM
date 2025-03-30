"""Example of using the evaluator model to select the best response.

This example demonstrates how to use the EvaluatorModel in both lightweight
and model-based modes to evaluate and select the best response from multiple candidates.
It shows how to use both statistical methods and a local HuggingFace model for evaluation.
"""
import sys
import os
import json

# Add the parent directory to the path so we can import the package
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Removed mock model import as it doesn't work with the eval model
from src.models.evaluator import EvaluatorModel
from src.models.huggingface import HuggingFaceModel


def main():
    # Create some mock responses to evaluate
    responses = [
        "Paris is the capital of France. It is known for the Eiffel Tower, the Louvre Museum, and its cuisine.",
        "The capital of France is Paris, which is located in the north-central part of the country.",
        "France's capital city is Paris. It's one of the most visited cities in the world."
    ]
    
    original_prompt = "What is the capital of France?"
    
    print("\n=== Lightweight Evaluation (Statistical Methods) ===\n")
    
    # Create an evaluator model in lightweight mode (default)
    lightweight_evaluator = EvaluatorModel(model_id="lightweight-evaluator", mode="lightweight")
    
    # Evaluate the responses
    lightweight_results = lightweight_evaluator.evaluate_responses(original_prompt, responses)
    
    # Print the results
    print(f"Best response: {lightweight_results['best_response']}")
    print(f"Best index: {lightweight_results['best_index']}")
    print(f"Scores: {[round(score, 3) for score in lightweight_results['scores']]}")
    print(f"Evaluation mode: {lightweight_results['evaluation_mode']}")
    
    # We've removed the mock model evaluation section as it doesn't work with the eval model
    
    # Now try with a real local HuggingFace model
    try:
        print("\n=== Model-Based Evaluation (Using Local HuggingFace Model) ===\n")
        
        # Create a local HuggingFace model for evaluation
        # You can replace this with any model from Hugging Face Hub
        model_id = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"  # Using a small model as an example
        local_eval_model = HuggingFaceLocalModel(
            model_id=model_id,
            device="cpu",  # Use "cuda" for GPU acceleration if available
            memory_size=1000,  # Size of memory buffer for online learning
            online_batch_size=1,  # Batch size for online updates
            online_learning_rate=1e-5,  # Learning rate for online updates
        )
        
        # Create an evaluator model in model-based mode with the local model
        local_model_evaluator = EvaluatorModel(
            model_id="local-model-evaluator",
            evaluation_model=local_eval_model,
            mode="model-based"
        )
        
        # Evaluate the responses
        local_model_results = local_model_evaluator.evaluate_responses(original_prompt, responses)
        
        # Print the results
        print(f"Best response: {local_model_results['best_response']}")
        print(f"Best index: {local_model_results['best_index']}")
        print(f"Scores: {[round(score, 3) for score in local_model_results['scores']]}")
        print(f"Evaluation mode: {local_model_results['evaluation_mode']}")
        
        # Compare results between different evaluation methods
        print("\n=== Comparison of Evaluation Methods ===\n")
        print(f"Lightweight best index: {lightweight_results['best_index']}")
        print(f"Local model best index: {local_model_results['best_index']}")
        
    except Exception as e:
        print(f"\nError demonstrating local model-based evaluation: {str(e)}")
        print("Make sure you have the required dependencies installed for HuggingFace local models.")

if __name__ == "__main__":
    main()