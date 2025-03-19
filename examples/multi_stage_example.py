"""Example of using the multi-stage LLM pipeline.

This example demonstrates how to chain multiple models together:
1. First process the prompt with a Hugging Face model
2. Feed that output to a local model to generate multiple variations
3. Use an evaluator model to select the best response

Note: This example requires implementing the actual model integration code
before it will work properly.
"""
import sys
import os

# Add the parent directory to the path so we can import the package
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from llm_eval.models.huggingface import HuggingFaceModel
from llm_eval.models.local import LocalModel
from llm_eval.models.evaluator import EvaluatorModel
from llm_eval.multi_stage_pipeline import MultiStagePipeline


def main():
    # Create the primary model (Hugging Face)
    # Note: You'll need to implement the actual API integration before this works
    primary_model = HuggingFaceModel(model_id="gpt2")
    
    # Create the secondary model (Local)
    # Note: You'll need to implement the actual model loading code before this works
    model_path = "./models/my_local_model"  # Path to your local model files
    secondary_model = LocalModel(model_path=model_path, device="cpu")
    secondary_model.load_model()
    
    # Create the evaluator model
    evaluator_model = EvaluatorModel(model_id="response-evaluator")
    
    # Create the multi-stage pipeline
    pipeline = MultiStagePipeline(
        primary_model=primary_model,
        secondary_model=secondary_model,
        evaluator_model=evaluator_model,
        num_variations=3  # Generate 3 variations with the local model
    )
    
    # Run a single prompt through the multi-stage pipeline
    prompt = "What is the capital of France?"
    print(f"\nRunning prompt through multi-stage pipeline: {prompt}")
    response = pipeline.run(prompt)
    print(f"Final response: {response}")
    
    # Run multiple prompts
    prompts = [
        "What is machine learning?",
        "Explain natural language processing.",
        "How do transformers work?"
    ]
    print(f"\nRunning batch of {len(prompts)} prompts through multi-stage pipeline...")
    responses = pipeline.run_batch(prompts)
    
    # Print the results
    for prompt, response in zip(prompts, responses):
        print(f"\nPrompt: {prompt}")
        print(f"Final response: {response}")
    
    # Save the results
    results_file = "multi_stage_results.json"
    pipeline.save_results(results_file)
    print(f"\nResults saved to {results_file}")


if __name__ == "__main__":
    main()