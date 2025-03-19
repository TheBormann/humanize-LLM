"""Example of using the LLM pipeline with a Hugging Face model.

Note: This example requires implementing the actual API integration in the
HuggingFaceModel class before it will work properly.
"""
import sys
import os

# Add the parent directory to the path so we can import the package
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from llm_eval.models.huggingface import HuggingFaceModel
from llm_eval.pipeline import LLMPipeline

def main():
    # Create a Hugging Face model
    # Note: You'll need to implement the actual API integration before this works
    model = HuggingFaceModel(model_id="gpt2")
    
    # Create a pipeline with the model
    pipeline = LLMPipeline(model)
    
    # Run a single prompt
    prompt = "What is the capital of France?"
    print(f"\nRunning single prompt: {prompt}")
    response = pipeline.run(prompt)
    print(f"Response: {response}")
    
    # Run multiple prompts
    prompts = [
        "What is machine learning?",
        "Explain natural language processing.",
        "How do transformers work?"
    ]
    print(f"\nRunning batch of {len(prompts)} prompts...")
    responses = pipeline.run_batch(prompts)
    
    # Print the results
    for prompt, response in zip(prompts, responses):
        print(f"\nPrompt: {prompt}")
        print(f"Response: {response}")
    
    # Save the results
    results_file = "huggingface_results.json"
    pipeline.save_results(results_file)
    print(f"\nResults saved to {results_file}")


if __name__ == "__main__":
    main()