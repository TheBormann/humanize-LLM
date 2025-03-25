"""Example of using the LLM pipeline with a Hugging Face model.

Before running this example:
1. Get your Hugging Face API token from https://huggingface.co/settings/tokens
2. Set the token in your environment as HF_API_KEY or pass it directly to HuggingFaceModel
"""
import sys
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Add the parent directory to the path so we can import the package
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.huggingface import HuggingFaceModel
from src.pipeline.pipeline import LLMPipeline

def main():
    # Create a Hugging Face model
    # Using a freely accessible model for demonstration
    model = HuggingFaceModel(model_id="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B", api_key=os.environ.get("HF_API_KEY"))
    
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