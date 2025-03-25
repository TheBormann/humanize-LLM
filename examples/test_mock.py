"""Simple example of using the LLM pipeline with a mock model."""
import sys
import os

# Add the parent directory to the path so we can import the package
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.mock import MockModel
from src.pipeline.pipeline import LLMPipeline


def main():
    # Create a mock model for testing
    model = MockModel(prefix="Test response: ")
    
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
    
    # Define a simple evaluator function
    def simple_evaluator(prompt, response):
        # This is just a dummy evaluator that scores based on response length
        # In a real scenario, you would implement a more meaningful evaluation
        return min(len(response) / 100, 1.0)
    
    # Evaluate the responses
    print("\nEvaluating responses...")
    evaluation = pipeline.evaluate(prompts, simple_evaluator)
    print(f"Average score: {evaluation['average_score']:.2f}")
    
    # Save the results
    results_file = "mock_results.json"
    pipeline.save_results(results_file)
    print(f"\nResults saved to {results_file}")


if __name__ == "__main__":
    main()