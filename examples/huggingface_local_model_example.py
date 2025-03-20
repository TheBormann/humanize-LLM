"""Example of using the LLM pipeline with a Hugging Face local model.

This example demonstrates using a pre-trained model from Hugging Face Hub
with the LLM evaluation pipeline.
"""
import sys
import os

# Add the parent directory to the path so we can import the package
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from llm_eval.models.huggingface_local import HuggingFaceModel
from llm_eval.pipeline import LLMPipeline


def main():
    # Create a Hugging Face model
    # You can replace this with any model from Hugging Face Hub
    model_id = "gpt2"  # Using GPT-2 as an example
    model = HuggingFaceModel(
        model_id=model_id,
        device="cpu",  # Use "cuda" for GPU acceleration
        memory_size=1000,  # Size of memory buffer for online learning
        online_batch_size=1,  # Batch size for online updates
        online_learning_rate=1e-5,  # Learning rate for online updates
        model_path="examples/models/gpt2"  # Store models in examples/models directory
    )
    
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
    
    # Example of fine-tuning the model
    print("\nFine-tuning the model...")
    train_data = [
        {"prompt": "What is the capital of France?", "response": "The capital of France is Paris."},
        {"prompt": "What is machine learning?", "response": "Machine learning is a branch of AI that enables systems to learn from data."},
        {"prompt": "Explain natural language processing.", "response": "Natural language processing is a field of AI that focuses on the interaction between computers and human language."},
    ]
    
    fine_tune_results = model.fine_tune(train_data)
    print(f"Fine-tuning results: {fine_tune_results}")
    
    # Run the same prompts after fine-tuning to see the difference
    print("\nRunning prompts after fine-tuning...")
    post_tuning_responses = pipeline.run_batch(prompts)
    
    # Print the results after fine-tuning
    for prompt, response in zip(prompts, post_tuning_responses):
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
    results_file = "huggingface_model_results.json"
    pipeline.save_results(results_file)
    print(f"\nResults saved to {results_file}")


if __name__ == "__main__":
    main()