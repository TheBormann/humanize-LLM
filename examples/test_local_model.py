"""Test script for LocalModel class."""
import logging
import sys
import time
from typing import List
import os

# Add the parent directory to the path so we can import the package
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.model_utils import get_local_model
from src.models.base import FinetuningArguments

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    stream=sys.stdout
)
logger = logging.getLogger(__name__)

def test_generation(model_name: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0", quantize: bool = False):
    """Test basic text generation."""
    logger.info(f"Testing generation with model: {model_name}")
    
    # Load model (use smaller models for testing)
    model = get_local_model(model_name, quantize=quantize)
    
    # Test single prompt generation
    prompt = "Explain the concept of recursion in programming in a simple way:"
    logger.info(f"Generating response for: {prompt}")
    
    start_time = time.time()
    response = model.generate(prompt, max_new_tokens=200)
    end_time = time.time()
    
    logger.info(f"Generation took {end_time - start_time:.2f} seconds")
    logger.info(f"Generated response: {response}")
    
    # Test batch generation
    prompts = [
        "What is machine learning?",
        "Explain the difference between supervised and unsupervised learning:",
        "What is transfer learning in AI?"
    ]
    
    logger.info(f"Generating batch responses for {len(prompts)} prompts")
    start_time = time.time()
    responses = model.batch_generate(prompts, max_new_tokens=100, batch_size=2)
    end_time = time.time()
    
    logger.info(f"Batch generation took {end_time - start_time:.2f} seconds")
    for i, response in enumerate(responses):
        logger.info(f"Response {i+1}: {response[:100]}...")
    
    return model

def test_finetuning(model, output_dir: str = "./test_output"):
    """Test basic fine-tuning functionality."""
    logger.info("Testing fine-tuning with tiny dataset")
    
    # Create tiny training dataset
    train_data = [
        {"prompt": "What is the capital of France? ", "response": "The capital of France is Paris."},
        {"prompt": "What is the capital of Japan? ", "response": "The capital of Japan is Tokyo."},
        {"prompt": "What is the capital of Germany? ", "response": "The capital of Germany is Berlin."}
    ]
    
    # Configure fine-tuning arguments
    ft_args = FinetuningArguments(
        train_data=train_data,
        epochs=1,
        batch_size=1,
        learning_rate=5e-5
    )
    
    # Run fine-tuning
    result = model.finetune(ft_args, output_dir=output_dir)
    
    logger.info(f"Fine-tuning result: {result}")
    
    # Test generation with fine-tuned model
    prompt = "What is the capital of Italy? "
    response = model.generate(prompt, max_new_tokens=20)
    logger.info(f"Response after fine-tuning: {response}")

if __name__ == "__main__":
    # Choose a small model for testing
    model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"  # Small model (1.1B parameters)
    # Or try even smaller ones:
    # model_name = "gpt2-medium"  # 355M parameters
    # model_name = "microsoft/phi-2" # 2.7B parameters
    
    # Load and test generation
    model = test_generation(model_name, quantize=False)
    
    # Uncomment to test fine-tuning (will take longer)
    # test_finetuning(model)