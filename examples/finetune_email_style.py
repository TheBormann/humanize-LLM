import os
import logging
import sys
import json
from typing import List, Dict

# Add the parent directory to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.model_utils import get_local_model
from src.utils.email_dataset import EmailDatasetGenerator, EmailDatasetManager
from src.models.huggingface import HuggingFaceModel
from src.models.base import FinetuningArguments, PEFTArguments

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    stream=sys.stdout
)
logger = logging.getLogger(__name__)

def prepare_dataset(email_csv_path: str, output_path: str, hf_model_id: str, hf_api_key: str) -> List[Dict[str, str]]:
    """Prepare dataset from email CSV using a Hugging Face model for key point extraction."""
    # Initialize Hugging Face model
    hf_model = HuggingFaceModel(model_id=hf_model_id, api_key=hf_api_key)
    
    # Initialize dataset generator
    dataset_generator = EmailDatasetGenerator(model=hf_model)
    
    # Generate dataset
    emails_df = dataset_generator.load_emails_from_csv(email_csv_path)
    emails_with_key_points = dataset_generator.extract_key_points(emails_df)
    training_data = dataset_generator.create_training_pairs(emails_with_key_points)
    dataset_generator.save_dataset(training_data, output_path)
    
    return training_data

def load_dataset(dataset_path: str) -> List[Dict[str, str]]:
    """Load dataset from JSON file."""
    with open(dataset_path, 'r') as f:
        return json.load(f)

def finetune_model(model_name: str, dataset: List[Dict[str, str]], 
                  output_dir: str, use_lora: bool = True):
    """Fine-tune model on email dataset."""
    # Load base model
    model = get_local_model(model_name)
    
    # Configure fine-tuning arguments
    ft_args = FinetuningArguments(
        train_data=dataset,
        epochs=3,  # Usually 2-5 epochs works well for style tuning
        batch_size=1,  # Small batch size for memory efficiency
        learning_rate=2e-5  # Lower learning rate to preserve general knowledge
    )
    
    # Configure LoRA if used
    peft_args = None
    if use_lora:
        peft_args = PEFTArguments(
            method="lora",
            rank=16,  # Higher rank for more expressiveness
            alpha=32,
            dropout=0.05
        )
    
    # Run fine-tuning
    result = model.finetune(ft_args, output_dir=output_dir, peft_args=peft_args)
    logger.info(f"Fine-tuning result: {result}")
    
    # Test the model
    test_prompt = "Write an email based on these key points:\n- Request for meeting next week\n- Discuss quarterly results\n- Need preparation materials\n\nEmail:"
    response = model.generate(test_prompt, max_new_tokens=300)
    
    logger.info(f"Sample output after fine-tuning:\n{response}")
    
    return model

if __name__ == "__main__":
    # Configuration
    EMAIL_CSV_PATH = "data/my_emails.csv"
    DATASET_PATH = "data/email_dataset.json"
    MODEL_NAME = "microsoft/phi-2"  # For fine-tuning
    OUTPUT_DIR = "./output/email_style_model"
    
    # Model for key point extraction
    HF_MODEL_ID = "mistralai/mistral-7b-instruct-v0.2"  # Good for summarization/key point extraction
    HF_API_KEY = os.environ.get("HF_API_KEY")
    
    # Create output directories
    os.makedirs(os.path.dirname(DATASET_PATH), exist_ok=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Generate or load dataset
    dataset_manager = EmailDatasetManager(DATASET_PATH)
    if not dataset_manager.dataset:
        logger.info("Generating new dataset...")
        dataset = prepare_dataset(EMAIL_CSV_PATH, DATASET_PATH, HF_MODEL_ID, HF_API_KEY)
        dataset_manager.add_examples(dataset)
        dataset_manager.save_dataset()
    else:
        logger.info("Loading existing dataset...")
        dataset = dataset_manager.dataset
    
    # Fine-tune the model
    finetune_model(MODEL_NAME, dataset, OUTPUT_DIR)