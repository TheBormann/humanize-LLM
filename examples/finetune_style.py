import os
import logging
import sys
import pandas as pd
from typing import List, Dict

# Add the parent directory to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.model_utils import get_local_model
from src.models.base import FinetuningArguments, PEFTArguments

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    stream=sys.stdout
)
logger = logging.getLogger(__name__)

def load_emails_from_csv(file_path: str) -> pd.DataFrame:
    """Load emails from a CSV file with semicolon delimiter."""
    df = pd.read_csv(file_path, sep=';')
    logger.info(f"Loaded {len(df)} emails from {file_path}")
    return df

def prepare_training_data(emails_df: pd.DataFrame) -> List[Dict[str, str]]:
    """Prepare training data from emails dataframe.
    
    Creates direct style transfer pairs where:
    - prompt: [AI-generated email from body_ai]
    - response: [Your styled version from body]
    """
    training_data = []
    
    for _, row in emails_df.iterrows():
        if pd.isna(row['body']) or pd.isna(row['body_ai']):
            continue
            
        sample = {
            "prompt": row['body_ai'],  # Direct AI-generated email
            "response": row['body']     # Your styled version
        }
        
        training_data.append(sample)
    
    logger.info(f"Created {len(training_data)} direct style transfer pairs")
    return training_data

def finetune_model(model_name: str, dataset: List[Dict[str, str]], 
                  output_dir: str, use_lora: bool = True):
    """Fine-tune model on pirate-style email dataset."""
    # Load base model
    model = get_local_model(model_name)
    
    # Load the model
    model.load_model(model_name)
    
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
    
    # Test the model with direct input
    test_prompt = "Dear Team,\n\nI wanted to remind everyone about our quarterly meeting next Tuesday at 2pm. Please bring your project updates and be prepared to discuss next steps.\n\nRegards,\nManager"
    response = model.generate(test_prompt, max_new_tokens=300)
    
    logger.info(f"Sample output after fine-tuning:\n{response}")
    
    return model

if __name__ == "__main__":
    # Configuration
    EMAIL_CSV_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "manual_emails.csv")
    MODEL_NAME = "ministral/Ministral-3b-instruct"  # For fine-tuning
    OUTPUT_DIR = "../output/pirate_style_model"
    
    # Create output directories
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Load and prepare dataset
    emails_df = load_emails_from_csv(EMAIL_CSV_PATH)
    training_data = prepare_training_data(emails_df)
    
    # Fine-tune the model
    finetune_model(MODEL_NAME, training_data, OUTPUT_DIR, use_lora=True)