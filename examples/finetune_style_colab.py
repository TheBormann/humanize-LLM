# -*- coding: utf-8 -*-
"""# Fine-tuning Style Transfer Model in Google Colab

This notebook fine-tunes a language model for style transfer using LoRA.
Make sure to:
1. Mount Google Drive
2. Install required dependencies
3. Set up GPU acceleration
"""

# Mount Google Drive
from google.colab import drive
drive.mount('/content/drive')

# Install required dependencies
!pip install -q transformers datasets accelerate peft tqdm pandas

import os
import logging
import sys
import pandas as pd
from typing import List, Dict
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model, TaskType
from datasets import Dataset

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

def create_dataset(training_data: List[Dict[str, str]], tokenizer) -> Dataset:
    """Create a HuggingFace dataset from training pairs."""
    def format_prompt(example):
        return f"### Instruction: Convert the following email to match my writing style.\n\n### Input: {example['prompt']}\n\n### Response: {example['response']}"

    # Format and tokenize the data
    formatted_data = [{
        "text": format_prompt(item)
    } for item in training_data]

    return Dataset.from_list(formatted_data)

def finetune_model(model_name: str, dataset: Dataset, 
                  output_dir: str, device: str = 'cuda'):
    """Fine-tune model on style transfer dataset using LoRA."""
    # Load base model and tokenizer
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map=device,
        trust_remote_code=True,
        use_cache=False
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token

    # Configure LoRA
    lora_config = LoraConfig(
        r=16,  # Rank
        lora_alpha=32,
        target_modules=["query_key_value"],
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM
    )

    # Create PEFT model
    model = get_peft_model(model, lora_config)
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=3,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        fp16=True,
        logging_steps=10,
        save_strategy="epoch",
        warmup_ratio=0.1,
    )

    # Initialize trainer
    trainer = Trainer(
        model=model,
        train_dataset=dataset,
        args=training_args,
        data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False)
    )

    # Train the model
    trainer.train()
    
    # Save the final model
    trainer.save_model(output_dir)
    
    # Test the model
    test_prompt = "Dear Team,\n\nI wanted to remind everyone about our quarterly meeting next Tuesday at 2pm. Please bring your project updates and be prepared to discuss next steps.\n\nRegards,\nManager"
    
    inputs = tokenizer(test_prompt, return_tensors="pt").to(device)
    outputs = model.generate(**inputs, max_new_tokens=300, temperature=0.7)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    logger.info(f"Sample output after fine-tuning:\n{response}")
    
    return model

if __name__ == "__main__":
    # Configuration
    DRIVE_PATH = "/content/drive/MyDrive/LLM_eval"  # Adjust this path as needed
    EMAIL_CSV_PATH = os.path.join(DRIVE_PATH, "data/manual_emails.csv")
    MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.2"  # Using Mistral instead of Ministral
    OUTPUT_DIR = os.path.join(DRIVE_PATH, "output/style_model")
    
    # Create output directories
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Load and prepare dataset
    emails_df = load_emails_from_csv(EMAIL_CSV_PATH)
    training_data = prepare_training_data(emails_df)
    
    # Initialize tokenizer for dataset creation
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    dataset = create_dataset(training_data, tokenizer)
    
    # Fine-tune the model
    finetune_model(MODEL_NAME, dataset, OUTPUT_DIR)