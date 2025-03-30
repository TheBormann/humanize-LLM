#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Fine-tune a language model using Hugging Face's cloud-based fine-tuning service.

This script provides a complete workflow for:
1. Loading and preparing data
2. Uploading the dataset to Hugging Face Hub
3. Configuring and initiating a fine-tuning job on Hugging Face
4. Setting up inference via Hugging Face API

Requires a Hugging Face Pro account with access to the fine-tuning API.
"""

import os
import logging
import sys
import argparse
import pandas as pd
import json
import tempfile
from typing import List, Dict, Optional, Union
from datasets import Dataset

# Add the parent directory to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import Hugging Face components
from huggingface_hub import HfApi, login
from src.models.huggingface import HuggingFaceModel
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


def prepare_training_data(emails_df: pd.DataFrame) -> List[Dict]:
    """Prepare training data in conversational format for fine-tuning.

    Creates direct style transfer pairs in conversational format:
    - system: instruction on style transformation
    - user: original AI-generated email
    - assistant: styled version
    """
    system_message = """Transform the given email into a custom-styled version that maintains the same content but uses a more personal, unique tone.
    Your goal is to make the text feel more human-written with natural speech patterns."""

    training_samples = []

    for _, row in emails_df.iterrows():
        if pd.isna(row['body']) or pd.isna(row['body_ai']):
            continue

        # Create conversation in the format expected by Hugging Face fine-tuning
        sample = {
            "messages": [
                {"role": "system", "content": system_message},
                {"role": "user", "content": row['body_ai']},  # AI-generated email
                {"role": "assistant", "content": row['body']}  # Custom style version
            ]
        }

        training_samples.append(sample)

    logger.info(f"Created {len(training_samples)} conversational training samples")
    return training_samples


def upload_dataset_to_hub(training_data: List[Dict], dataset_name: str, hf_token: str) -> str:
    """Upload the training dataset to Hugging Face Hub.
    
    Args:
        training_data: List of training samples
        dataset_name: Name for the dataset on Hugging Face Hub
        hf_token: Hugging Face API token
        
    Returns:
        Dataset path on Hugging Face Hub
    """
    # Login to Hugging Face
    login(token=hf_token)
    api = HfApi()
    
    # Save dataset to temporary file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
        dataset_path = f.name
        for item in training_data:
            f.write(json.dumps(item) + '\n')
    
    # Create or get repository
    try:
        api.create_repo(repo_id=dataset_name, repo_type="dataset", private=True)
        logger.info(f"Created new dataset repository: {dataset_name}")
    except Exception as e:
        logger.info(f"Repository {dataset_name} already exists or couldn't be created: {str(e)}")
    
    # Upload dataset file
    api.upload_file(
        path_or_fileobj=dataset_path,
        path_in_repo="train.jsonl",
        repo_id=dataset_name,
        repo_type="dataset"
    )
    logger.info(f"Uploaded dataset to {dataset_name}/train.jsonl")
    
    # Clean up temporary file
    os.unlink(dataset_path)
    
    return f"{dataset_name}/train.jsonl"


def configure_fine_tuning_job(model_id: str, dataset_path: str, 
                             epochs: int = 3, batch_size: int = 1, 
                             learning_rate: float = 2e-5,
                             use_peft: bool = True, peft_args: Optional[PEFTArguments] = None) -> Dict:
    """Configure a fine-tuning job for Hugging Face.
    
    Args:
        model_id: Base model to fine-tune
        dataset_path: Path to the dataset on Hugging Face Hub
        epochs: Number of training epochs
        batch_size: Training batch size
        learning_rate: Learning rate
        use_peft: Whether to use PEFT/LoRA
        peft_args: PEFT configuration arguments
        
    Returns:
        Fine-tuning configuration dictionary
    """
    # Basic configuration
    config = {
        "model": model_id,
        "training_dataset": dataset_path,
        "method": "sft",  # Supervised fine-tuning
        "hyperparameters": {
            "epochs": epochs,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "max_steps": 1000,  # Can be adjusted based on dataset size
        }
    }
    
    # Add PEFT/LoRA configuration if requested
    if use_peft:
        if peft_args:
            config["hyperparameters"]["peft_method"] = "lora"
            config["hyperparameters"]["lora_r"] = peft_args.rank
            config["hyperparameters"]["lora_alpha"] = peft_args.alpha
            config["hyperparameters"]["lora_dropout"] = peft_args.dropout
        else:
            # Default LoRA parameters
            config["hyperparameters"]["peft_method"] = "lora"
            config["hyperparameters"]["lora_r"] = 16
            config["hyperparameters"]["lora_alpha"] = 32
            config["hyperparameters"]["lora_dropout"] = 0.05
    
    return config


def setup_inference_api(model_id: str, hf_token: str) -> str:
    """Set up inference API for the model on Hugging Face Hub.
    
    Args:
        model_id: Model ID on Hugging Face Hub
        hf_token: Hugging Face API token
        
    Returns:
        API endpoint URL
    """
    try:
        # Login to Hugging Face
        login(token=hf_token)
        
        # Get the API
        api = HfApi()
        
        # Enable inference API for the model
        api.update_repo_visibility(
            repo_id=model_id,
            private=False,  # Make the model public
        )
        
        logger.info(f"Model {model_id} is now available via Hugging Face Inference API")
        logger.info(f"You can access it at: https://huggingface.co/{model_id}")
        
        # Return the API endpoint
        return f"https://api-inference.huggingface.co/models/{model_id}"
    
    except Exception as e:
        logger.error(f"Failed to set up inference API: {str(e)}")
        return None


def test_inference_api(api_endpoint: str, prompt: str, hf_token: str) -> str:
    """Test the inference API with a sample prompt.
    
    Args:
        api_endpoint: API endpoint URL
        prompt: Test prompt
        hf_token: Hugging Face API token
        
    Returns:
        Generated response
    """
    import requests
    import json
    
    headers = {
        "Authorization": f"Bearer {hf_token}",
        "Content-Type": "application/json"
    }
    
    data = {
        "inputs": prompt,
        "parameters": {
            "max_new_tokens": 200,
            "temperature": 0.7,
            "top_p": 0.9,
            "do_sample": True
        }
    }
    
    response = requests.post(api_endpoint, headers=headers, json=data)
    
    if response.status_code == 200:
        return response.json()[0]["generated_text"]
    else:
        return f"Error: {response.status_code} - {response.text}"


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Fine-tune a language model using Hugging Face's cloud service")
    parser.add_argument("--data_path", type=str, default="./data/manual_emails.csv", 
                        help="Path to the CSV file with email data")
    parser.add_argument("--model_id", type=str, default="TinyLlama/TinyLlama-1.1B-Chat-v1.0", 
                        help="Base model to fine-tune")
    parser.add_argument("--output_dir", type=str, default="./output/huggingface_finetune", 
                        help="Directory to save configuration and results")
    parser.add_argument("--epochs", type=int, default=3, 
                        help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=1, 
                        help="Training batch size")
    parser.add_argument("--learning_rate", type=float, default=2e-5, 
                        help="Learning rate")
    parser.add_argument("--use_peft", action="store_true", default=True,
                        help="Use PEFT/LoRA for fine-tuning")
    parser.add_argument("--hub_model_id", type=str, required=True,
                        help="Model ID for Hugging Face Hub (e.g., 'username/model-name')")
    parser.add_argument("--hf_token", type=str, 
                        help="Hugging Face API token")
    parser.add_argument("--test_prompt", type=str, 
                        default="Dear Team,\n\nI wanted to remind everyone about our quarterly meeting next Tuesday at 2pm. Please bring your project updates and be prepared to discuss next steps.\n\nRegards,\nManager", 
                        help="Test prompt for the fine-tuned model")
    
    args = parser.parse_args()
    
    # Get Hugging Face token
    hf_token = args.hf_token or os.environ.get("HF_API_KEY")
    if not hf_token:
        raise ValueError("Hugging Face API token must be provided either through --hf_token or HF_API_KEY environment variable")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load and prepare dataset
    emails_df = load_emails_from_csv(args.data_path)
    training_data = prepare_training_data(emails_df)
    
    # Create dataset name from model ID
    dataset_name = f"{args.hub_model_id.split('/')[-1]}-dataset"
    
    # Upload dataset to Hugging Face Hub
    dataset_path = upload_dataset_to_hub(training_data, dataset_name, hf_token)
    
    # Configure fine-tuning job
    peft_args = PEFTArguments(rank=16, alpha=32, dropout=0.05) if args.use_peft else None
    fine_tuning_config = configure_fine_tuning_job(
        model_id=args.model_id,
        dataset_path=dataset_path,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        use_peft=args.use_peft,
        peft_args=peft_args
    )
    
    # Save fine-tuning configuration
    config_path = os.path.join(args.output_dir, "fine_tuning_config.json")
    with open(config_path, 'w') as f:
        json.dump(fine_tuning_config, f, indent=2)
    
    logger.info(f"Fine-tuning configuration saved to {config_path}")
    logger.info("To start fine-tuning, use the Hugging Face CLI or API with this configuration")
    logger.info(f"Dataset uploaded to Hugging Face Hub as {dataset_path}")
    
    # Provide instructions for starting the fine-tuning job
    logger.info("\nTo start the fine-tuning job, you can use the Hugging Face CLI:")
    logger.info(f"huggingface-cli login --token {hf_token}")
    logger.info(f"huggingface-cli fine-tune --config {config_path} --push-to-hub {args.hub_model_id}")
    
    # Or provide instructions for using the HuggingFaceModel class
    logger.info("\nAlternatively, you can use the HuggingFaceModel class in your code:")
    logger.info("from src.models.huggingface import HuggingFaceModel")
    logger.info("from src.models.base import FinetuningArguments")
    logger.info("")
    logger.info("# Initialize the model")
    logger.info(f"model = HuggingFaceModel(model_id='{args.model_id}', api_key='{hf_token}')")
    logger.info("")
    logger.info("# Prepare training arguments")
    logger.info("training_args = FinetuningArguments(")
    logger.info("    train_data=training_data,")
    logger.info(f"    epochs={args.epochs},")
    logger.info(f"    batch_size={args.batch_size},")
    logger.info(f"    learning_rate={args.learning_rate}")
    logger.info(")")
    logger.info("")
    logger.info("# Fine-tune the model")
    logger.info("result = model.finetune(")
    logger.info("    training_args,")
    logger.info(f"    output_dir='{args.output_dir}',")
    logger.info(f"    hub_model_id='{args.hub_model_id}',")
    logger.info("    push_to_hub=True,")
    logger.info(f"    use_peft={str(args.use_peft)}")
    logger.info(")")
    
    # Setup inference API
    logger.info("\nAfter fine-tuning is complete, you can set up the inference API:")
    logger.info(f"api_endpoint = setup_inference_api('{args.hub_model_id}', '{hf_token}')")
    
    # Provide example curl command for inference
    logger.info("\nExample curl command for inference:")
    logger.info(f"curl -X POST \\ \n  https://api-inference.huggingface.co/models/{args.hub_model_id} \\ \n  -H 'Authorization: Bearer {hf_token}' \\ \n  -H 'Content-Type: application/json' \\ \n  -d '{{\"inputs\": \"{args.test_prompt.replace('\n', '\\n')}\"}}'")


if __name__ == "__main__":
    main()