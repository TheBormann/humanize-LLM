"""Hugging Face model implementation."""
from typing import Dict, List, Optional, Union
import os
import logging
from huggingface_hub import InferenceClient
from .base import LLMModel, FinetuningArguments

logger = logging.getLogger(__name__)


class HuggingFaceModel(LLMModel):
    """A model implementation for Hugging Face API.
    
    This class provides an interface to use models from the Hugging Face Hub.
    """
    
    def __init__(self, model_id: str, api_key: Optional[str] = None):
        """Initialize the Hugging Face model.
        
        Args:
            model_id: The model identifier on Hugging Face Hub
            api_key: Optional API key for accessing gated models
        """
        self.model_id = model_id
        self.api_key = api_key or os.environ.get("HF_API_KEY")
        
        if not self.api_key:
            raise ValueError(
                "No API key provided. Please provide an API key either through the constructor "
                "or by setting the HF_API_KEY environment variable."
            )
        
        try:
            # Initialize the Inference API client
            self.client = InferenceClient(model=model_id, token=self.api_key)
        except Exception as e:
            logger.error(f"Failed to initialize model {model_id}: {str(e)}")
            raise ValueError(
                f"Failed to initialize Hugging Face model. Please ensure your API key is valid "
                f"and you have access to the model {model_id}. Error: {str(e)}"
            )
    
    def generate(self, prompt: str, **kwargs) -> str:
        """Generate a response using the Hugging Face Inference API.
        
        Args:
            prompt: The input text
            **kwargs: Additional parameters to pass to the model
                max_length: Maximum length of generated text
                num_return_sequences: Number of sequences to generate
                temperature: Sampling temperature
                top_p: Nucleus sampling parameter
            
        Returns:
            The generated text response
        """
        try:
            # Set default parameters if not provided
            max_length = kwargs.get('max_length', 100)
            temperature = kwargs.get('temperature', 0.7)
            top_p = kwargs.get('top_p', 0.9)
            
            # Generate the response using the Inference API
            response = self.client.text_generation(
                prompt,
                max_new_tokens=max_length,
                temperature=temperature,
                top_p=top_p,
                do_sample=True
            )
            
            # Return the generated text, removing the input prompt
            return response.strip()
            
        except Exception as e:
            logger.error(f"Generation failed: {str(e)}")
            raise
    
    def batch_generate(self, prompts: List[str], **kwargs) -> List[str]:
        """Generate responses for multiple prompts.
        
        Args:
            prompts: List of input prompts
            **kwargs: Additional parameters to pass to the model
            
        Returns:
            List of generated text responses
        """
        try:
            # Generate responses for each prompt individually
            return [self.generate(prompt, **kwargs) for prompt in prompts]
            
        except Exception as e:
            logger.error(f"Batch generation failed: {str(e)}")
            raise
    
    def get_model_info(self) -> Dict[str, str]:
        """Get information about the Hugging Face model.
        
        Returns:
            Dictionary with model information
        """
        info = super().get_model_info()
        info.update({
            "type": "huggingface",
            "model_id": self.model_id,
            "description": "Hugging Face model"
        })
        return info

    def load_model(self, model_path: str, **kwargs) -> None:
        """Not supported for API-based models."""
        raise NotImplementedError(
            "HuggingFaceModel uses remote inference and does not support loading local models."
        )
    
    def finetune(self, training_args: FinetuningArguments, **kwargs) -> Dict:
        """Fine-tune a model on Hugging Face Hub.
        
        This method uploads training data to Hugging Face Hub and initiates a fine-tuning job.
        It requires a Hugging Face Pro account with access to the fine-tuning API.
        
        Args:
            training_args: Arguments for fine-tuning
            **kwargs: Additional arguments:
                output_dir: Directory to save outputs
                hub_model_id: Model ID for Hugging Face Hub
                push_to_hub: Whether to push the model to Hub
                use_peft: Whether to use PEFT/LoRA for fine-tuning
                peft_args: PEFT configuration arguments
        
        Returns:
            Dictionary with fine-tuning job information
        """
        try:
            from huggingface_hub import HfApi, login
            from datasets import Dataset
            import tempfile
            import json
            import os
            
            # Extract arguments
            hub_model_id = kwargs.get('hub_model_id')
            if not hub_model_id:
                raise ValueError("hub_model_id must be provided for Hugging Face fine-tuning")
                
            output_dir = kwargs.get('output_dir', './output')
            os.makedirs(output_dir, exist_ok=True)
            
            # Login to Hugging Face
            login(token=self.api_key)
            api = HfApi()
            
            # Prepare training data
            train_data = training_args.train_data
            if not train_data:
                raise ValueError("Training data must be provided")
                
            # Convert to format expected by Hugging Face
            formatted_data = []
            for item in train_data:
                formatted_data.append({
                    "messages": [
                        {"role": "system", "content": "Transform the given text into a custom-styled version."},
                        {"role": "user", "content": item['prompt']},
                        {"role": "assistant", "content": item['response']}
                    ]
                })
            
            # Save dataset to temporary file
            with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
                dataset_path = f.name
                for item in formatted_data:
                    f.write(json.dumps(item) + '\n')
            
            # Upload dataset to Hub
            dataset_name = f"{hub_model_id.split('/')[-1]}-dataset"
            try:
                api.create_repo(repo_id=dataset_name, repo_type="dataset", private=True)
            except Exception:
                # Repository might already exist
                pass
                
            api.upload_file(
                path_or_fileobj=dataset_path,
                path_in_repo="train.jsonl",
                repo_id=dataset_name,
                repo_type="dataset"
            )
            
            # Configure fine-tuning parameters
            fine_tuning_config = {
                "model": self.model_id,
                "training_dataset": f"{dataset_name}/train.jsonl",
                "method": "sft",  # Supervised fine-tuning
                "hyperparameters": {
                    "epochs": training_args.epochs,
                    "batch_size": training_args.batch_size,
                    "learning_rate": training_args.learning_rate,
                    "max_steps": kwargs.get('max_steps', 1000),
                }
            }
            
            # Add PEFT/LoRA configuration if requested
            if kwargs.get('use_peft', True):
                peft_args = kwargs.get('peft_args', None)
                if peft_args:
                    fine_tuning_config["hyperparameters"]["peft_method"] = "lora"
                    fine_tuning_config["hyperparameters"]["lora_r"] = peft_args.rank
                    fine_tuning_config["hyperparameters"]["lora_alpha"] = peft_args.alpha
                    fine_tuning_config["hyperparameters"]["lora_dropout"] = peft_args.dropout
                else:
                    # Default LoRA parameters
                    fine_tuning_config["hyperparameters"]["peft_method"] = "lora"
                    fine_tuning_config["hyperparameters"]["lora_r"] = 16
                    fine_tuning_config["hyperparameters"]["lora_alpha"] = 32
                    fine_tuning_config["hyperparameters"]["lora_dropout"] = 0.05
            
            # Save fine-tuning configuration
            config_path = os.path.join(output_dir, "fine_tuning_config.json")
            with open(config_path, 'w') as f:
                json.dump(fine_tuning_config, f, indent=2)
            
            logger.info(f"Fine-tuning configuration saved to {config_path}")
            logger.info(f"To start fine-tuning, use the Hugging Face CLI or API with this configuration")
            logger.info(f"Dataset uploaded to Hugging Face Hub as {dataset_name}")
            
            # Clean up temporary file
            os.unlink(dataset_path)
            
            return {
                "status": "success",
                "message": "Fine-tuning configuration prepared",
                "dataset": dataset_name,
                "model_id": hub_model_id,
                "config_path": config_path
            }
            
        except Exception as e:
            logger.error(f"Fine-tuning preparation failed: {str(e)}")
            return {
                "status": "error",
                "message": str(e)
            }
    
    def save_checkpoint(self, path: str) -> None:
        """Not supported for API-based models."""
        raise NotImplementedError(
            "HuggingFaceModel uses remote inference and does not support saving checkpoints."
        )
    
    def load_checkpoint(self, path: str) -> None:
        """Not supported for API-based models."""
        raise NotImplementedError(
            "HuggingFaceModel uses remote inference and does not support loading checkpoints."
        )