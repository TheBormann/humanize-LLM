"""Local model implementation for fine-tuning and inference."""
import logging
import os
from typing import Dict, List, Optional, Union

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    Trainer,
    TrainingArguments
)
from datasets import Dataset as HFDataset
from peft import (
    LoraConfig, 
    get_peft_model, 
    prepare_model_for_kbit_training,
    TaskType
)

from .base import LLMModel, FinetuningArguments, PEFTArguments

logger = logging.getLogger(__name__)


class LocalModel(LLMModel):
    """A model implementation for local models.
    
    This class provides an interface to use and fine-tune local models.
    """
    
    def __init__(self, device: str = "cuda" if torch.cuda.is_available() else "cpu"):
        super().__init__(device=device)
        self.model = None
        self.tokenizer = None
        self.is_peft_model = False
        
    def load_model(self, model_path: str, **kwargs) -> None:
        """Load the model from a given path.
        
        Args:
            model_path: Path to the model or model identifier
            **kwargs: Additional arguments for model loading
                quantize: Whether to use quantization (4bit or 8bit)
                quantize_type: Type of quantization ('4bit' or '8bit')
                lora_path: Optional path to LoRA weights
        """
        try:
            # Configure quantization if requested
            model_kwargs = {}
            quantize = kwargs.get('quantize', False)
            quantize_type = kwargs.get('quantize_type', '4bit')
            
            if quantize:
                logger.info(f"Loading model with {quantize_type} quantization")
                if quantize_type == '4bit':
                    model_kwargs['quantization_config'] = BitsAndBytesConfig(
                        load_in_4bit=True,
                        bnb_4bit_compute_dtype=torch.float16,
                        bnb_4bit_quant_type="nf4",
                        bnb_4bit_use_double_quant=True
                    )
                elif quantize_type == '8bit':
                    model_kwargs['load_in_8bit'] = True
                else:
                    logger.warning(f"Unknown quantization type: {quantize_type}, using default")
            
            # Load tokenizer first (doesn't depend on quantization)
            self.tokenizer = AutoTokenizer.from_pretrained(model_path)
            
            # Handle tokenizer configuration
            if not self.tokenizer.pad_token:
                self.tokenizer.pad_token = self.tokenizer.eos_token
                
            # Load the model with configured options
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path, 
                device_map=self.device if self.device != 'cpu' else None,
                **model_kwargs
            )
            
            # Load LoRA weights if provided
            lora_path = kwargs.get('lora_path', None)
            if lora_path and os.path.exists(lora_path):
                logger.info(f"Loading LoRA weights from {lora_path}")
                from peft import PeftModel
                self.model = PeftModel.from_pretrained(self.model, lora_path)
                self.is_peft_model = True
                
            self.model_path = model_path
            logger.info(f"Successfully loaded model from {model_path} on {self.device}")
                
        except Exception as e:
            logger.error(f"Failed to load model: {str(e)}")
            raise
    
    def generate(self, prompt: str, **kwargs) -> str:
        """Generate text response from prompt."""
        if not self.model or not self.tokenizer:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        # Extract generation parameters with defaults
        max_new_tokens = kwargs.get('max_new_tokens', 100)
        temperature = kwargs.get('temperature', 0.7)
        top_p = kwargs.get('top_p', 0.9)
        repetition_penalty = kwargs.get('repetition_penalty', 1.0)

        try:
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
            
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    repetition_penalty=repetition_penalty,
                    pad_token_id=self.tokenizer.eos_token_id
                )
            
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            # Remove the prompt from the response
            return response[len(self.tokenizer.decode(inputs.input_ids[0], skip_special_tokens=True)):].strip()
        except Exception as e:
            logger.error(f"Error during generation: {str(e)}")
            return f"Error: {str(e)}"
            
    def batch_generate(self, prompts: List[str], **kwargs) -> List[str]:
        """Generate responses for multiple prompts efficiently."""
        if not self.model or not self.tokenizer:
            raise RuntimeError("Model not loaded. Call load_model() first.")
            
        if not prompts:
            return []
            
        # Extract generation parameters with defaults
        max_new_tokens = kwargs.get('max_new_tokens', 100)
        temperature = kwargs.get('temperature', 0.7)
        top_p = kwargs.get('top_p', 0.9)
        repetition_penalty = kwargs.get('repetition_penalty', 1.0)
        batch_size = kwargs.get('batch_size', 1)
        
        all_responses = []
        
        # Process prompts in batches
        for i in range(0, len(prompts), batch_size):
            batch_prompts = prompts[i:i+batch_size]
            try:
                # Tokenize all prompts in this batch
                inputs = self.tokenizer(batch_prompts, padding=True, return_tensors="pt").to(self.device)
                
                with torch.no_grad():
                    outputs = self.model.generate(
                        **inputs,
                        max_new_tokens=max_new_tokens,
                        temperature=temperature,
                        top_p=top_p,
                        repetition_penalty=repetition_penalty,
                        pad_token_id=self.tokenizer.eos_token_id
                    )
                
                # Decode outputs and remove prompts
                decoded_outputs = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
                decoded_inputs = self.tokenizer.batch_decode(inputs.input_ids, skip_special_tokens=True)
                
                # Remove prompt from each response
                for j, (output, input_text) in enumerate(zip(decoded_outputs, decoded_inputs)):
                    all_responses.append(output[len(input_text):].strip())
                    
            except Exception as e:
                logger.error(f"Error during batch generation: {str(e)}")
                # Add error responses for this batch
                all_responses.extend([f"Error: {str(e)}"] * len(batch_prompts))
                
        return all_responses

    def finetune(self, training_args: FinetuningArguments, **kwargs) -> Dict:
        """Finetune the model with provided training data.
        
        Args:
            training_args: Arguments for fine-tuning
            **kwargs: Additional arguments:
                output_dir: Directory to save outputs
                peft_args: PEFT configuration arguments
                eval_data: Optional evaluation dataset
        """
        if not self.model or not self.tokenizer:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        
        train_data = training_args.train_data
        if not train_data:
            raise ValueError("Training data must be provided")
            
        # Convert to HuggingFace dataset format
        formatted_data = {
            'prompt': [item['prompt'] for item in train_data],
            'response': [item['response'] for item in train_data]
        }
        dataset = HFDataset.from_dict(formatted_data)
        
        # Check for PEFT arguments
        peft_args = kwargs.get('peft_args', None)
        output_dir = kwargs.get('output_dir', './output')
        
        # Apply PEFT if requested
        if peft_args and isinstance(peft_args, PEFTArguments):
            logger.info(f"Using PEFT method: {peft_args.method}")
            
            if peft_args.method.lower() in ['lora', 'qlora']:
                # Prepare model for kbit training if using quantization
                if getattr(self.model, 'is_loaded_in_8bit', False) or getattr(self.model, 'is_loaded_in_4bit', False):
                    self.model = prepare_model_for_kbit_training(self.model)
                
                # Configure LoRA
                lora_config = LoraConfig(
                    r=peft_args.rank,
                    lora_alpha=peft_args.alpha,
                    lora_dropout=peft_args.dropout,
                    bias="none",
                    task_type=TaskType.CAUSAL_LM,
                    target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
                )
                
                # Apply LoRA adapter
                self.model = get_peft_model(self.model, lora_config)
                self.is_peft_model = True
                
                # Log trainable parameters
                trainable_params = 0
                all_params = 0
                for _, param in self.model.named_parameters():
                    all_params += param.numel()
                    if param.requires_grad:
                        trainable_params += param.numel()
                logger.info(f"Trainable params: {trainable_params:,d} ({100 * trainable_params / all_params:.2f}%)")
        
        # Create a simple training dataset
        def tokenize_function(examples):
            # Combine prompt and response
            texts = [p + r for p, r in zip(examples['prompt'], examples['response'])]
            return self.tokenizer(
                texts, 
                padding="max_length",
                truncation=True,
                max_length=512,
                return_tensors="pt"
            )
        
        tokenized_dataset = dataset.map(
            tokenize_function,
            batched=True,
            remove_columns=['prompt', 'response']
        )
        
        # Configure training arguments
        args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=training_args.epochs,
            per_device_train_batch_size=training_args.batch_size,
            learning_rate=training_args.learning_rate,
            weight_decay=0.01,
            logging_dir=os.path.join(output_dir, 'logs'),
            logging_steps=10,
            save_strategy="epoch"
        )
        
        # Initialize trainer and train
        trainer = Trainer(
            model=self.model,
            args=args,
            train_dataset=tokenized_dataset
        )
        
        try:
            train_result = trainer.train()
            
            # Save the model
            if self.is_peft_model:
                self.model.save_pretrained(output_dir)
            else:
                self.model.save_pretrained(output_dir)
                self.tokenizer.save_pretrained(output_dir)
            
            return {
                "status": "success",
                "loss": train_result.training_loss,
                "epochs": training_args.epochs
            }
        except Exception as e:
            logger.error(f"Error during fine-tuning: {str(e)}")
            return {
                "status": "error",
                "message": str(e)
            }
        
    def save_checkpoint(self, path: str) -> None:
        """Save model checkpoint."""
        if not self.model or not self.tokenizer:
            raise RuntimeError("Model not loaded. Call load_model() first.")
            
        try:
            os.makedirs(path, exist_ok=True)
            
            if self.is_peft_model:
                self.model.save_pretrained(path)
                logger.info(f"PEFT model adapter saved to {path}")
            else:
                self.model.save_pretrained(path)
                self.tokenizer.save_pretrained(path)
                logger.info(f"Model checkpoint saved to {path}")
        except Exception as e:
            logger.error(f"Error saving checkpoint: {str(e)}")
            raise

    def load_checkpoint(self, path: str) -> None:
        """Load model from checkpoint."""
        try:
            # Check if this is a PEFT checkpoint
            if os.path.exists(os.path.join(path, "adapter_config.json")):
                if not self.model:
                    raise RuntimeError("Base model must be loaded before loading a PEFT adapter")
                    
                from peft import PeftModel
                self.model = PeftModel.from_pretrained(self.model, path)
                self.is_peft_model = True
                logger.info(f"PEFT adapter loaded from checkpoint at {path}")
            else:
                # Load as regular model
                self.load_model(path)
                logger.info(f"Model loaded from checkpoint at {path}")
        except Exception as e:
            logger.error(f"Error loading checkpoint: {str(e)}")
            raise
    
    def get_model_info(self) -> Dict[str, str]:
        """Get information about the local model."""
        info = super().get_model_info()
        info.update({
            "type": "local",
            "device": self.device,
            "description": "Local fine-tunable model",
            "is_peft_model": str(self.is_peft_model)
        })
        if hasattr(self, 'model_path'):
            info["model_path"] = str(self.model_path)
        return info