"""Local model implementation for fine-tuning and inference."""
from typing import Dict, List, Optional, Union, Any, Deque
import os
import logging
from pathlib import Path
from collections import deque
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer, get_scheduler

from .base import LLMModel

logger = logging.getLogger(__name__)


class LocalModel(LLMModel):
    """A model implementation for local models.
    
    This class provides an interface to use and fine-tune local models.
    It supports initial fine-tuning and online learning.
    """
    
    def __init__(self, model_path: Union[str, Path], device: Optional[str] = None,
                 memory_size: int = 1000, online_batch_size: int = 1,
                 online_learning_rate: float = 1e-5):
        """Initialize the local model.
        
        Args:
            model_path: Path to the local model files
            device: Device to run the model on ('cuda', 'cpu', or None for auto)
        """
        self.model_path = Path(model_path)
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.tokenizer = None
        self.memory_buffer: Deque[Dict[str, str]] = deque(maxlen=memory_size)
        self.online_batch_size = online_batch_size
        self.online_learning_rate = online_learning_rate
        self.optimizer = None
        self.scheduler = None
        self.load_model()
    
    def load_model(self) -> None:
        """Load the model from the specified path."""
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
            self.model = AutoModelForCausalLM.from_pretrained(self.model_path).to(self.device)
            self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.online_learning_rate)
            self.scheduler = get_scheduler(
                "linear",
                optimizer=self.optimizer,
                num_warmup_steps=0,
                num_training_steps=1000
            )
            logger.info(f"Successfully loaded model from {self.model_path} on {self.device}")
        except Exception as e:
            logger.error(f"Failed to load model: {str(e)}")
            raise
    
    def update_memory(self, prompt: str, response: str) -> None:
        """Update the memory buffer with new interaction.
        
        Args:
            prompt: The input prompt
            response: The model's response
        """
        self.memory_buffer.append({"prompt": prompt, "response": response})
        
        # Perform online learning if buffer has enough samples
        if len(self.memory_buffer) >= self.online_batch_size:
            self._online_update()
    
    def _online_update(self) -> None:
        """Perform an online update step using recent interactions."""
        try:
            recent_data = list(self.memory_buffer)[-self.online_batch_size:]
            dataset = self.TrainingDataset(recent_data, self.tokenizer)
            dataloader = DataLoader(dataset, batch_size=self.online_batch_size, shuffle=True)
            
            self.model.train()
            for batch in dataloader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                
                outputs = self.model(input_ids=input_ids,
                                   attention_mask=attention_mask,
                                   labels=input_ids)
                
                loss = outputs.loss
                loss.backward()
                
                # Clip gradients to prevent catastrophic forgetting
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                
                self.optimizer.step()
                self.scheduler.step()
                self.optimizer.zero_grad()
                
                logger.info(f"Online update step completed. Loss: {loss.item():.4f}")
            
            # Save the updated model periodically
            if len(self.memory_buffer) % 100 == 0:
                self.model.save_pretrained(self.model_path)
                self.tokenizer.save_pretrained(self.model_path)
                
        except Exception as e:
            logger.error(f"Online update failed: {str(e)}")
            raise
    
    def generate(self, prompt: str, **kwargs) -> str:
        """Generate a response using the local model.
        
        Args:
            prompt: The input text
            **kwargs: Additional parameters to pass to the model
                max_length: Maximum length of generated text
                temperature: Sampling temperature
                top_p: Nucleus sampling parameter
            
        Returns:
            The generated text response
        """
        try:
            if not self.model or not self.tokenizer:
                raise RuntimeError("Model not loaded. Call load_model() first.")

            max_length = kwargs.get('max_length', 100)
            temperature = kwargs.get('temperature', 0.7)
            top_p = kwargs.get('top_p', 0.9)

            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
            
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_length=max_length,
                    temperature=temperature,
                    top_p=top_p,
                    pad_token_id=self.tokenizer.eos_token_id
                )
            
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            response_text = response[len(prompt):].strip()
            
            # Update memory buffer with the interaction
            self.update_memory(prompt, response_text)
            
            return response_text
            
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
            return [self.generate(prompt, **kwargs) for prompt in prompts]
        except Exception as e:
            logger.error(f"Batch generation failed: {str(e)}")
            raise

    class TrainingDataset(Dataset):
        def __init__(self, data: List[Dict[str, str]], tokenizer):
            self.data = data
            self.tokenizer = tokenizer

        def __len__(self):
            return len(self.data)

        def __getitem__(self, idx):
            item = self.data[idx]
            combined = f"{item['prompt']}{item['response']}"
            encoding = self.tokenizer(combined, truncation=True, max_length=512,
                                    padding='max_length', return_tensors='pt')
            return {
                'input_ids': encoding['input_ids'].squeeze(),
                'attention_mask': encoding['attention_mask'].squeeze()
            }
    
    def fine_tune(self, train_data: List[Dict[str, str]], **kwargs) -> Any:
        """Fine-tune the model on the provided training data.
        
        Args:
            train_data: List of training examples, each containing 'prompt' and 'response'
            **kwargs: Additional fine-tuning parameters
                epochs: Number of training epochs
                batch_size: Training batch size
                learning_rate: Learning rate for optimization
            
        Returns:
            Training results or metrics
        """
        try:
            if not self.model or not self.tokenizer:
                raise RuntimeError("Model not loaded. Call load_model() first.")

            epochs = kwargs.get('epochs', 3)
            batch_size = kwargs.get('batch_size', 4)
            learning_rate = kwargs.get('learning_rate', 2e-5)

            # Prepare dataset and dataloader
            dataset = self.TrainingDataset(train_data, self.tokenizer)
            dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

            # Setup training
            optimizer = torch.optim.AdamW(self.model.parameters(), lr=learning_rate)
            self.model.train()

            total_loss = 0
            for epoch in range(epochs):
                epoch_loss = 0
                for batch in dataloader:
                    input_ids = batch['input_ids'].to(self.device)
                    attention_mask = batch['attention_mask'].to(self.device)

                    outputs = self.model(input_ids=input_ids,
                                       attention_mask=attention_mask,
                                       labels=input_ids)
                    
                    loss = outputs.loss
                    loss.backward()
                    optimizer.step()
                    optimizer.zero_grad()

                    epoch_loss += loss.item()

                avg_epoch_loss = epoch_loss / len(dataloader)
                total_loss += avg_epoch_loss
                logger.info(f"Epoch {epoch+1}/{epochs}, Loss: {avg_epoch_loss:.4f}")

            # Save the fine-tuned model
            self.model.save_pretrained(self.model_path)
            self.tokenizer.save_pretrained(self.model_path)

            return {
                "status": "success",
                "average_loss": total_loss / epochs,
                "epochs_completed": epochs
            }

        except Exception as e:
            logger.error(f"Fine-tuning failed: {str(e)}")
            raise
    
    def get_model_info(self) -> Dict[str, str]:
        """Get information about the local model.
        
        Returns:
            Dictionary with model information
        """
        info = super().get_model_info()
        info.update({
            "type": "local",
            "model_path": str(self.model_path),
            "device": self.device,
            "description": "Local fine-tunable model"
        })
        return info