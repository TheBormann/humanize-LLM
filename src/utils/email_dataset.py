import os
import json
import logging
from typing import List, Dict, Any, Optional
import pandas as pd
from tqdm import tqdm

from ..models.huggingface import HuggingFaceModel
from ..models.base import LLMModel

logger = logging.getLogger(__name__)

class EmailDatasetGenerator:
    """Generate training data from emails for fine-tuning."""
    
    def __init__(self, model: LLMModel):
        """Initialize with a model for key point extraction.
        
        Args:
            model: An LLM model instance (HuggingFaceModel, LocalModel, etc.)
        """
        self.model = model
        
    def load_emails_from_csv(self, file_path: str) -> pd.DataFrame:
        """Load emails from a CSV file with columns 'subject', 'body', 'date'."""
        df = pd.read_csv(file_path)
        logger.info(f"Loaded {len(df)} emails from {file_path}")
        return df
    
    def extract_key_points(self, emails: pd.DataFrame) -> pd.DataFrame:
        """Extract key points from emails using an LLM."""
        if not self.model:
            raise ValueError("Model not initialized for key point extraction")
            
        key_points_list = []
        
        # Define prompt template for key point extraction
        prompt_template = """
        Extract the key points from the following email. 
        Return only the essential information that would be needed to reconstruct this email.
        Format the key points as a bullet point list.
        
        EMAIL:
        {email_text}
        
        KEY POINTS:
        """
        
        for _, row in tqdm(emails.iterrows(), total=len(emails), desc="Extracting key points"):
            email_text = f"Subject: {row['subject']}\n\n{row['body']}"
            
            # Call LLM to extract key points
            try:
                response = self.model.generate(
                    prompt=prompt_template.format(email_text=email_text),
                    max_length=300
                )
                key_points_list.append(response)
            except Exception as e:
                logger.error(f"Error extracting key points: {e}")
                key_points_list.append(None)
            
        emails['key_points'] = key_points_list
        return emails
    
    def create_training_pairs(self, emails: pd.DataFrame) -> List[Dict[str, str]]:
        """Create training pairs from emails and key points."""
        training_data = []
        
        for _, row in emails.iterrows():
            # Ensure we have the key points
            if 'key_points' not in row or pd.isna(row['key_points']):
                continue
                
            # Create a training sample
            sample = {
                "prompt": f"Write an email based on these key points:\n{row['key_points']}\n\nEmail:",
                "response": f"{row['body']}"
            }
            
            training_data.append(sample)
            
        logger.info(f"Created {len(training_data)} training pairs")
        return training_data
    
    def save_dataset(self, training_data: List[Dict[str, str]], output_file: str) -> None:
        """Save the training dataset to a JSON file."""
        with open(output_file, 'w') as f:
            json.dump(training_data, f, indent=2)
        logger.info(f"Saved {len(training_data)} training samples to {output_file}")
    
    def generate_dataset(self, email_file: str, output_file: str) -> List[Dict[str, str]]:
        """Generate a complete dataset from emails."""
        emails_df = self.load_emails_from_csv(email_file)
        emails_with_key_points = self.extract_key_points(emails_df)
        training_data = self.create_training_pairs(emails_with_key_points)
        self.save_dataset(training_data, output_file)
        return training_data