"""
Module for converting datasets to the format expected by the email style app.
"""
import os
import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

def convert_cover_letter_dataset(dataset, output_path, split='train'):
    """
    Convert the cover letter dataset to the format expected by the email style app.
    
    Args:
        dataset: HuggingFace dataset object
        output_path (str): Path to save the converted dataset
        split (str): Dataset split to use (train/test)
        
    Returns:
        str: Path to the saved dataset
    """
    logger.info(f"Converting cover letter dataset (split: {split})")
    
    if split not in dataset:
        raise ValueError(f"Split '{split}' not found in dataset")
    
    # Create the converted data
    converted_data = []
    for item in dataset[split]:
        # Create prompt from job details
        prompt = f"""Create a cover letter with the following information:
Job Title: {item['Job Title']}
Hiring Company: {item['Hiring Company']}
Applicant: {item['Applicant Name']}
Experience: {item['Past Working Experience']}, {item['Current Working Experience']}
Skills: {item['Skillsets']}
Qualifications: {item['Qualifications']}
Preferred Qualifications: {item['Preferred Qualifications']}
"""
        # Response is the actual cover letter
        response = item['Cover Letter']
        
        converted_data.append({
            "prompt": prompt,
            "response": response
        })
    
    # Ensure the directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Save as JSON
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(converted_data, f, indent=2, ensure_ascii=False)
    
    logger.info(f"Saved {len(converted_data)} examples to {output_path}")
    return output_path