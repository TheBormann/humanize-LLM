"""
Script to download and process the cover letter dataset.
"""
import os
import logging
import argparse
from pathlib import Path

from download import download_hf_dataset
from convert import convert_cover_letter_dataset
from analyze import dataset_statistics
from utils import split_dataset

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(description="Process cover letter dataset")
    parser.add_argument("--dataset", default="ShashiVish/cover-letter-dataset", help="HuggingFace dataset ID")
    parser.add_argument("--output", default="data/cover_letters.json", help="Output path for processed dataset")
    parser.add_argument("--split", default="train", choices=["train", "test", "all"], help="Dataset split to use")
    parser.add_argument("--do_split", action="store_true", help="Split the dataset into train/val/test")
    args = parser.parse_args()

    # Ensure output directory exists
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    
    # Download dataset
    dataset = download_hf_dataset(args.dataset)
    
    # Process dataset
    if args.split == "all":
        # Process both train and test splits and combine
        train_path = convert_cover_letter_dataset(dataset, "data/cover_letters_train_tmp.json", "train")
        test_path = convert_cover_letter_dataset(dataset, "data/cover_letters_test_tmp.json", "test")
        
        # Combine datasets
        import json
        with open(train_path, 'r') as f:
            train_data = json.load(f)
        with open(test_path, 'r') as f:
            test_data = json.load(f)
        
        combined_data = train_data + test_data
        
        with open(args.output, 'w') as f:
            json.dump(combined_data, f, indent=2)
        
        # Clean up temporary files
        os.remove(train_path)
        os.remove(test_path)
        
        logger.info(f"Combined dataset saved to {args.output} with {len(combined_data)} examples")
    else:
        output_path = convert_cover_letter_dataset(dataset, args.output, args.split)
        logger.info(f"Processed dataset saved to {output_path}")
    
    # Generate statistics
    stats = dataset_statistics(args.output)
    logger.info("Dataset statistics:")
    for key, value in stats.items():
        logger.info(f"  {key}: {value}")
    
    # Split dataset if requested
    if args.do_split:
        train_path, val_path, test_path = split_dataset(args.output)
        logger.info(f"Dataset split into: {train_path}, {val_path}, {test_path}")

if __name__ == "__main__":
    main()