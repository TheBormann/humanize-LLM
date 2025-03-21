import os
import logging
import sys
import json
import pandas as pd
import gradio as gr
import time
from typing import List, Dict, Tuple, Optional
import torch
from pathlib import Path

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

# Global variables
DATASET_DIR = "data"
MODELS_DIR = "output"
DEFAULT_HF_MODEL = "mistralai/mistral-7b-instruct-v0.2"
DEFAULT_LOCAL_MODEL = "microsoft/phi-2"

# Ensure directories exist
os.makedirs(DATASET_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

# Global state
current_model = None
email_dataset = None
dataset_manager = None

# Dataset Management Functions
def upload_csv(file):
    """Process uploaded CSV file and extract emails."""
    try:
        df = pd.read_csv(file.name)
        required_cols = ['subject', 'body']
        if not all(col in df.columns for col in required_cols):
            return None, f"Error: CSV must contain columns: {', '.join(required_cols)}"
        
        return df, f"Successfully loaded {len(df)} emails"
    except Exception as e:
        return None, f"Error loading CSV: {str(e)}"

def extract_key_points(df, hf_model_id, api_key, dataset_name):
    """Extract key points from emails using the specified model."""
    if df is None:
        return None, "No data loaded"
    
    try:
        # Initialize model
        hf_model = HuggingFaceModel(model_id=hf_model_id, api_key=api_key)
        
        # Create dataset generator
        generator = EmailDatasetGenerator(model=hf_model)
        
        # Process emails
        emails_with_key_points = generator.extract_key_points(df)
        
        # Create training pairs
        training_pairs = generator.create_training_pairs(emails_with_key_points)
        
        # Save dataset
        dataset_path = os.path.join(DATASET_DIR, f"{dataset_name}.json")
        global dataset_manager
        dataset_manager = EmailDatasetManager(dataset_path)
        dataset_manager.add_examples(training_pairs)
        dataset_manager.save_dataset()
        
        return training_pairs, f"Created {len(training_pairs)} training examples. Saved to {dataset_name}.json"
    except Exception as e:
        return None, f"Error extracting key points: {str(e)}"

def view_dataset(dataset_name):
    """Load and view a dataset."""
    try:
        dataset_path = os.path.join(DATASET_DIR, f"{dataset_name}.json")
        global dataset_manager
        dataset_manager = EmailDatasetManager(dataset_path)
        return dataset_manager.get_dataset(), f"Loaded {len(dataset_manager.get_dataset())} examples"
    except Exception as e:
        return None, f"Error loading dataset: {str(e)}"

def display_dataset(dataset):
    """Format dataset for display."""
    if not dataset:
        return "No data to display"
    
    html = "<div style='max-height: 500px; overflow-y: auto;'>"
    for i, item in enumerate(dataset[:10]):  # Show first 10 examples
        html += f"<div style='margin-bottom: 20px; padding: 10px; border: 1px solid #ddd;'>"
        html += f"<h4>Example {i+1}</h4>"
        html += f"<p><strong>Prompt:</strong><br>{item['prompt'].replace('\n', '<br>')}</p>"
        html += f"<p><strong>Response:</strong><br>{item['response'].replace('\n', '<br>')}</p>"
        html += "</div>"
    
    if len(dataset) > 10:
        html += f"<p>...and {len(dataset) - 10} more examples</p>"
    
    html += "</div>"
    return html

# Fine-tuning Functions
def run_finetune(dataset_name, model_name, output_name, epochs, batch_size, learning_rate, use_lora):
    """Run fine-tuning on the selected dataset."""
    try:
        # Load dataset
        dataset_path = os.path.join(DATASET_DIR, f"{dataset_name}.json")
        global dataset_manager
        dataset_manager = EmailDatasetManager(dataset_path)
        dataset = dataset_manager.get_dataset()
        
        if not dataset:
            return f"Error: No data in dataset {dataset_name}"
        
        # Prepare output directory
        output_dir = os.path.join(MODELS_DIR, output_name)
        os.makedirs(output_dir, exist_ok=True)
        
        # Configure fine-tuning arguments
        ft_args = FinetuningArguments(
            train_data=dataset,
            epochs=int(epochs),
            batch_size=int(batch_size),
            learning_rate=float(learning_rate)
        )
        
        # Configure LoRA if requested
        peft_args = None
        if use_lora:
            peft_args = PEFTArguments(
                method="lora",
                rank=16,
                alpha=32,
                dropout=0.05
            )
        
        # Load model
        model = get_local_model(model_name)
        
        # Run fine-tuning
        result = model.finetune(ft_args, output_dir=output_dir, peft_args=peft_args)
        
        # Save model globally
        global current_model
        current_model = model
        
        return f"Fine-tuning completed! Model saved to {output_dir}\nLoss: {result.get('loss', 'N/A')}"
    except Exception as e:
        return f"Error during fine-tuning: {str(e)}"

def list_available_datasets():
    """List available datasets in the data directory."""
    try:
        files = [f.replace('.json', '') for f in os.listdir(DATASET_DIR) if f.endswith('.json')]
        return files or ["No datasets found"]
    except Exception:
        return ["Error listing datasets"]

def list_available_models():
    """List available fine-tuned models."""
    try:
        models = [d for d in os.listdir(MODELS_DIR) if os.path.isdir(os.path.join(MODELS_DIR, d))]
        return models or ["No models found"]
    except Exception:
        return ["Error listing models"]

# Text Generation Functions
def load_model_for_generation(model_name):
    """Load a fine-tuned model for text generation."""
    try:
        model_path = os.path.join(MODELS_DIR, model_name)
        if not os.path.exists(model_path):
            return f"Model {model_name} not found"
        
        # Load the model
        global current_model
        current_model = get_local_model(model_path)
        
        return f"Model {model_name} loaded successfully"
    except Exception as e:
        return f"Error loading model: {str(e)}"

def generate_text(key_points, model_name, max_length, temperature):
    """Generate text based on key points using the selected model."""
    try:
        # Load model if not already loaded or if a different one is selected
        global current_model
        model_path = os.path.join(MODELS_DIR, model_name)
        
        if not current_model or current_model.get_model_info().get("model_path") != model_path:
            msg = load_model_for_generation(model_name)
            if "Error" in msg or "not found" in msg:
                return msg
        
        # Format prompt
        prompt = f"Write an email based on these key points:\n{key_points}\n\nEmail:"
        
        # Generate response
        response = current_model.generate(
            prompt, 
            max_new_tokens=int(max_length),
            temperature=float(temperature)
        )
        
        return response
    except Exception as e:
        return f"Error generating text: {str(e)}"

# Define Gradio interface
with gr.Blocks(title="Email Style Trainer") as app:
    gr.Markdown("# Email Style Model Training App")
    
    with gr.Tabs():
        # Dataset Creation Tab
        with gr.Tab("Dataset Creation"):
            gr.Markdown("## Create training dataset from emails")
            
            with gr.Row():
                with gr.Column():
                    # Inputs for dataset creation
                    csv_upload = gr.File(label="Upload CSV with emails (needs 'subject' and 'body' columns)")
                    hf_api_key = gr.Textbox(label="Hugging Face API Key", type="password")
                    hf_model_id = gr.Textbox(label="HF Model for Key Point Extraction", value=DEFAULT_HF_MODEL)
                    dataset_name = gr.Textbox(label="Dataset Name", value="my_email_dataset")
                    
                    with gr.Row():
                        upload_btn = gr.Button("Upload & Parse CSV")
                        extract_btn = gr.Button("Extract Key Points")
                
                with gr.Column():
                    # Display results
                    csv_status = gr.Textbox(label="CSV Upload Status")
                    dataset_status = gr.Textbox(label="Dataset Creation Status")
                    
                    # Dataset viewer
                    dataset_selector = gr.Dropdown(label="Select Dataset to View", choices=list_available_datasets())
                    refresh_datasets_btn = gr.Button("Refresh Datasets")
                    view_dataset_btn = gr.Button("View Dataset")
                    dataset_display = gr.HTML(label="Dataset Preview")
            
            # Connect functions
            upload_btn.click(upload_csv, inputs=[csv_upload], outputs=[csv_status])
            extract_btn.click(
                lambda df_status, model, key, name: extract_key_points(
                    df_status[0] if isinstance(df_status, tuple) and df_status[0] is not None else None,
                    model, key, name
                ),
                inputs=[csv_status, hf_model_id, hf_api_key, dataset_name],
                outputs=[dataset_status]  # Only include valid Gradio components
            )
            refresh_datasets_btn.click(
                lambda: gr.Dropdown(choices=list_available_datasets()), 
                outputs=[dataset_selector]
            )
            view_dataset_btn.click(
                lambda name: display_dataset(view_dataset(name)[0]),
                inputs=[dataset_selector],
                outputs=[dataset_display]
            )
        
        # Fine-tuning Tab
        with gr.Tab("Fine-tuning"):
            gr.Markdown("## Fine-tune model on email dataset")
            
            with gr.Row():
                with gr.Column():
                    # Fine-tuning inputs
                    ft_dataset_selector = gr.Dropdown(
                        label="Select Dataset", 
                        choices=list_available_datasets()
                    )
                    ft_refresh_btn = gr.Button("Refresh Datasets")
                    
                    ft_model = gr.Textbox(
                        label="Base Model", 
                        value=DEFAULT_LOCAL_MODEL
                    )
                    ft_output = gr.Textbox(
                        label="Output Model Name", 
                        value="my_email_model"
                    )
                    
                    with gr.Row():
                        ft_epochs = gr.Slider(label="Epochs", minimum=1, maximum=10, value=3, step=1)
                        ft_batch_size = gr.Slider(label="Batch Size", minimum=1, maximum=8, value=1, step=1)
                    
                    ft_lr = gr.Slider(
                        label="Learning Rate", 
                        minimum=1e-6, 
                        maximum=1e-4, 
                        value=2e-5, 
                        step=1e-6
                    )
                    ft_lora = gr.Checkbox(label="Use LoRA", value=True)
                    
                    ft_start_btn = gr.Button("Start Fine-tuning")
                
                with gr.Column():
                    # Fine-tuning status
                    ft_status = gr.Textbox(label="Fine-tuning Status", lines=10)
        
        # Generation Tab
        with gr.Tab("Text Generation"):
            gr.Markdown("## Generate emails with your fine-tuned model")
            
            with gr.Row():
                with gr.Column():
                    # Generation inputs
                    gen_model_selector = gr.Dropdown(
                        label="Select Fine-tuned Model", 
                        choices=list_available_models()
                    )
                    gen_refresh_btn = gr.Button("Refresh Models")
                    gen_load_btn = gr.Button("Load Selected Model")
                    
                    key_points = gr.Textbox(
                        label="Key Points for Email", 
                        placeholder="- Meeting scheduled for Monday\n- Need to discuss Q2 results\n- Bring presentation materials",
                        lines=5
                    )
                    
                    with gr.Row():
                        max_length = gr.Slider(label="Max Length", minimum=50, maximum=500, value=200, step=10)
                        temperature = gr.Slider(label="Temperature", minimum=0.1, maximum=1.0, value=0.7, step=0.1)
                    
                    gen_btn = gr.Button("Generate Email")
                
                with gr.Column():
                    # Generation output
                    gen_status = gr.Textbox(label="Model Loading Status")
                    gen_output = gr.Textbox(label="Generated Email", lines=10)
    
    # Connect functions for fine-tuning tab
    ft_refresh_btn.click(lambda: gr.Dropdown(choices=list_available_datasets()), outputs=[ft_dataset_selector])
    ft_start_btn.click(
        run_finetune,
        inputs=[ft_dataset_selector, ft_model, ft_output, ft_epochs, ft_batch_size, ft_lr, ft_lora],
        outputs=[ft_status]
    )
    
    # Connect functions for generation tab
    gen_refresh_btn.click(lambda: gr.Dropdown(choices=list_available_models()), outputs=[gen_model_selector])
    gen_load_btn.click(load_model_for_generation, inputs=[gen_model_selector], outputs=[gen_status])
    gen_btn.click(
        generate_text,
        inputs=[key_points, gen_model_selector, max_length, temperature],
        outputs=[gen_output]
    )

# Launch the app
if __name__ == "__main__":
    print("Starting Email Style Trainer app...")
    app.launch()