#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Gradio app for fine-tuning Hugging Face models and using them in pipelines.

This app provides a user interface for:
1. Selecting and configuring Hugging Face models
2. Uploading and preparing training data
3. Fine-tuning models with customizable parameters
4. Testing fine-tuned models with different pipeline implementations
"""

import os
import sys
import json
import pandas as pd
import gradio as gr
import logging
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Union, Any

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    stream=sys.stdout
)
logger = logging.getLogger(__name__)

# Add the parent directory to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import project components
from src.models.huggingface import HuggingFaceModel
from src.models.local import LocalModel
from src.models.evaluator import EvaluatorModel
from src.models.base import FinetuningArguments, PEFTArguments
from src.pipeline.pipeline import LLMPipeline
from src.pipeline.personalized_pipeline import PersonalizedPipeline
from src.pipeline.multi_stage_pipeline import MultiStagePipeline

# Configure paths
DATA_DIR = "data"
OUTPUT_DIR = "output"
DEFAULT_DATASET = os.path.join(DATA_DIR, "fine_tuning_dataset_full.csv")

# Default HuggingFace models
DEFAULT_HF_MODELS = [
    "mistralai/mistral-7b-instruct-v0.2",
    "deepseek-ai/deepseek-coder-7b-instruct",
    "meta-llama/llama-2-7b-chat",
    "google/gemma-7b-it"
]

# Global variables
hf_model = None
local_model = None
evaluator_model = None
pipeline = None
fine_tuning_job = None
training_data = None

# Helper functions
def load_api_key():
    """Load Hugging Face API key from environment or .env file."""
    api_key = os.environ.get("HF_API_KEY")
    if not api_key:
        try:
            from dotenv import load_dotenv
            load_dotenv()
            api_key = os.environ.get("HF_API_KEY")
        except ImportError:
            pass
    return api_key

def load_csv(file_path):
    """Load a CSV file with training data."""
    try:
        # Try different separators
        for sep in [",", ";", "\t"]:
            try:
                df = pd.read_csv(file_path, sep=sep)
                if len(df.columns) > 1:
                    return df, f"Loaded {len(df)} rows from {os.path.basename(file_path)}"
            except Exception:
                continue
        
        return None, "Failed to load CSV file. Please check the format."
    except Exception as e:
        return None, f"Error loading CSV: {str(e)}"

def prepare_training_data(df, prompt_col, response_col):
    """Prepare training data for fine-tuning."""
    if prompt_col not in df.columns or response_col not in df.columns:
        return None, f"Columns not found: {prompt_col}, {response_col}"
    
    training_samples = []
    for _, row in df.iterrows():
        if pd.isna(row[prompt_col]) or pd.isna(row[response_col]):
            continue
            
        sample = {
            "prompt": row[prompt_col],
            "response": row[response_col]
        }
        training_samples.append(sample)
    
    return training_samples, f"Prepared {len(training_samples)} training samples"

# UI Components
def initialize_model(model_id, api_key):
    """Initialize a Hugging Face model."""
    global hf_model
    
    try:
        if not api_key:
            return "No API key provided. Please enter your Hugging Face API key."
            
        # Set environment variable for other components
        os.environ["HF_API_KEY"] = api_key
        
        # Initialize the model
        hf_model = HuggingFaceModel(model_id=model_id, api_key=api_key)
        return f"Successfully initialized model: {model_id}"
    except Exception as e:
        return f"Error initializing model: {str(e)}"

def test_model(prompt, temperature, max_length):
    """Test the initialized Hugging Face model."""
    global hf_model
    
    if hf_model is None:
        return "Please initialize a model first."
    
    try:
        response = hf_model.generate(
            prompt, 
            temperature=float(temperature), 
            max_length=int(max_length)
        )
        return response
    except Exception as e:
        return f"Error generating response: {str(e)}"

def upload_dataset(file):
    """Handle dataset upload."""
    if file is None:
        return "No file uploaded", None, [], []
    
    try:
        df, message = load_csv(file.name)
        if df is not None:
            columns = df.columns.tolist()
            return message, df, columns, columns
        else:
            return message, None, [], []
    except Exception as e:
        return f"Error processing uploaded file: {str(e)}", None, [], []

def prepare_data(df, prompt_column, response_column):
    """Prepare training data from the uploaded dataset."""
    global training_data
    
    if df is None:
        return "No dataset loaded. Please upload a dataset first."
    
    try:
        training_data, message = prepare_training_data(df, prompt_column, response_column)
        if training_data:
            # Show a sample
            sample = training_data[0] if training_data else {}
            sample_str = json.dumps(sample, indent=2)
            return f"{message}\n\nSample:\n{sample_str}"
        else:
            return message
    except Exception as e:
        return f"Error preparing training data: {str(e)}"

def start_finetuning(model_id, output_name, epochs, batch_size, learning_rate, use_peft, lora_r, lora_alpha, lora_dropout):
    """Start a fine-tuning job on Hugging Face."""
    global hf_model, training_data, fine_tuning_job
    
    if hf_model is None:
        return "Please initialize a model first."
    
    if training_data is None or len(training_data) == 0:
        return "No training data prepared. Please upload and prepare a dataset first."
    
    try:
        # Create output directory
        output_dir = os.path.join(OUTPUT_DIR, output_name)
        os.makedirs(output_dir, exist_ok=True)
        
        # Prepare fine-tuning arguments
        training_args = FinetuningArguments(
            train_data=training_data,
            epochs=int(epochs),
            batch_size=int(batch_size),
            learning_rate=float(learning_rate)
        )
        
        # Prepare PEFT arguments if enabled
        peft_args = None
        if use_peft:
            peft_args = PEFTArguments(
                rank=int(lora_r),
                alpha=float(lora_alpha),
                dropout=float(lora_dropout)
            )
        
        # Start fine-tuning
        hub_model_id = f"{output_name}"
        fine_tuning_job = hf_model.finetune(
            training_args=training_args,
            output_dir=output_dir,
            hub_model_id=hub_model_id,
            use_peft=use_peft,
            peft_args=peft_args
        )
        
        # Save configuration
        config_path = os.path.join(output_dir, "fine_tuning_config.json")
        with open(config_path, 'w') as f:
            json.dump(fine_tuning_job, f, indent=2)
        
        return f"Fine-tuning job prepared:\n{json.dumps(fine_tuning_job, indent=2)}"
    except Exception as e:
        return f"Error starting fine-tuning job: {str(e)}"

def setup_pipeline(pipeline_type, primary_model_id, api_key, style_model_path=None, evaluator_model_id=None, 
use_lora=False, lora_path=None):
    """Set up a pipeline with the specified models."""
    global hf_model, local_model, evaluator_model, pipeline
    
    try:
        # Initialize primary model
        primary_hf_model = HuggingFaceModel(model_id=primary_model_id, api_key=api_key)
        
        # Initialize pipeline based on type
        if pipeline_type == "Basic Pipeline":
            pipeline = LLMPipeline(model=primary_hf_model)
            return f"Initialized basic pipeline with model: {primary_model_id}"
            
        elif pipeline_type == "Multi-Stage Pipeline":
            # Initialize style model if provided
            if not style_model_path:
                return "Style model path is required for Multi-Stage Pipeline"
                
            style_model = LocalModel()
            style_model.load_model(
                model_path=style_model_path,
                quantize=True,
                quantize_type="4bit",
                lora_path=lora_path if use_lora else None
            )
            
            # Initialize evaluator model
            if evaluator_model_id and evaluator_model_id != primary_model_id:
                eval_hf_model = HuggingFaceModel(model_id=evaluator_model_id, api_key=api_key)
                evaluator_model = EvaluatorModel(evaluation_model=eval_hf_model)
            else:
                evaluator_model = EvaluatorModel(evaluation_model=primary_hf_model)
            
            # Create pipeline
            pipeline = MultiStagePipeline(
                primary_model=primary_hf_model,
                secondary_model=style_model,
                evaluator_model=evaluator_model
            )
            return f"Initialized Multi-Stage Pipeline with models:\nPrimary: {primary_model_id}\nStyle: {style_model_path}\nEvaluator: {evaluator_model_id or primary_model_id}"
            
        elif pipeline_type == "Personalized Pipeline":
            # Initialize style model if provided
            if not style_model_path:
                return "Style model path is required for Personalized Pipeline"
                
            style_model = LocalModel()
            style_model.load_model(
                model_path=style_model_path,
                quantize=True,
                quantize_type="4bit",
                lora_path=lora_path if use_lora else None
            )
            
            # Initialize evaluator model
            if evaluator_model_id and evaluator_model_id != primary_model_id:
                eval_hf_model = HuggingFaceModel(model_id=evaluator_model_id, api_key=api_key)
                evaluator_model = EvaluatorModel(evaluation_model=eval_hf_model)
            else:
                evaluator_model = EvaluatorModel(evaluation_model=primary_hf_model)
            
            # Create pipeline
            pipeline = PersonalizedPipeline(
                initial_model=primary_hf_model,
                style_model=style_model,
                evaluator_model=evaluator_model
            )
            return f"Initialized Personalized Pipeline with models:\nPrimary: {primary_model_id}\nStyle: {style_model_path}\nEvaluator: {evaluator_model_id or primary_model_id}"
        else:
            return f"Unknown pipeline type: {pipeline_type}"
    except Exception as e:
        return f"Error setting up pipeline: {str(e)}"

def test_pipeline(prompt, temperature, max_length, auto_improve=False, num_variations=3):
    """Test the current pipeline with a prompt."""
    global pipeline
    
    if pipeline is None:
        return "Please set up a pipeline first."
    
    try:
        # Set parameters
        kwargs = {
            "temperature": float(temperature),
            "max_length": int(max_length)
        }
        
        # Run the pipeline
        if isinstance(pipeline, PersonalizedPipeline):
            result = pipeline.run(
                prompt=prompt, 
                auto_improve=auto_improve,
                num_variations=int(num_variations),
                **kwargs
            )
            # Format the result for display
            if isinstance(result, dict):
                return f"Best Response:\n\n{result['best_response']}\n\nScore: {result.get('score', 'N/A')}\n\nVariations: {len(result.get('variations', []))}"
            else:
                return str(result)
        else:
            # For basic and multi-stage pipelines
            response = pipeline.run(prompt=prompt, **kwargs)
            return response
    except Exception as e:
        return f"Error running pipeline: {str(e)}"

# Build the Gradio interface
def build_interface():
    with gr.Blocks(title="Hugging Face Fine-tuning & Pipeline App") as app:
        gr.Markdown("# Hugging Face Fine-tuning & Pipeline App")
        
        with gr.Tab("Model Setup"):
            with gr.Row():
                with gr.Column():
                    api_key_input = gr.Textbox(
                        label="Hugging Face API Key", 
                        placeholder="Enter your API key",
                        value=load_api_key() or "",
                        type="password"
                    )
                    model_dropdown = gr.Dropdown(
                        label="Select Model", 
                        choices=DEFAULT_HF_MODELS,
                        value=DEFAULT_HF_MODELS[0] if DEFAULT_HF_MODELS else ""
                    )
                    custom_model_input = gr.Textbox(
                        label="Or enter custom model ID", 
                        placeholder="e.g., mistralai/mistral-7b-instruct-v0.2"
                    )
                    init_model_btn = gr.Button("Initialize Model")
                    init_model_output = gr.Textbox(label="Initialization Status")
            
            with gr.Row():
                with gr.Column():
                    gr.Markdown("### Test Model")
                    test_prompt = gr.Textbox(
                        label="Test Prompt", 
                        placeholder="Enter a prompt to test the model",
                        lines=3
                    )
                    with gr.Row():
                        test_temp = gr.Slider(
                            label="Temperature", 
                            minimum=0.1, 
                            maximum=1.0, 
                            value=0.7, 
                            step=0.1
                        )
                        test_max_length = gr.Slider(
                            label="Max Length", 
                            minimum=50, 
                            maximum=1000, 
                            value=200, 
                            step=50
                        )
                    test_model_btn = gr.Button("Generate Response")
                    test_output = gr.Textbox(label="Model Output", lines=10)
        
        with gr.Tab("Fine-tuning"):
            with gr.Row():
                with gr.Column():
                    gr.Markdown("### Upload Training Data")
                    dataset_upload = gr.File(
                        label="Upload Dataset (CSV)",
                        file_types=["csv"]
                    )
                    upload_output = gr.Textbox(label="Upload Status")
                    
                    with gr.Row():
                        prompt_col = gr.Dropdown(label="Prompt Column")
                        response_col = gr.Dropdown(label="Response Column")
                    
                    prepare_data_btn = gr.Button("Prepare Training Data")
                    data_prep_output = gr.Textbox(label="Data Preparation Status", lines=5)
            
            with gr.Row():
                with gr.Column():
                    gr.Markdown("### Fine-tuning Configuration")
                    ft_model_id = gr.Textbox(
                        label="Base Model ID", 
                        value=DEFAULT_HF_MODELS[0] if DEFAULT_HF_MODELS else ""
                    )
                    output_name = gr.Textbox(
                        label="Output Model Name", 
                        placeholder="my-finetuned-model"
                    )
                    
                    with gr.Row():
                        epochs = gr.Number(label="Epochs", value=3, minimum=1, maximum=10)
                        batch_size = gr.Number(label="Batch Size", value=8, minimum=1, maximum=64)
                        learning_rate = gr.Number(label="Learning Rate", value=2e-5, minimum=1e-6, maximum=1e-3)
                    
                    use_peft = gr.Checkbox(label="Use PEFT/LoRA", value=True)
                    
                    with gr.Row(visible=True) as peft_options:
                        lora_r = gr.Number(label="LoRA Rank", value=16, minimum=1, maximum=64)
                        lora_alpha = gr.Number(label="LoRA Alpha", value=32, minimum=1, maximum=128)
                        lora_dropout = gr.Number(label="LoRA Dropout", value=0.05, minimum=0, maximum=0.5)
                    
                    start_ft_btn = gr.Button("Prepare Fine-tuning Job")
                    ft_output = gr.Textbox(label="Fine-tuning Status", lines=10)
        
        with gr.Tab("Pipeline Integration"):
            with gr.Row():
                with gr.Column():
                    gr.Markdown("### Pipeline Configuration")
                    pipeline_type = gr.Radio(
                        label="Pipeline Type",
                        choices=["Basic Pipeline", "Multi-Stage Pipeline", "Personalized Pipeline"],
                        value="Basic Pipeline"
                    )
                    
                    # Primary model configuration
                    gr.Markdown("#### Primary Model (Complex Tasks)")
                    primary_model_id = gr.Textbox(
                        label="Primary Model ID", 
                        value=DEFAULT_HF_MODELS[0] if DEFAULT_HF_MODELS else ""
                    )
                    
                    pipeline_api_key = gr.Textbox(
                        label="API Key", 
                        value=load_api_key() or "",
                        type="password"
                    )
                    
                    # Advanced options for multi-stage and personalized pipelines
                    with gr.Row(visible=False) as advanced_pipeline_options:
                        # Style model configuration
                        with gr.Column():
                            gr.Markdown("#### Style Model (Style Transfer)")
                            style_model_path = gr.Textbox(
                                label="Style Model Path", 
                                placeholder="Path to local style model"
                            )
                            use_lora_checkbox = gr.Checkbox(label="Use LoRA Weights", value=False)
                            lora_weights_path = gr.Textbox(
                                label="LoRA Weights Path", 
                                placeholder="Path to LoRA weights",
                                visible=False
                            )
                        
                        # Evaluator model configuration
                        with gr.Column():
                            gr.Markdown("#### Evaluator Model (Response Evaluation)")
                            use_separate_evaluator = gr.Checkbox(
                                label="Use Separate Evaluator Model", 
                                value=False
                            )
                            evaluator_model_id = gr.Textbox(
                                label="Evaluator Model ID", 
                                placeholder="Leave empty to use primary model",
                                visible=False
                            )
                    
                    setup_pipeline_btn = gr.Button("Setup Pipeline")
                    pipeline_setup_output = gr.Textbox(label="Pipeline Setup Status")
            
            with gr.Row():
                with gr.Column():
                    gr.Markdown("### Test Pipeline")
                    pipeline_prompt = gr.Textbox(
                        label="Test Prompt", 
                        placeholder="Enter a prompt to test the pipeline",
                        lines=3
                    )
                    
                    with gr.Row():
                        pipeline_temp = gr.Slider(
                            label="Temperature", 
                            minimum=0.1, 
                            maximum=1.0, 
                            value=0.7, 
                            step=0.1
                        )
                        pipeline_max_length = gr.Slider(
                            label="Max Length", 
                            minimum=50, 
                            maximum=1000, 
                            value=200, 
                            step=50
                        )
                    
                    with gr.Row(visible=False) as personalized_options:
                        auto_improve = gr.Checkbox(label="Auto-improve Response", value=False)
                        num_variations = gr.Slider(
                            label="Number of Variations", 
                            minimum=1, 
                            maximum=5, 
                            value=3, 
                            step=1
                        )
                    
                    test_pipeline_btn = gr.Button("Run Pipeline")
                    pipeline_output = gr.Textbox(label="Pipeline Output", lines=10)
        
        # Event handlers
        def get_model_id(dropdown, custom):
            return custom if custom else dropdown
        
        # Model setup events
        init_model_btn.click(
            fn=lambda dropdown, custom, key: initialize_model(get_model_id(dropdown, custom), key),
            inputs=[model_dropdown, custom_model_input, api_key_input],
            outputs=init_model_output
        )
        
        test_model_btn.click(
            fn=test_model,
            inputs=[test_prompt, test_temp, test_max_length],
            outputs=test_output
        )
        
        # Data upload events
        # Use a single upload handler with all outputs
        dataset_df = gr.State(None)
        dataset_upload.upload(
            fn=upload_dataset,
            inputs=[dataset_upload],
            outputs=[upload_output, dataset_df, prompt_col, response_col]
        )
        
        # Data preparation events
        prepare_data_btn.click(
            fn=prepare_data,
            inputs=[dataset_df, prompt_col, response_col],  # Pass the dataframe state
            outputs=data_prep_output
        )
        
        # Fine-tuning events
        start_ft_btn.click(
            fn=start_finetuning,
            inputs=[ft_model_id, output_name, epochs, batch_size, learning_rate, 
                   use_peft, lora_r, lora_alpha, lora_dropout],
            outputs=ft_output
        )
        
        # Pipeline events
        def update_pipeline_options(pipeline_type):
            if pipeline_type in ["Multi-Stage Pipeline", "Personalized Pipeline"]:
                return gr.update(visible=True), gr.update(visible=(pipeline_type == "Personalized Pipeline"))
            else:
                return gr.update(visible=False), gr.update(visible=False)
        
        def update_lora_path_visibility(use_lora):
            return gr.update(visible=use_lora)
            
        def update_evaluator_model_visibility(use_separate):
            return gr.update(visible=use_separate)
        
        pipeline_type.change(
            fn=update_pipeline_options,
            inputs=[pipeline_type],
            outputs=[advanced_pipeline_options, personalized_options]
        )
        
        use_lora_checkbox.change(
            fn=update_lora_path_visibility,
            inputs=[use_lora_checkbox],
            outputs=lora_weights_path
        )
        
        use_separate_evaluator.change(
            fn=update_evaluator_model_visibility,
            inputs=[use_separate_evaluator],
            outputs=evaluator_model_id
        )
        
        def get_evaluator_model_id(use_separate, evaluator_id):
            return evaluator_id if use_separate else None
        
        setup_pipeline_btn.click(
            fn=lambda p_type, p_id, api_key, style_path, use_sep, eval_id, use_lora, lora_path: 
                setup_pipeline(
                    p_type, p_id, api_key, style_path, 
                    get_evaluator_model_id(use_sep, eval_id),
                    use_lora, lora_path
                ),
            inputs=[pipeline_type, primary_model_id, pipeline_api_key, 
                   style_model_path, use_separate_evaluator, evaluator_model_id,
                   use_lora_checkbox, lora_weights_path],
            outputs=pipeline_setup_output
        )
        
        test_pipeline_btn.click(
            fn=test_pipeline,
            inputs=[pipeline_prompt, pipeline_temp, pipeline_max_length, 
                   auto_improve, num_variations],
            outputs=pipeline_output
        )
        
        # PEFT visibility toggle
        use_peft.change(
            fn=lambda x: gr.update(visible=x),
            inputs=[use_peft],
            outputs=peft_options
        )
        
    return app

# Main function
def main():
    # Create output directory if it doesn't exist
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Build and launch the interface
    app = build_interface()
    app.launch(share=False)

if __name__ == "__main__":
    main()