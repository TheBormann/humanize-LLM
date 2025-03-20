"""Comprehensive UI for testing all LLM evaluation pipeline features.

This interface provides a unified web UI for testing and evaluating different LLM models,
including Hugging Face models, local models, and multi-stage pipelines.
"""
import sys
import os
import json
import gradio as gr
import torch

# Add the parent directory to the path so we can import the package
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from llm_eval.models.huggingface_local import HuggingFaceModel
from llm_eval.models.local import LocalModel
from llm_eval.models.mock import MockModel
from llm_eval.models.evaluator import EvaluatorModel
from llm_eval.pipeline import LLMPipeline
from llm_eval.multi_stage_pipeline import MultiStagePipeline

# Global variables to store models and pipelines
models = {
    "primary": None,
    "secondary": None,
    "evaluator": None
}
pipeline = None
multi_stage_pipeline = None
model_info = {}
results_history = []

# List of available HuggingFace models (small models suitable for CPU)
DEFAULT_MODELS = [
    "gpt2",
    "distilgpt2",
    "facebook/opt-125m",
    "EleutherAI/pythia-70m",
    "bigscience/bloom-560m"
]

def initialize_huggingface_model(model_id, device, memory_size, batch_size, learning_rate, model_role):
    """Initialize a Hugging Face model with the given parameters."""
    global models, model_info
    try:
        # Convert parameters to appropriate types
        memory_size = int(memory_size)
        batch_size = int(batch_size)
        learning_rate = float(learning_rate)
        
        # Create the model
        model = HuggingFaceModel(
            model_id=model_id,
            device=device,
            memory_size=memory_size,
            online_batch_size=batch_size,
            online_learning_rate=learning_rate
        )
        
        # Store the model in the appropriate role
        models[model_role] = model
        
        # Get model info
        model_info[model_role] = model.get_model_info()
        
        return f"Model initialized: {model_id} on {device} as {model_role}", json.dumps(model_info[model_role], indent=2)
    except Exception as e:
        return f"Error initializing model: {str(e)}", ""

def initialize_local_model(model_path, device, memory_size, batch_size, learning_rate, model_role):
    """Initialize a local model with the given parameters."""
    global models, model_info
    try:
        # Convert parameters to appropriate types
        memory_size = int(memory_size)
        batch_size = int(batch_size)
        learning_rate = float(learning_rate)
        
        # Create the model
        model = LocalModel(
            model_path=model_path,
            device=device,
            memory_size=memory_size,
            online_batch_size=batch_size,
            online_learning_rate=learning_rate
        )
        
        # Store the model in the appropriate role
        models[model_role] = model
        
        # Get model info
        model_info[model_role] = model.get_model_info()
        
        return f"Local model initialized: {model_path} on {device} as {model_role}", json.dumps(model_info[model_role], indent=2)
    except Exception as e:
        return f"Error initializing local model: {str(e)}", ""

def initialize_mock_model(prefix, model_role):
    """Initialize a mock model with the given parameters."""
    global models, model_info
    try:
        # Create the model
        model = MockModel(prefix=prefix)
        
        # Store the model in the appropriate role
        models[model_role] = model
        
        # Get model info
        model_info[model_role] = model.get_model_info()
        
        return f"Mock model initialized with prefix '{prefix}' as {model_role}", json.dumps(model_info[model_role], indent=2)
    except Exception as e:
        return f"Error initializing mock model: {str(e)}", ""

def initialize_evaluator_model(model_id, model_role):
    """Initialize an evaluator model."""
    global models, model_info
    try:
        # Create the model
        model = EvaluatorModel(model_id=model_id)
        
        # Store the model in the appropriate role
        models[model_role] = model
        
        # Get model info
        model_info[model_role] = model.get_model_info()
        
        return f"Evaluator model initialized: {model_id} as {model_role}", json.dumps(model_info[model_role], indent=2)
    except Exception as e:
        return f"Error initializing evaluator model: {str(e)}", ""

def setup_pipeline(pipeline_type, num_variations):
    """Set up the pipeline based on the selected type."""
    global pipeline, multi_stage_pipeline, models
    
    try:
        if pipeline_type == "single":
            if not models["primary"]:
                return "Please initialize a primary model first."
            
            pipeline = LLMPipeline(models["primary"])
            return "Single-stage pipeline initialized with primary model."
        
        elif pipeline_type == "multi":
            if not models["primary"] or not models["secondary"] or not models["evaluator"]:
                return "Please initialize all required models (primary, secondary, and evaluator) first."
            
            multi_stage_pipeline = MultiStagePipeline(
                primary_model=models["primary"],
                secondary_model=models["secondary"],
                evaluator_model=models["evaluator"],
                num_variations=int(num_variations)
            )
            return f"Multi-stage pipeline initialized with {num_variations} variations."
        
        else:
            return "Invalid pipeline type selected."
    
    except Exception as e:
        return f"Error setting up pipeline: {str(e)}"

def generate_response(prompt, max_length, temperature, top_p, pipeline_type):
    """Generate a response for a single prompt using the selected pipeline."""
    global pipeline, multi_stage_pipeline, results_history
    
    try:
        # Convert parameters to appropriate types
        max_length = int(max_length)
        temperature = float(temperature)
        top_p = float(top_p)
        
        if pipeline_type == "single":
            if not pipeline:
                return "Please initialize a single-stage pipeline first."
            
            # Generate response
            response = pipeline.run(
                prompt, 
                max_length=max_length,
                temperature=temperature,
                top_p=top_p
            )
            
        elif pipeline_type == "multi":
            if not multi_stage_pipeline:
                return "Please initialize a multi-stage pipeline first."
            
            # Generate response using multi-stage pipeline
            response = multi_stage_pipeline.run(
                prompt,
                max_length=max_length,
                temperature=temperature,
                top_p=top_p
            )
        else:
            return "Invalid pipeline type selected."
        
        # Add to history
        results_history.append({
            "prompt": prompt, 
            "response": response, 
            "pipeline_type": pipeline_type
        })
        
        return response
    except Exception as e:
        return f"Error generating response: {str(e)}"

def batch_generate(prompts, max_length, temperature, top_p, pipeline_type):
    """Generate responses for multiple prompts using the selected pipeline."""
    global pipeline, multi_stage_pipeline, results_history
    
    try:
        # Split prompts by newline and filter out empty lines
        prompt_list = [p.strip() for p in prompts.split('\n') if p.strip()]
        
        if not prompt_list:
            return "No prompts provided."
        
        # Convert parameters to appropriate types
        max_length = int(max_length)
        temperature = float(temperature)
        top_p = float(top_p)
        
        if pipeline_type == "single":
            if not pipeline:
                return "Please initialize a single-stage pipeline first."
            
            # Generate responses
            responses = pipeline.run_batch(
                prompt_list,
                max_length=max_length,
                temperature=temperature,
                top_p=top_p
            )
            
        elif pipeline_type == "multi":
            if not multi_stage_pipeline:
                return "Please initialize a multi-stage pipeline first."
            
            # Generate responses using multi-stage pipeline
            responses = multi_stage_pipeline.run_batch(
                prompt_list,
                max_length=max_length,
                temperature=temperature,
                top_p=top_p
            )
        else:
            return "Invalid pipeline type selected."
        
        # Format results
        result = ""
        for prompt, response in zip(prompt_list, responses):
            result += f"Prompt: {prompt}\nResponse: {response}\n\n"
            results_history.append({
                "prompt": prompt, 
                "response": response, 
                "pipeline_type": pipeline_type
            })
        
        return result
    except Exception as e:
        return f"Error generating batch responses: {str(e)}"

def fine_tune_model(train_data, epochs, batch_size, learning_rate, model_role):
    """Fine-tune the selected model with the provided training data."""
    global models
    
    if not models[model_role]:
        return f"Please initialize a {model_role} model first."
    
    try:
        # Parse the training data
        # Format should be: prompt1 ||| response1\nprompt2 ||| response2\n...
        data_pairs = [line.strip() for line in train_data.split('\n') if line.strip()]
        train_examples = []
        
        for pair in data_pairs:
            if '|||' in pair:
                prompt, response = pair.split('|||', 1)
                train_examples.append({
                    "prompt": prompt.strip(),
                    "response": response.strip()
                })
        
        if not train_examples:
            return "No valid training examples provided. Use format: prompt ||| response"
        
        # Convert parameters to appropriate types
        epochs = int(epochs)
        batch_size = int(batch_size)
        learning_rate = float(learning_rate)
        
        # Fine-tune the model
        results = models[model_role].fine_tune(
            train_examples,
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=learning_rate
        )
        
        return f"Fine-tuning completed for {model_role} model: {json.dumps(results, indent=2)}"
    except Exception as e:
        return f"Error during fine-tuning: {str(e)}"

def evaluate_responses(evaluator_type):
    """Evaluate the responses using the selected evaluator."""
    global pipeline, multi_stage_pipeline, results_history
    
    if not pipeline and not multi_stage_pipeline:
        return "Please initialize a pipeline first."
    
    if not results_history:
        return "No responses to evaluate. Generate some responses first."
    
    try:
        # Extract prompts from history
        prompts = [item["prompt"] for item in results_history]
        
        # Define evaluator function based on type
        if evaluator_type == "length":
            def evaluator(prompt, response):
                return min(len(response) / 100, 1.0)
        elif evaluator_type == "keyword":
            def evaluator(prompt, response):
                # Simple keyword matching evaluator
                keywords = prompt.lower().split()
                matches = sum(1 for word in keywords if word in response.lower())
                return min(matches / len(keywords) if keywords else 0, 1.0)
        else:
            return "Invalid evaluator type."
        
        # Evaluate the responses using the appropriate pipeline
        if pipeline:
            evaluation = pipeline.evaluate(prompts, evaluator)
        else:
            # For multi-stage pipeline, we'll use the primary model's pipeline for evaluation
            temp_pipeline = LLMPipeline(models["primary"])
            evaluation = temp_pipeline.evaluate(prompts, evaluator)
        
        return f"Evaluation results:\n{json.dumps(evaluation, indent=2)}"
    except Exception as e:
        return f"Error during evaluation: {str(e)}"

def save_results(filename, pipeline_type):
    """Save the results to a file."""
    global pipeline, multi_stage_pipeline
    
    try:
        if not filename.endswith('.json'):
            filename += '.json'
        
        if pipeline_type == "single":
            if not pipeline:
                return "Please initialize a single-stage pipeline first."
            pipeline.save_results(filename)
        elif pipeline_type == "multi":
            if not multi_stage_pipeline:
                return "Please initialize a multi-stage pipeline first."
            multi_stage_pipeline.save_results(filename)
        else:
            return "Invalid pipeline type selected."
        
        return f"Results saved to {filename}"
    except Exception as e:
        return f"Error saving results: {str(e)}"

def clear_history():
    """Clear the results history."""
    global results_history
    results_history = []
    return "History cleared."

# Create the Gradio interface
def create_interface():
    with gr.Blocks(title="LLM Evaluation Pipeline") as interface:
        gr.Markdown("# LLM Evaluation Pipeline")
        gr.Markdown("A comprehensive interface for testing and evaluating different LLM models and pipelines.")
        
        with gr.Tab("Model Setup"):
            gr.Markdown("### Initialize Models")
            gr.Markdown("Set up the models you want to use in your pipeline. For a single-stage pipeline, you only need a primary model. For a multi-stage pipeline, you need primary, secondary, and evaluator models.")
            
            # Primary Model Setup
            with gr.Accordion("Primary Model", open=True):
                with gr.Row():
                    with gr.Column():
                        primary_model_type = gr.Radio(
                            choices=["huggingface", "local", "mock"], 
                            value="huggingface", 
                            label="Model Type",
                            info="Select the type of model to initialize"
                        )
                        
                        # HuggingFace model parameters
                        with gr.Group(visible=True) as primary_hf_params:
                            primary_hf_model_id = gr.Dropdown(
                                choices=DEFAULT_MODELS,
                                value="gpt2",
                                label="Model ID",
                                allow_custom_value=True,
                                info="Select a model or enter a custom Hugging Face model ID"
                            )
                            primary_hf_device = gr.Radio(
                                choices=["cpu", "cuda"], 
                                value="cpu", 
                                label="Device",
                                info="Select the device to run the model on"
                            )
                            primary_hf_memory_size = gr.Slider(
                                minimum=100, 
                                maximum=5000, 
                                value=1000, 
                                step=100, 
                                label="Memory Size",
                                info="Size of memory buffer for online learning"
                            )
                            primary_hf_batch_size = gr.Slider(
                                minimum=1, 
                                maximum=16, 
                                value=1, 
                                step=1, 
                                label="Online Batch Size",
                                info="Batch size for online updates"
                            )
                            primary_hf_learning_rate = gr.Slider(
                                minimum=1e-6, 
                                maximum=1e-3, 
                                value=1e-5, 
                                step=1e-6, 
                                label="Online Learning Rate",
                                info="Learning rate for online updates"
                            )
                        
                        # Local model parameters
                        with gr.Group(visible=False) as primary_local_params:
                            primary_local_model_path = gr.Textbox(
                                value="examples/models/my_local_model",
                                label="Model Path",
                                info="Path to the local model files"
                            )
                            primary_local_device = gr.Radio(
                                choices=["cpu", "cuda"], 
                                value="cpu", 
                                label="Device",
                                info="Select the device to run the model on"
                            )
                            primary_local_memory_size = gr.Slider(
                                minimum=100, 
                                maximum=5000, 
                                value=1000, 
                                step=100, 
                                label="Memory Size",
                                info="Size of memory buffer for online learning"
                            )
                            primary_local_batch_size = gr.Slider(
                                minimum=1, 
                                maximum=16, 
                                value=1, 
                                step=1, 
                                label="Online Batch Size",
                                info="Batch size for online updates"
                            )
                            primary_local_learning_rate = gr.Slider(
                                minimum=1e-6, 
                                maximum=1e-3, 
                                value=1e-5, 
                                step=1e-6, 
                                label="Online Learning Rate",
                                info="Learning rate for online updates"
                            )
                        
                        # Mock model parameters
                        with gr.Group(visible=False) as primary_mock_params:
                            primary_mock_prefix = gr.Textbox(
                                value="Primary model response: ",
                                label="Response Prefix",
                                info="Prefix to add to mock responses"
                            )
                        
                        primary_init_button = gr.Button("Initialize Primary Model")
                    
                    with gr.Column():
                        primary_init_output = gr.Textbox(label="Initialization Status")
                        primary_model_info_output = gr.Textbox(label="Model Info", lines=10)
            
            # Secondary Model Setup
            with gr.Accordion("Secondary Model", open=False):
                with gr.Row():
                    with gr.Column():
                        secondary_model_type = gr.Radio(
                            choices=["huggingface", "local", "mock"], 
                            value="local", 
                            label="Model Type",
                            info="Select the type of model to initialize"
                        )
                        
                        # HuggingFace model parameters
                        with gr.Group(visible=False) as secondary_hf_params:
                            secondary_hf_model_id = gr.Dropdown(
                                choices=DEFAULT_MODELS,
                                value="gpt2",
                                label="Model ID",
                                allow_custom_value=True,
                                info="Select a model or enter a custom Hugging Face model ID"
                            )
                            secondary_hf_device = gr.Radio(
                                choices=["cpu", "cuda"], 
                                value="cpu", 
                                label="Device",
                                info="Select the device to run the model on"
                            )
                            secondary_hf_memory_size = gr.Slider(
                                minimum=100, 
                                maximum=5000, 
                                value=1000, 
                                step=100, 
                                label="Memory Size",
                                info="Size of memory buffer for online learning"
                            )
                            secondary_hf_batch_size = gr.Slider(
                                minimum=1, 
                                maximum=16, 
                                value=1, 
                                step=1, 
                                label="Online Batch Size",
                                info="Batch size for online updates"
                            )
                            secondary_hf_learning_rate = gr.Slider(
                                minimum=1e-6, 
                                maximum=1e-3, 
                                value=1e-5, 
                                step=1e-6, 
                                label="Online Learning Rate",
                                info="Learning rate for online updates"
                            )
                        
                        # Local model parameters
                        with gr.Group(visible=True) as secondary_local_params:
                            secondary_local_model_path = gr.Textbox(
                                value="examples/models/my_local_model",
                                label="Model Path",
                                info="Path to the local model files"
                            )
                            secondary_local_device = gr.Radio(
                                choices=["cpu", "cuda"], 
                                value="cpu", 
                                label="Device",
                                info="Select the device to run the model on"
                            )
                            secondary_local_memory_size = gr.Slider(
                                minimum=100, 
                                maximum=5000, 
                                value=1000, 
                                step=100, 
                                label="Memory Size",
                                info="Size of memory buffer for online learning"
                            )
                            secondary_local_batch_size = gr.Slider(
                                minimum=1, 
                                maximum=16, 
                                value=1, 
                                step=1, 
                                label="Online Batch Size",
                                info="Batch size for online updates"
                            )
                            secondary_local_learning_rate = gr.Slider(
                                minimum=1e-6, 
                                maximum=1e-3, 
                                value=1e-5, 
                                step=1e-6, 
                                label="Online Learning Rate",
                                info="Learning rate for online updates"
                            )
                        
                        # Mock model parameters
                        with gr.Group(visible=False) as secondary_mock_params:
                            secondary_mock_prefix = gr.Textbox(
                                value="Secondary model response: ",
                                label="Response Prefix",
                                info="Prefix to add to mock responses"
                            )
                        
                        secondary_init_button = gr.Button("Initialize Secondary Model")
                    
                    with gr.Column():
                        secondary_init_output = gr.Textbox(label="Initialization Status")
                        secondary_model_info_output = gr.Textbox(label="Model Info", lines=10)
            
            # Evaluator Model Setup
            with gr.Accordion("Evaluator Model", open=False):
                with gr.Row():
                    with gr.Column():
                        evaluator_model_id = gr.Textbox(
                            value="response-evaluator",
                            label="Evaluator Model ID",
                            info="Identifier for the evaluator model"
                        )
                        evaluator_init_button = gr.Button("Initialize Evaluator Model")
                    
                    with gr.Column():
                        evaluator_init_output = gr.Textbox(label="Initialization Status")
                        evaluator_model_info_output = gr.Textbox(label="Model Info", lines=10)
            
            # Pipeline Setup
            with gr.Accordion("Pipeline Setup", open=True):
                with gr.Row():
                    with gr.Column():
                        pipeline_type = gr.Radio(
                            choices=["single", "multi"], 
                            value="single", 
                            label="Pipeline Type",
                            info="Select the type of pipeline to initialize"
                        )
                        num_variations = gr.Slider(
                            minimum=1, 
                            maximum=10, 
                            value=3, 
                            step=1, 
                            label="Number of Variations",
                            info="Number of variations to generate with the secondary model (for multi-stage pipeline)",
                            visible=False
                        )
                        setup_pipeline_button = gr.Button("Setup Pipeline")
                    
                    with gr.Column():
                        pipeline_setup_output = gr.Textbox(label="Pipeline Setup Status")
        
        with gr.Tab("Single Prompt"):
            with gr.Row():
                with gr.Column():
                    single_pipeline_type = gr.Radio(
                        choices=["single", "multi"], 
                        value="single", 
                        label="Pipeline Type",
                        info="Select which pipeline to use"
                    )
                    single_prompt = gr.Textbox(
                        lines=3, 
                        label="Prompt",
                        placeholder="Enter your prompt here..."
                    )
                    with gr.Row():
                        single_max_length = gr.Slider(
                            minimum=10, 
                            maximum=500, 
                            value=100, 
                            step=10, 
                            label="Max Length"
                        )
                        single_temperature = gr.Slider(
                            minimum=0.1, 
                            maximum=1.5, 
                            value=0.7, 
                            step=0.1, 
                            label="Temperature"
                        )
                        single_top_p = gr.Slider(
                            minimum=0.1, 
                            maximum=1.0, 
                            value=0.9, 
                            step=0.1, 
                            label="Top P"
                        )
                    generate_button = gr.Button("Generate Response")
                
                with gr.Column():
                    single_output = gr.Textbox(label="Response", lines=8)
        
        with gr.Tab("Batch Processing"):
            with gr.Row():
                with gr.Column():
                    batch_pipeline_type = gr.Radio(
                        choices=["single", "multi"], 
                        value="single", 
                        label="Pipeline Type",
                        info="Select which pipeline to use"
                    )
                    batch_prompts = gr.Textbox(
                        lines=5, 
                        label="Prompts (one per line)",
                        placeholder="Enter one prompt per line..."
                    )
                    with gr.Row():
                        batch_max_length = gr.Slider(
                            minimum=10, 
                            maximum=500, 
                            value=100, 
                            step=10, 
                            label="Max Length"
                        )
                        batch_temperature = gr.Slider(
                            minimum=0.1, 
                            maximum=1.5, 
                            value=0.7, 
                            step=0.1, 
                            label="Temperature"
                        )
                        batch_top_p = gr.Slider(
                            minimum=0.1, 
                            maximum=1.0, 
                            value=0.9, 
                            step=0.1, 
                            label="Top P"
                        )
                    batch_button = gr.Button("Generate Batch Responses")
                
                with gr.Column():
                    batch_output = gr.Textbox(label="Batch Responses", lines=10)
        
        with gr.Tab("Fine-Tuning"):
            with gr.Row():
                with gr.Column():
                    fine_tune_model_role = gr.Radio(
                        choices=["primary", "secondary"], 
                        value="primary", 
                        label="Model to Fine-Tune",
                        info="Select which model to fine-tune"
                    )
                    train_data = gr.Textbox(
                        lines=5, 
                        label="Training Data",
                        placeholder="Format: prompt ||| response (one pair per line)"
                    )
                    with gr.Row():
                        ft_epochs = gr.Slider(
                            minimum=1, 
                            maximum=10, 
                            value=3, 
                            step=1, 
                            label="Epochs"
                        )
                        ft_batch_size = gr.Slider(
                            minimum=1, 
                            maximum=16, 
                            value=4,
                            step=1,
                            label="Batch Size"
                        )