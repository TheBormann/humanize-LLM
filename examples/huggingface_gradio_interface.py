"""Gradio interface for testing the LLM pipeline with a Hugging Face local model.

This example provides a web interface for interactively testing and fine-tuning
models from Hugging Face Hub with the LLM evaluation pipeline.
"""
import sys
import os
import json
import gradio as gr
import torch

# Add the parent directory to the path so we can import the package
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from llm_eval.models.huggingface_local import HuggingFaceModel
from llm_eval.pipeline import LLMPipeline

# Global variables to store model and pipeline
model = None
pipeline = None
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

def initialize_model(model_id, device, memory_size, batch_size, learning_rate):
    """Initialize the model with the given parameters."""
    global model, pipeline, model_info
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
        
        # Create the pipeline
        pipeline = LLMPipeline(model)
        
        # Get model info
        model_info = model.get_model_info()
        
        return f"Model initialized: {model_id} on {device}", json.dumps(model_info, indent=2)
    except Exception as e:
        return f"Error initializing model: {str(e)}", ""

def generate_response(prompt, max_length, temperature, top_p):
    """Generate a response for a single prompt."""
    global pipeline, results_history
    
    if not pipeline:
        return "Please initialize a model first."
    
    try:
        # Convert parameters to appropriate types
        max_length = int(max_length)
        temperature = float(temperature)
        top_p = float(top_p)
        
        # Generate response
        response = pipeline.run(
            prompt, 
            max_length=max_length,
            temperature=temperature,
            top_p=top_p
        )
        
        # Add to history
        results_history.append({"prompt": prompt, "response": response})
        
        return response
    except Exception as e:
        return f"Error generating response: {str(e)}"

def batch_generate(prompts, max_length, temperature, top_p):
    """Generate responses for multiple prompts."""
    global pipeline, results_history
    
    if not pipeline:
        return "Please initialize a model first."
    
    try:
        # Split prompts by newline and filter out empty lines
        prompt_list = [p.strip() for p in prompts.split('\n') if p.strip()]
        
        if not prompt_list:
            return "No prompts provided."
        
        # Convert parameters to appropriate types
        max_length = int(max_length)
        temperature = float(temperature)
        top_p = float(top_p)
        
        # Generate responses
        responses = pipeline.run_batch(
            prompt_list,
            max_length=max_length,
            temperature=temperature,
            top_p=top_p
        )
        
        # Format results
        result = ""
        for prompt, response in zip(prompt_list, responses):
            result += f"Prompt: {prompt}\nResponse: {response}\n\n"
            results_history.append({"prompt": prompt, "response": response})
        
        return result
    except Exception as e:
        return f"Error generating batch responses: {str(e)}"

def fine_tune_model(train_data, epochs, batch_size, learning_rate):
    """Fine-tune the model with the provided training data."""
    global model
    
    if not model:
        return "Please initialize a model first."
    
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
        results = model.fine_tune(
            train_examples,
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=learning_rate
        )
        
        return f"Fine-tuning completed: {json.dumps(results, indent=2)}"
    except Exception as e:
        return f"Error during fine-tuning: {str(e)}"

def evaluate_responses(evaluator_type):
    """Evaluate the responses using the selected evaluator."""
    global pipeline, results_history
    
    if not pipeline:
        return "Please initialize a model first."
    
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
        
        # Evaluate the responses
        evaluation = pipeline.evaluate(prompts, evaluator)
        
        return f"Evaluation results:\n{json.dumps(evaluation, indent=2)}"
    except Exception as e:
        return f"Error during evaluation: {str(e)}"

def save_results(filename):
    """Save the results to a file."""
    global pipeline
    
    if not pipeline:
        return "Please initialize a model first."
    
    try:
        if not filename.endswith('.json'):
            filename += '.json'
        
        pipeline.save_results(filename)
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
        gr.Markdown("# LLM Evaluation Pipeline with Hugging Face Models")
        
        with gr.Tab("Model Setup"):
            with gr.Row():
                with gr.Column():
                    model_id = gr.Dropdown(
                        choices=DEFAULT_MODELS,
                        value="gpt2",
                        label="Model ID",
                        allow_custom_value=True,
                        info="Select a model or enter a custom Hugging Face model ID"
                    )
                    device = gr.Radio(
                        choices=["cpu", "cuda"], 
                        value="cpu", 
                        label="Device",
                        info="Select the device to run the model on"
                    )
                    memory_size = gr.Slider(
                        minimum=100, 
                        maximum=5000, 
                        value=1000, 
                        step=100, 
                        label="Memory Size",
                        info="Size of memory buffer for online learning"
                    )
                    batch_size = gr.Slider(
                        minimum=1, 
                        maximum=16, 
                        value=1, 
                        step=1, 
                        label="Online Batch Size",
                        info="Batch size for online updates"
                    )
                    learning_rate = gr.Slider(
                        minimum=1e-6, 
                        maximum=1e-3, 
                        value=1e-5, 
                        step=1e-6, 
                        label="Online Learning Rate",
                        info="Learning rate for online updates"
                    )
                    init_button = gr.Button("Initialize Model")
                
                with gr.Column():
                    init_output = gr.Textbox(label="Initialization Status")
                    model_info_output = gr.Textbox(label="Model Info", lines=10)
        
        with gr.Tab("Single Prompt"):
            with gr.Row():
                with gr.Column():
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
                        ft_learning_rate = gr.Slider(
                            minimum=1e-6, 
                            maximum=1e-3, 
                            value=2e-5, 
                            step=1e-6, 
                            label="Learning Rate"
                        )
                    fine_tune_button = gr.Button("Fine-Tune Model")
                
                with gr.Column():
                    fine_tune_output = gr.Textbox(label="Fine-Tuning Results", lines=10)
        
        with gr.Tab("Evaluation & Results"):
            with gr.Row():
                with gr.Column():
                    evaluator_type = gr.Radio(
                        choices=["length", "keyword"], 
                        value="length", 
                        label="Evaluator Type"
                    )
                    evaluate_button = gr.Button("Evaluate Responses")
                    clear_button = gr.Button("Clear History")
                    
                    filename = gr.Textbox(
                        label="Results Filename",
                        value="model_results.json",
                        placeholder="Enter filename for results"
                    )
                    save_button = gr.Button("Save Results")
                
                with gr.Column():
                    evaluation_output = gr.Textbox(label="Evaluation Results", lines=10)
                    save_output = gr.Textbox(label="Save Status")
        
        # Connect components with functions
        init_button.click(
            initialize_model,
            inputs=[model_id, device, memory_size, batch_size, learning_rate],
            outputs=[init_output, model_info_output]
        )
        
        generate_button.click(
            generate_response,
            inputs=[single_prompt, single_max_length, single_temperature, single_top_p],
            outputs=single_output
        )
        
        batch_button.click(
            batch_generate,
            inputs=[batch_prompts, batch_max_length, batch_temperature, batch_top_p],
            outputs=batch_output
        )
        
        fine_tune_button.click(
            fine_tune_model,
            inputs=[train_data, ft_epochs, ft_batch_size, ft_learning_rate],
            outputs=fine_tune_output
        )
        
        evaluate_button.click(
            evaluate_responses,
            inputs=evaluator_type,
            outputs=evaluation_output
        )
        
        clear_button.click(
            clear_history,
            inputs=None,
            outputs=evaluation_output
        )
        
        save_button.click(
            save_results,
            inputs=filename,
            outputs=save_output
        )
        
        return interface


def main():
    # Create and launch the interface
    interface = create_interface()
    interface.launch(share=True)
    

if __name__ == "__main__":
    main()