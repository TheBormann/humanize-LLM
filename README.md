# LLM Evaluation Pipeline

A simple framework for developing and testing LLM pipelines. This project provides a modular structure for working with different LLM implementations, including:

- Mock models for quick testing
- Hugging Face API integration (placeholder)
- Local model fine-tuning (placeholder)

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/humanize-LLM.git
   cd humanize-LLM
   ```

2. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows, use: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Set up environment variables:
   ```bash
   cp .env.example .env
   ```
   Edit the `.env` file with your configuration settings.

## Project Structure

```
llm_eval/
├── __init__.py           # Package initialization
├── pipeline.py           # Core pipeline implementation
├── models/
    ├── __init__.py       # Models package
    ├── base.py           # Base model interface
    ├── mock.py           # Mock model for testing
    ├── huggingface.py    # Hugging Face API integration (placeholder)
    └── local.py          # Local model implementation (placeholder)
examples/
└── simple_test.py        # Example usage with mock model
```

## Quick Start

1. Clone the repository
2. Run the example script:

```bash
python examples/simple_test.py
```

This will run a simple test using the mock model implementation.

## Using Different Models

### Mock Model

The mock model is useful for testing the pipeline without requiring any external dependencies:

```python
from llm_eval.models.mock import MockModel
from llm_eval.pipeline import LLMPipeline

model = MockModel(prefix="Test response: ")
pipeline = LLMPipeline(model)
response = pipeline.run("What is the capital of France?")
print(response)
```

### Hugging Face Model (Placeholder)

The Hugging Face model implementation is currently a placeholder. To use it in the future:

```python
from llm_eval.models.huggingface import HuggingFaceModel
from llm_eval.pipeline import LLMPipeline

model = HuggingFaceModel(model_id="gpt2", api_key="your_api_key")
pipeline = LLMPipeline(model)
response = pipeline.run("What is the capital of France?")
print(response)
```

### Local Model (Placeholder)

The local model implementation is currently a placeholder. To use it in the future:

```python
from llm_eval.models.local import LocalModel
from llm_eval.pipeline import LLMPipeline

model = LocalModel(model_path="path/to/model")
pipeline = LLMPipeline(model)
response = pipeline.run("What is the capital of France?")
print(response)
```

## Future Development

### Implementing Hugging Face Integration

To implement the Hugging Face integration, you'll need to:

1. Install the required dependencies: `pip install transformers`
2. Update the `huggingface.py` file with actual API integration code

### Implementing Local Model Fine-tuning

To implement local model fine-tuning, you'll need to:

1. Choose a framework (PyTorch, TensorFlow, etc.)
2. Install the required dependencies
3. Update the `local.py` file with actual model loading and fine-tuning code

## License

MIT