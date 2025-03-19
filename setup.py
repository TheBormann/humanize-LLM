from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="llm_eval",
    version="0.1.0",
    author="User",
    author_email="user@example.com",
    description="A simple framework for developing and testing LLM pipelines",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/username/llm_eval",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.7",
    install_requires=[
        # No external dependencies required for basic functionality
        # Add dependencies as needed for specific model implementations
        # e.g., "transformers" for Hugging Face integration
    ],
)