"""Evaluator model implementation for selecting the best response."""
from typing import Dict, List, Optional, Union, Any, Callable
import logging
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import torch

from .local import LocalModel

logger = logging.getLogger(__name__)


class EvaluatorModel(LocalModel):
    """A model implementation for evaluating and selecting the best response.
    
    This model is designed to evaluate multiple responses and select the best one
    based on predefined criteria or a scoring mechanism. It can operate in two modes:
    1. Lightweight mode: Uses statistical methods for quick evaluation
    2. Model-based mode: Uses a smaller LLM for more sophisticated evaluation
    """
    
    def __init__(self, model_id: str = "evaluator", 
                 evaluation_model: Optional[LocalModel] = None,
                 evaluation_criteria: Optional[List[str]] = None,
                 mode: str = "model-based"):  # Changed default to model-based
        """Initialize the evaluator model.
        
        Args:
            model_id: Identifier for the evaluator model
            evaluation_model: Optional LLM model to use for evaluation
            evaluation_criteria: List of criteria to evaluate responses on
            mode: Evaluation mode - 'lightweight' or 'model-based'
        """
        self.model_id = model_id
        self.evaluation_model = evaluation_model
        self.mode = mode
        
        # Default evaluation criteria if none provided
        self.evaluation_criteria = evaluation_criteria or [
            "relevance",      # How relevant the response is to the prompt
            "accuracy",       # Factual correctness
            "completeness",   # How thoroughly it addresses the prompt
            "coherence",      # Logical flow and readability
            "helpfulness"     # Overall utility to the user
        ]
        
        # Initialize TF-IDF vectorizer for lightweight mode
        self.vectorizer = TfidfVectorizer(stop_words='english')
        
        logger.info(f"Initialized {self.mode} evaluator model with ID: {model_id}")
    
    def generate(self, prompt: str, **kwargs) -> str:
        """Generate a response using the evaluator model.
        
        This method is not typically used directly with the evaluator model.
        
        Args:
            prompt: The input text
            **kwargs: Additional parameters to pass to the model
            
        Returns:
            The generated text response
        """
        if self.evaluation_model:
            return self.evaluation_model.generate(prompt, **kwargs)
        return f"[Evaluator] This model is designed for evaluation, not generation. Prompt: {prompt}"
    
    def batch_generate(self, prompts: List[str], **kwargs) -> List[str]:
        """Generate responses for multiple prompts.
        
        This method is not typically used directly with the evaluator model.
        
        Args:
            prompts: List of input prompts
            **kwargs: Additional parameters to pass to the model
            
        Returns:
            List of generated text responses
        """
        return [self.generate(prompt, **kwargs) for prompt in prompts]
    
    def evaluate_responses(self, original_prompt: str, responses: List[str]) -> Dict[str, Any]:
        """Evaluate multiple responses and select the best one.
        
        Args:
            original_prompt: The original input prompt
            responses: List of responses to evaluate
            
        Returns:
            Dictionary with evaluation results including the best response
        """
        if not responses:
            logger.warning("No responses to evaluate")
            return {"best_response": "", "best_index": -1, "scores": []}
        
        # Prefer model-based if available, even if in lightweight mode
        if self.evaluation_model:
            return self._model_based_evaluation(original_prompt, responses)
        else:
            logger.warning("No evaluation model available, using lightweight fallback")
            return self._lightweight_evaluation(original_prompt, responses)
    
    def _lightweight_evaluation(self, original_prompt: str, responses: List[str]) -> Dict[str, Any]:
        """Perform lightweight evaluation using statistical methods.
        
        Args:
            original_prompt: The original input prompt
            responses: List of responses to evaluate
            
        Returns:
            Dictionary with evaluation results
        """
        scores = []
        
        # 1. Relevance score using TF-IDF and cosine similarity
        try:
            # Combine prompt and responses for vectorization
            all_texts = [original_prompt] + responses
            tfidf_matrix = self.vectorizer.fit_transform(all_texts)
            
            # Calculate cosine similarity between prompt and each response
            prompt_vector = tfidf_matrix[0:1]
            response_vectors = tfidf_matrix[1:]
            relevance_scores = cosine_similarity(prompt_vector, response_vectors)[0]
            
            # 2. Length-based completeness score (normalized)
            length_scores = [min(len(response) / 500, 1.0) for response in responses]
            
            # 3. Coherence approximation (sentence count vs. length ratio)
            coherence_scores = []
            for response in responses:
                sentences = response.split('.')
                if len(sentences) > 1:
                    avg_sentence_length = len(response) / len(sentences)
                    # Penalize very short or very long sentences
                    coherence_score = min(avg_sentence_length / 20, 1.0) if avg_sentence_length < 40 else 2 - (avg_sentence_length / 40)
                    coherence_scores.append(max(0, min(coherence_score, 1.0)))
                else:
                    coherence_scores.append(0.5)  # Default for single-sentence responses
            
            # Combine scores with weights
            for i in range(len(responses)):
                combined_score = (
                    0.5 * relevance_scores[i] +  # Relevance is most important
                    0.3 * length_scores[i] +     # Completeness
                    0.2 * coherence_scores[i]    # Coherence
                )
                scores.append(combined_score)
                
        except Exception as e:
            logger.error(f"Error in lightweight evaluation: {str(e)}")
            # Fallback to simple length-based scoring
            scores = [len(response) / max(len(max(responses, key=len)), 1) for response in responses]
        
        # Find the best response
        if scores:
            best_index = int(np.argmax(scores))
            best_response = responses[best_index]
        else:
            best_index = 0
            best_response = responses[0] if responses else ""
        
        return {
            "best_response": best_response,
            "best_index": best_index,
            "scores": scores,
            "evaluation_mode": "lightweight"
        }
    
    def _model_based_evaluation(self, original_prompt: str, responses: List[str]) -> Dict[str, Any]:
        """Perform evaluation using a language model.
        
        Args:
            original_prompt: The original input prompt
            responses: List of responses to evaluate
            
        Returns:
            Dictionary with evaluation results
        """
        if not self.evaluation_model:
            logger.warning("No evaluation model provided, falling back to lightweight evaluation")
            return self._lightweight_evaluation(original_prompt, responses)
        
        try:
            scores = []
            detailed_scores = []
            
            for i, response in enumerate(responses):
                # Create an enhanced evaluation prompt
                eval_prompt = f"""Evaluate the following response to the prompt: "{original_prompt}"

Response to evaluate: "{response}"

**Evaluation Task:**
1. Rate this response (0-10) for each criterion: {', '.join(self.evaluation_criteria)}
2. Provide specific improvements for each criterion (even if scoring high)
3. Summarize key strengths and weaknesses

**Required JSON format:**
{{
    "ratings": {{"{self.evaluation_criteria[0]}": score, ...}},
    "improvements": {{"{self.evaluation_criteria[0]}": [suggestion1, ...], ...}},
    "summary": "concise analysis of strengths/weaknesses"
}}"""

                # Get evaluation from the model
                eval_result = self.evaluation_model.generate(eval_prompt)
                
                # Try to parse enhanced result
                try:
                    import json
                    import re
                    
                    # Try to extract JSON from the response if it contains other text
                    json_match = re.search(r'\{[^\{\}]*\}', eval_result)
                    if json_match:
                        criteria_scores = json.loads(json_match.group(0))
                    else:
                        criteria_scores = json.loads(eval_result)
                    
                    # Calculate weighted average of criteria scores
                    weighted_score = sum(criteria_scores.values()) / len(criteria_scores)
                    scores.append(weighted_score / 10.0)  # Normalize to 0-1 scale
                    detailed_scores.append(criteria_scores)
                    
                except (json.JSONDecodeError, AttributeError) as json_err:
                    logger.warning(f"Failed to parse model evaluation result: {str(json_err)}")
                    # Fallback: extract numbers from the response and average them
                    number_pattern = r'\b(?:10|[0-9])(?:\.\d+)?\b'
                    numbers = re.findall(number_pattern, eval_result)
                    if numbers:
                        avg_score = sum(float(n) for n in numbers) / len(numbers)
                        scores.append(min(avg_score / 10.0, 1.0))  # Normalize to 0-1 scale
                    else:
                        # Last resort: use lightweight evaluation for this response
                        lightweight_score = self._lightweight_evaluation(original_prompt, [response])
                        scores.append(lightweight_score["scores"][0])
            
            # Find the best response
            if scores:
                best_index = int(np.argmax(scores))
                best_response = responses[best_index]
            else:
                best_index = 0
                best_response = responses[0] if responses else ""
            
            return {
                "best_response": best_response,
                "best_index": best_index,
                "scores": scores,
                "detailed_scores": detailed_scores if detailed_scores else None,
                "evaluation_mode": "model-based"
            }
            
        except Exception as e:
            logger.error(f"Error in model-based evaluation: {str(e)}")
            # Fallback to lightweight evaluation
            return self._lightweight_evaluation(original_prompt, responses)
    
    def get_model_info(self) -> Dict[str, str]:
        """Get information about the evaluator model.
        
        Returns:
            Dictionary with model information
        """
        info = super().get_model_info()
        info.update({
            "type": "evaluator",
            "model_id": self.model_id,
            "mode": self.mode,
            "description": "Model for evaluating and selecting the best response",
            "criteria": ", ".join(self.evaluation_criteria)
        })
        return info