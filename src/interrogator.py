"""
MODULE B: The Interrogator - Response Collection
Queries target LLMs with question variants and collects responses with log-probabilities.
"""

import os
import json
import yaml
import logging
import time
from typing import List, Dict, Optional, Any
from dataclasses import dataclass, asdict
import numpy as np

# API clients
try:
    import openai
    from openai import OpenAI
except ImportError:
    openai = None

try:
    import anthropic
    from anthropic import Anthropic
except ImportError:
    anthropic = None


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class LLMResponse:
    """Represents a response from an LLM"""
    question_id: str
    variant_type: str  # 'original' or transformation type
    question_text: str
    response_text: str
    model_name: str
    logprobs: Optional[List[float]] = None  # Token log-probabilities
    tokens: Optional[List[str]] = None  # Actual tokens
    timestamp: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None

    def to_dict(self):
        """Convert to dictionary for JSON serialization"""
        return asdict(self)


class InterrogatorEngine:
    """
    Queries target LLMs with questions and collects responses.

    Supports:
    - OpenAI (GPT-3.5, GPT-4)
    - Anthropic (Claude)
    - Hugging Face models
    """

    def __init__(self, config_path: str = "config/config.yaml",
                 prompts_path: str = "config/prompts.yaml"):
        """
        Initialize the Interrogator.

        Args:
            config_path: Path to config.yaml
            prompts_path: Path to prompts.yaml
        """
        # Load configuration
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)['interrogator']

        # Load prompts
        with open(prompts_path, 'r') as f:
            self.prompts = yaml.safe_load(f)['interrogator']

        # Initialize API clients
        self._init_clients()

        logger.info(f"Interrogator initialized with models: {self.config['target_models']}")

    def _init_clients(self):
        """Initialize API clients for different LLM providers"""
        self.clients = {}

        # OpenAI
        if any('gpt' in model.lower() for model in self.config['target_models']):
            api_key = os.getenv('OPENAI_API_KEY')
            if api_key:
                self.clients['openai'] = OpenAI(api_key=api_key)
                logger.info("✓ OpenAI client initialized")
            else:
                logger.warning("⚠ OPENAI_API_KEY not found in environment")

        # Anthropic
        if any('claude' in model.lower() for model in self.config['target_models']):
            api_key = os.getenv('ANTHROPIC_API_KEY')
            if api_key:
                self.clients['anthropic'] = Anthropic(api_key=api_key)
                logger.info("✓ Anthropic client initialized")
            else:
                logger.warning("⚠ ANTHROPIC_API_KEY not found in environment")

    def query_single(self, question: str, question_id: str,
                    variant_type: str = "original",
                    model_name: Optional[str] = None) -> LLMResponse:
        """
        Query a single question to an LLM.

        Args:
            question: The question text
            question_id: Unique identifier
            variant_type: Type of variant ('original', 'paraphrase', etc.)
            model_name: Specific model to use (default: first in config)

        Returns:
            LLMResponse object
        """
        if model_name is None:
            model_name = self.config['target_models'][0]

        logger.info(f"Querying {model_name} with {variant_type} variant...")

        # Route to appropriate provider
        if 'gpt' in model_name.lower():
            return self._query_openai(question, question_id, variant_type, model_name)
        elif 'claude' in model_name.lower():
            return self._query_anthropic(question, question_id, variant_type, model_name)
        else:
            return self._query_huggingface(question, question_id, variant_type, model_name)

    def _query_openai(self, question: str, question_id: str,
                     variant_type: str, model_name: str) -> LLMResponse:
        """Query OpenAI models (GPT-3.5, GPT-4)"""
        if 'openai' not in self.clients:
            raise RuntimeError("OpenAI client not initialized. Set OPENAI_API_KEY.")

        client = self.clients['openai']

        # Prepare messages
        system_prompt = self.prompts['system_prompt']
        user_prompt = self.prompts['user_template'].format(question=question)

        try:
            # Call API with logprobs enabled
            response = client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=self.config['temperature'],
                max_tokens=self.config['max_tokens'],
                logprobs=self.config['return_logprobs'],
                top_logprobs=5 if self.config['return_logprobs'] else None
            )

            # Extract response
            response_text = response.choices[0].message.content

            # Extract logprobs if available
            logprobs_data = None
            tokens_data = None

            if self.config['return_logprobs'] and response.choices[0].logprobs:
                logprobs_content = response.choices[0].logprobs.content
                if logprobs_content:
                    logprobs_data = [token.logprob for token in logprobs_content]
                    tokens_data = [token.token for token in logprobs_content]

            return LLMResponse(
                question_id=question_id,
                variant_type=variant_type,
                question_text=question,
                response_text=response_text,
                model_name=model_name,
                logprobs=logprobs_data,
                tokens=tokens_data,
                timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
                metadata={
                    "finish_reason": response.choices[0].finish_reason,
                    "usage": {
                        "prompt_tokens": response.usage.prompt_tokens,
                        "completion_tokens": response.usage.completion_tokens,
                        "total_tokens": response.usage.total_tokens
                    }
                }
            )

        except Exception as e:
            logger.error(f"OpenAI API error: {e}")
            raise

    def _query_anthropic(self, question: str, question_id: str,
                        variant_type: str, model_name: str) -> LLMResponse:
        """Query Anthropic models (Claude)"""
        if 'anthropic' not in self.clients:
            raise RuntimeError("Anthropic client not initialized. Set ANTHROPIC_API_KEY.")

        client = self.clients['anthropic']

        # Prepare prompts
        system_prompt = self.prompts['system_prompt']
        user_prompt = self.prompts['user_template'].format(question=question)

        try:
            response = client.messages.create(
                model=model_name,
                system=system_prompt,
                messages=[{"role": "user", "content": user_prompt}],
                temperature=self.config['temperature'],
                max_tokens=self.config['max_tokens']
            )

            response_text = response.content[0].text

            # Note: Claude API doesn't provide token logprobs directly
            # We'll need to estimate entropy differently for Claude

            return LLMResponse(
                question_id=question_id,
                variant_type=variant_type,
                question_text=question,
                response_text=response_text,
                model_name=model_name,
                logprobs=None,  # Not available for Claude
                tokens=None,
                timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
                metadata={
                    "stop_reason": response.stop_reason,
                    "usage": {
                        "input_tokens": response.usage.input_tokens,
                        "output_tokens": response.usage.output_tokens
                    }
                }
            )

        except Exception as e:
            logger.error(f"Anthropic API error: {e}")
            raise

    def _query_huggingface(self, question: str, question_id: str,
                          variant_type: str, model_name: str) -> LLMResponse:
        """Query Hugging Face models"""
        # Placeholder for HF implementation
        logger.warning("Hugging Face querying not fully implemented")

        return LLMResponse(
            question_id=question_id,
            variant_type=variant_type,
            question_text=question,
            response_text="[HF Response placeholder]",
            model_name=model_name,
            timestamp=time.strftime("%Y-%m-%d %H:%M:%S")
        )

    def query_variants(self, original_question: str, question_id: str,
                      variants: List[Dict], model_name: Optional[str] = None) -> List[LLMResponse]:
        """
        Query both original question and all variants.

        Args:
            original_question: The original question text
            question_id: Unique identifier
            variants: List of variant dicts from Morpher (with 'variant_text' and 'transformation_type')
            model_name: Model to use (default: first in config)

        Returns:
            List of LLMResponse objects
        """
        responses = []

        # Query original question first
        logger.info(f"Querying original question: {question_id}")
        original_response = self.query_single(
            original_question,
            question_id,
            variant_type="original",
            model_name=model_name
        )
        responses.append(original_response)

        # Query each variant
        for i, variant in enumerate(variants, 1):
            logger.info(f"Querying variant {i}/{len(variants)}: {variant.get('transformation_type', 'unknown')}")

            variant_response = self.query_single(
                variant['variant_text'],
                question_id,
                variant_type=variant.get('transformation_type', f'variant_{i}'),
                model_name=model_name
            )
            responses.append(variant_response)

            # Rate limiting - small delay between requests
            time.sleep(0.5)

        logger.info(f"Collected {len(responses)} responses for {question_id}")
        return responses

    def batch_query(self, questions_with_variants: Dict[str, Dict],
                   model_name: Optional[str] = None) -> Dict[str, List[LLMResponse]]:
        """
        Query multiple questions with their variants.

        Args:
            questions_with_variants: Dict mapping question_id to:
                {
                    'original': str,
                    'variants': List[Dict]
                }
            model_name: Model to use

        Returns:
            Dict mapping question_id to list of responses
        """
        all_responses = {}

        for question_id, data in questions_with_variants.items():
            logger.info(f"\nProcessing question: {question_id}")

            original = data['original']
            variants = data['variants']

            responses = self.query_variants(original, question_id, variants, model_name)
            all_responses[question_id] = responses

            # Longer delay between questions to respect rate limits
            time.sleep(1.0)

        return all_responses

    def save_responses(self, responses: List[LLMResponse], output_path: str):
        """
        Save responses to JSON file.

        Args:
            responses: List of LLMResponse objects
            output_path: Path to save JSON file
        """
        # Convert to serializable format
        data = [r.to_dict() for r in responses]

        with open(output_path, 'w') as f:
            json.dump(data, f, indent=2)

        logger.info(f"Saved {len(responses)} responses to {output_path}")


# Test/Demo function
if __name__ == "__main__":
    print("="*70)
    print("MODULE B: The Interrogator - Demo")
    print("="*70)

    # This is just a structure demo - actual API calls require keys
    interrogator = InterrogatorEngine()

    print("\n✓ Interrogator initialized")
    print(f"  Target models: {interrogator.config['target_models']}")
    print(f"  Temperature: {interrogator.config['temperature']}")
    print(f"  Max tokens: {interrogator.config['max_tokens']}")
    print(f"  Return logprobs: {interrogator.config['return_logprobs']}")

    print("\n📝 Demo complete. To test with real API:")
    print("  1. Set API keys in .env file")
    print("  2. Run: python -m src.interrogator")
