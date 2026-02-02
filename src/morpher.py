"""
MODULE A: The Morpher - Adaptive Variant Generation
Generates semantically equivalent variants of logic questions using metamorphic transformations.
"""

import json
import yaml
import logging
from typing import List, Dict, Optional
from dataclasses import dataclass
import numpy as np

# Sentence-BERT for semantic similarity
from sentence_transformers import SentenceTransformer, util

# LLM API clients
try:
    import openai
except ImportError:
    openai = None

try:
    import anthropic
except ImportError:
    anthropic = None


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class Variant:
    """Represents a generated question variant"""
    original_id: str
    variant_text: str
    transformation_type: str
    similarity_score: float
    is_valid: bool


class MorpherEngine:
    """
    Generates semantically equivalent variants of logic questions.

    Transformations:
    1. Paraphrase - Rewrite with different words
    2. Negation - Apply double negation or contrapositive
    3. Variable Substitution - Replace entity names
    4. Premise Reordering - Change order of assumptions
    5. Redundant Context - Add irrelevant background
    """

    def __init__(self, config_path: str = "config/config.yaml",
                 prompts_path: str = "config/prompts.yaml"):
        """
        Initialize the Morpher with configuration and prompts.

        Args:
            config_path: Path to config.yaml
            prompts_path: Path to prompts.yaml
        """
        # Load configuration
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)['morpher']

        # Load prompts
        with open(prompts_path, 'r') as f:
            self.prompts = yaml.safe_load(f)['morpher']

        # Initialize Sentence-BERT for similarity checking
        logger.info("Loading Sentence-BERT model for similarity filtering...")
        self.similarity_model = SentenceTransformer('all-MiniLM-L6-v2')

        # Initialize LLM client based on config
        self.llm_model = self.config['model']
        self._init_llm_client()

        logger.info(f"Morpher initialized with model: {self.llm_model}")

    def _init_llm_client(self):
        """Initialize the appropriate LLM client"""
        if 'gpt' in self.llm_model.lower():
            if openai is None:
                raise ImportError("OpenAI library not installed. Run: pip install openai")
            # OpenAI client will be initialized per-request with API key from env
            self.llm_provider = 'openai'

        elif 'claude' in self.llm_model.lower():
            if anthropic is None:
                raise ImportError("Anthropic library not installed. Run: pip install anthropic")
            self.llm_provider = 'anthropic'

        else:
            # Default to Hugging Face Inference API
            self.llm_provider = 'huggingface'
            logger.warning("Using Hugging Face Inference API - may have rate limits")

    def generate_variants(self, question: str, question_id: str,
                         num_variants: Optional[int] = None) -> List[Variant]:
        """
        Generate multiple semantic variants of a question.

        Args:
            question: Original logic question
            question_id: Unique identifier for the question
            num_variants: Number of variants to generate (default from config)

        Returns:
            List of Variant objects (only valid ones with similarity > threshold)
        """
        if num_variants is None:
            num_variants = self.config['num_variants']

        logger.info(f"Generating {num_variants} variants for question: {question_id}")

        # Get active transformation types from config
        active_transforms = [
            t for t, enabled in self.config['transformations'].items()
            if enabled
        ]

        variants = []

        # Generate one variant per transformation type
        for transform_type in active_transforms[:num_variants]:
            try:
                variant_text = self._apply_transformation(question, transform_type)

                # Calculate semantic similarity
                similarity = self._calculate_similarity(question, variant_text)

                # Check if variant is valid
                is_valid = similarity >= self.config['similarity_threshold']

                variant = Variant(
                    original_id=question_id,
                    variant_text=variant_text,
                    transformation_type=transform_type,
                    similarity_score=similarity,
                    is_valid=is_valid
                )

                variants.append(variant)

                if is_valid:
                    logger.info(f"✓ Valid variant ({transform_type}): similarity={similarity:.3f}")
                else:
                    logger.warning(f"✗ Invalid variant ({transform_type}): similarity={similarity:.3f} < {self.config['similarity_threshold']}")

            except Exception as e:
                logger.error(f"Error generating {transform_type} variant: {e}")
                continue

        # Return only valid variants
        valid_variants = [v for v in variants if v.is_valid]
        logger.info(f"Generated {len(valid_variants)}/{len(variants)} valid variants")

        return valid_variants

    def _apply_transformation(self, question: str, transform_type: str) -> str:
        """
        Apply a specific transformation to generate a variant.

        Args:
            question: Original question
            transform_type: Type of transformation (paraphrase, negation, etc.)

        Returns:
            Generated variant text
        """
        # Get the prompt template for this transformation
        prompt_template = self.prompts['transformation_prompts'][transform_type]

        # Format the prompt with the question
        user_prompt = prompt_template.format(question=question)
        system_prompt = self.prompts['system_prompt']

        # Call LLM to generate variant
        response = self._call_llm(system_prompt, user_prompt)

        # Parse the response (expected to be JSON format)
        variant_text = self._parse_llm_response(response, transform_type)

        return variant_text

    def _call_llm(self, system_prompt: str, user_prompt: str) -> str:
        """
        Call the configured LLM with the given prompts.

        Args:
            system_prompt: System instruction
            user_prompt: User query

        Returns:
            LLM response text
        """
        # For now, implement a simple version
        # In production, this would call actual API

        if self.llm_provider == 'openai':
            return self._call_openai(system_prompt, user_prompt)
        elif self.llm_provider == 'anthropic':
            return self._call_anthropic(system_prompt, user_prompt)
        else:
            return self._call_huggingface(system_prompt, user_prompt)

    def _call_openai(self, system_prompt: str, user_prompt: str) -> str:
        """Call OpenAI API"""
        import os
        from openai import OpenAI

        client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

        response = client.chat.completions.create(
            model=self.llm_model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=self.config['temperature'],
            max_tokens=self.config['max_tokens']
        )

        return response.choices[0].message.content

    def _call_anthropic(self, system_prompt: str, user_prompt: str) -> str:
        """Call Anthropic API"""
        import os
        from anthropic import Anthropic

        client = Anthropic(api_key=os.getenv('ANTHROPIC_API_KEY'))

        response = client.messages.create(
            model=self.llm_model,
            system=system_prompt,
            messages=[{"role": "user", "content": user_prompt}],
            temperature=self.config['temperature'],
            max_tokens=self.config['max_tokens']
        )

        return response.content[0].text

    def _call_huggingface(self, system_prompt: str, user_prompt: str) -> str:
        """Call Hugging Face Inference API"""
        # For now, return a placeholder
        # In production, implement HF API call
        logger.warning("Hugging Face Inference not fully implemented - using placeholder")

        # Simple placeholder transformations for testing
        return f"Transformed: {user_prompt[:100]}..."

    def _parse_llm_response(self, response: str, transform_type: str) -> str:
        """
        Parse LLM response to extract the variant text.
        Expected format: {"variant": "...", "type": "..."}

        Args:
            response: Raw LLM response
            transform_type: Expected transformation type

        Returns:
            Extracted variant text
        """
        try:
            # Try to parse as JSON
            data = json.loads(response)
            return data['variant']
        except (json.JSONDecodeError, KeyError):
            # If not valid JSON, try to extract text
            logger.warning(f"LLM response not in expected JSON format, using raw text")

            # Remove common wrapper phrases
            cleaned = response.replace('{"variant": "', '').replace('"}', '')
            cleaned = cleaned.replace("variant:", "").strip()

            return cleaned

    def _calculate_similarity(self, text1: str, text2: str) -> float:
        """
        Calculate semantic similarity between two texts using Sentence-BERT.

        Args:
            text1: First text (original question)
            text2: Second text (variant)

        Returns:
            Cosine similarity score (0 to 1)
        """
        # Encode both texts
        embeddings = self.similarity_model.encode([text1, text2], convert_to_tensor=True)

        # Calculate cosine similarity
        similarity = util.cos_sim(embeddings[0], embeddings[1])

        return float(similarity.item())

    def batch_generate(self, questions: List[Dict]) -> Dict[str, List[Variant]]:
        """
        Generate variants for multiple questions.

        Args:
            questions: List of question dicts with 'id' and 'question' keys

        Returns:
            Dictionary mapping question IDs to their variants
        """
        results = {}

        for q in questions:
            question_id = q['id']
            question_text = q['question']

            variants = self.generate_variants(question_text, question_id)
            results[question_id] = variants

        return results


# Test/Demo function
if __name__ == "__main__":
    # Example usage
    print("="*60)
    print("MODULE A: The Morpher - Demo")
    print("="*60)

    # Initialize morpher
    morpher = MorpherEngine()

    # Test question
    test_question = "All humans are mortal. Socrates is a human. Is Socrates mortal?"

    # Generate variants
    print(f"\nOriginal Question: {test_question}\n")

    variants = morpher.generate_variants(test_question, "test_001", num_variants=3)

    print(f"\nGenerated {len(variants)} valid variants:\n")
    for i, v in enumerate(variants, 1):
        print(f"{i}. [{v.transformation_type}] (similarity: {v.similarity_score:.3f})")
        print(f"   {v.variant_text}\n")
