"""
MODULE D: Complexity Engine - Entropy and Compression Analysis
Measures information-theoretic properties of LLM responses.
"""

import yaml
import logging
import zlib
import gzip
import bz2
import numpy as np
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
import matplotlib.pyplot as plt


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class ComplexityAnalysis:
    """Results of complexity analysis"""
    question_id: str
    response_text: str
    variant_type: str

    # Entropy metrics
    token_entropy: Optional[float] = None  # Average token uncertainty
    sequence_entropy: Optional[float] = None  # Total sequence uncertainty

    # Compression metrics
    raw_length: int = 0
    compressed_length: int = 0
    compression_ratio: float = 0.0  # compressed / raw
    ncd_score: Optional[float] = None  # Normalized Compression Distance

    # Normalized scores (0-1)
    entropy_normalized: Optional[float] = None
    ncd_normalized: Optional[float] = None


class ComplexityEngine:
    """
    Analyzes information complexity of LLM responses.

    Metrics:
    1. Token Entropy: -1/T * Σ log(p(token))
    2. Compression Ratio: len(compressed) / len(raw)
    3. NCD: Normalized Compression Distance between responses
    """

    def __init__(self, config_path: str = "config/config.yaml"):
        """
        Initialize the Complexity Engine.

        Args:
            config_path: Path to config.yaml
        """
        # Load configuration
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)['complexity']

        # Compression algorithm
        self.compression_algo = self.config.get('compression_algorithm', 'zlib')

        logger.info(f"Complexity Engine initialized (algo: {self.compression_algo})")

    def calculate_token_entropy(self, logprobs: List[float]) -> float:
        """
        Calculate average token entropy from log-probabilities.

        Args:
            logprobs: List of log-probabilities for each token

        Returns:
            Average entropy per token (in bits if log base 2)
        """
        if not logprobs or len(logprobs) == 0:
            logger.warning("No logprobs provided for entropy calculation")
            return 0.0

        # Convert log-probs to actual probabilities
        # Note: logprobs are typically natural log (ln), convert to bits if needed
        # H = -1/T * Σ log_p(token) where log_p(token) is the log-probability

        # Average negative log-probability
        avg_negative_logprob = -np.mean(logprobs)

        # This is equivalent to entropy in nats (natural logarithm)
        # To convert to bits: divide by ln(2)
        entropy_nats = avg_negative_logprob
        entropy_bits = entropy_nats / np.log(2)

        return float(entropy_bits)

    def calculate_sequence_entropy(self, logprobs: List[float]) -> float:
        """
        Calculate total sequence entropy.

        Args:
            logprobs: List of log-probabilities

        Returns:
            Total entropy of the sequence
        """
        if not logprobs:
            return 0.0

        # Sum of negative log-probs
        total_entropy = -np.sum(logprobs)

        return float(total_entropy)

    def compress_text(self, text: str) -> bytes:
        """
        Compress text using configured algorithm.

        Args:
            text: Text to compress

        Returns:
            Compressed bytes
        """
        text_bytes = text.encode('utf-8')

        if self.compression_algo == 'zlib':
            return zlib.compress(text_bytes)
        elif self.compression_algo == 'gzip':
            return gzip.compress(text_bytes)
        elif self.compression_algo == 'bz2':
            return bz2.compress(text_bytes)
        else:
            raise ValueError(f"Unknown compression algorithm: {self.compression_algo}")

    def calculate_compression_ratio(self, text: str) -> Tuple[int, int, float]:
        """
        Calculate compression ratio.

        Args:
            text: Text to compress

        Returns:
            Tuple of (raw_length, compressed_length, ratio)
        """
        raw_bytes = text.encode('utf-8')
        raw_length = len(raw_bytes)

        compressed_bytes = self.compress_text(text)
        compressed_length = len(compressed_bytes)

        ratio = compressed_length / raw_length if raw_length > 0 else 0.0

        return raw_length, compressed_length, ratio

    def calculate_ncd(self, text1: str, text2: str) -> float:
        """
        Calculate Normalized Compression Distance between two texts.

        Formula: NCD(x,y) = [C(xy) - min(C(x), C(y))] / max(C(x), C(y))

        Args:
            text1: First text
            text2: Second text

        Returns:
            NCD score (0 to 1+, lower = more similar)
        """
        # Compress individual texts
        c_x = len(self.compress_text(text1))
        c_y = len(self.compress_text(text2))

        # Compress concatenation
        c_xy = len(self.compress_text(text1 + text2))

        # Calculate NCD
        numerator = c_xy - min(c_x, c_y)
        denominator = max(c_x, c_y)

        ncd = numerator / denominator if denominator > 0 else 0.0

        return float(ncd)

    def analyze_response(self, response_text: str, question_id: str,
                        variant_type: str = "original",
                        logprobs: Optional[List[float]] = None) -> ComplexityAnalysis:
        """
        Perform full complexity analysis on a single response.

        Args:
            response_text: The response text
            question_id: Question identifier
            variant_type: Type of variant
            logprobs: Optional log-probabilities for entropy calculation

        Returns:
            ComplexityAnalysis object
        """
        logger.info(f"Analyzing complexity for {question_id} ({variant_type})")

        # Entropy analysis
        token_entropy = None
        sequence_entropy = None

        if logprobs:
            token_entropy = self.calculate_token_entropy(logprobs)
            sequence_entropy = self.calculate_sequence_entropy(logprobs)
            logger.info(f"  Token entropy: {token_entropy:.4f} bits")

        # Compression analysis
        raw_length, compressed_length, compression_ratio = self.calculate_compression_ratio(response_text)
        logger.info(f"  Compression: {raw_length}B → {compressed_length}B (ratio: {compression_ratio:.4f})")

        return ComplexityAnalysis(
            question_id=question_id,
            response_text=response_text,
            variant_type=variant_type,
            token_entropy=token_entropy,
            sequence_entropy=sequence_entropy,
            raw_length=raw_length,
            compressed_length=compressed_length,
            compression_ratio=compression_ratio
        )

    def batch_analyze(self, responses: List[Dict]) -> List[ComplexityAnalysis]:
        """
        Analyze complexity for multiple responses.

        Args:
            responses: List of response dicts with keys:
                      'response_text', 'question_id', 'variant_type', 'logprobs' (optional)

        Returns:
            List of ComplexityAnalysis objects
        """
        results = []

        for response in responses:
            analysis = self.analyze_response(
                response_text=response['response_text'],
                question_id=response['question_id'],
                variant_type=response.get('variant_type', 'original'),
                logprobs=response.get('logprobs')
            )
            results.append(analysis)

        return results

    def calculate_ncd_matrix(self, responses: List[str]) -> np.ndarray:
        """
        Calculate NCD for all pairs of responses.

        Args:
            responses: List of response texts

        Returns:
            NxN matrix of NCD scores
        """
        n = len(responses)
        matrix = np.zeros((n, n))

        for i in range(n):
            for j in range(i + 1, n):
                ncd = self.calculate_ncd(responses[i], responses[j])
                matrix[i, j] = ncd
                matrix[j, i] = ncd

        return matrix

    def normalize_scores(self, analyses: List[ComplexityAnalysis]) -> List[ComplexityAnalysis]:
        """
        Normalize entropy and NCD scores to 0-1 range.

        Args:
            analyses: List of ComplexityAnalysis objects

        Returns:
            Updated analyses with normalized scores
        """
        # Extract values
        entropies = [a.token_entropy for a in analyses if a.token_entropy is not None]
        compression_ratios = [a.compression_ratio for a in analyses]

        if not entropies:
            logger.warning("No entropy values to normalize")
            return analyses

        # Normalize using min-max scaling
        entropy_min, entropy_max = min(entropies), max(entropies)
        comp_min, comp_max = min(compression_ratios), max(compression_ratios)

        for analysis in analyses:
            # Normalize entropy
            if analysis.token_entropy is not None:
                if entropy_max > entropy_min:
                    analysis.entropy_normalized = (analysis.token_entropy - entropy_min) / (entropy_max - entropy_min)
                else:
                    analysis.entropy_normalized = 0.5  # All equal

            # Normalize compression (using ratio)
            if comp_max > comp_min:
                analysis.ncd_normalized = (analysis.compression_ratio - comp_min) / (comp_max - comp_min)
            else:
                analysis.ncd_normalized = 0.5

        return analyses

    def visualize_metrics(self, analyses: List[ComplexityAnalysis],
                         output_path: str, title: str = "Complexity Metrics"):
        """
        Visualize entropy and compression metrics.

        Args:
            analyses: List of ComplexityAnalysis objects
            output_path: Path to save plot
            title: Plot title
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

        labels = [a.variant_type for a in analyses]
        x_pos = np.arange(len(labels))

        # Plot 1: Entropy
        entropies = [a.token_entropy if a.token_entropy else 0 for a in analyses]
        ax1.bar(x_pos, entropies, color='steelblue', alpha=0.7)
        ax1.set_xlabel('Response Variant', fontsize=12)
        ax1.set_ylabel('Token Entropy (bits)', fontsize=12)
        ax1.set_title('Token Entropy by Variant', fontsize=14, fontweight='bold')
        ax1.set_xticks(x_pos)
        ax1.set_xticklabels(labels, rotation=45, ha='right')
        ax1.grid(axis='y', alpha=0.3)

        # Plot 2: Compression Ratio
        ratios = [a.compression_ratio for a in analyses]
        ax2.bar(x_pos, ratios, color='coral', alpha=0.7)
        ax2.set_xlabel('Response Variant', fontsize=12)
        ax2.set_ylabel('Compression Ratio', fontsize=12)
        ax2.set_title('Compression Ratio by Variant', fontsize=14, fontweight='bold')
        ax2.set_xticks(x_pos)
        ax2.set_xticklabels(labels, rotation=45, ha='right')
        ax2.grid(axis='y', alpha=0.3)

        plt.suptitle(title, fontsize=16, fontweight='bold', y=1.02)
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()

        logger.info(f"✓ Metrics visualization saved to {output_path}")


# Test/Demo function
if __name__ == "__main__":
    print("="*70)
    print("MODULE D: Complexity Engine - Demo")
    print("="*70)

    # Initialize engine
    engine = ComplexityEngine()

    # Test responses with mock logprobs
    print("\n[TEST 1] Single response analysis:")
    response1 = "Yes, Socrates is mortal because all humans are mortal and Socrates is a human."

    # Mock logprobs (simulating model confidence)
    logprobs1 = [-0.5, -1.2, -0.8, -0.3, -1.5, -0.9, -1.1, -0.7, -1.0, -0.6]

    analysis1 = engine.analyze_response(
        response1,
        "test_001",
        "original",
        logprobs1
    )

    print(f"  Response: {response1[:60]}...")
    print(f"  Token Entropy: {analysis1.token_entropy:.4f} bits")
    print(f"  Compression: {analysis1.raw_length}B → {analysis1.compressed_length}B")
    print(f"  Compression Ratio: {analysis1.compression_ratio:.4f}")

    # Test multiple responses
    print("\n[TEST 2] Batch analysis:")
    responses = [
        {
            'response_text': "Yes, Socrates is mortal.",
            'question_id': 'q1',
            'variant_type': 'original',
            'logprobs': [-0.5, -1.0, -0.8, -0.3]
        },
        {
            'response_text': "Affirmative. Given that all humans are mortal and Socrates is human, he must be mortal.",
            'question_id': 'q1',
            'variant_type': 'paraphrase',
            'logprobs': [-0.7, -1.5, -0.9, -1.2, -0.6, -1.1, -0.8, -1.0, -0.5, -1.3]
        },
        {
            'response_text': "No, penguins cannot fly.",
            'question_id': 'q2',
            'variant_type': 'original',
            'logprobs': [-0.3, -0.8, -0.5, -0.4]
        }
    ]

    analyses = engine.batch_analyze(responses)

    for a in analyses:
        print(f"\n  [{a.variant_type}] Entropy: {a.token_entropy:.4f}, Compression: {a.compression_ratio:.4f}")

    # Normalize scores
    print("\n[TEST 3] Normalization:")
    analyses = engine.normalize_scores(analyses)

    for a in analyses:
        print(f"  [{a.variant_type}] Normalized - Entropy: {a.entropy_normalized:.4f}, NCD: {a.ncd_normalized:.4f}")

    # NCD matrix
    print("\n[TEST 4] NCD Matrix:")
    texts = [a.response_text for a in analyses]
    ncd_matrix = engine.calculate_ncd_matrix(texts)
    print(f"  Matrix shape: {ncd_matrix.shape}")
    print(f"  NCD between response 0 and 1: {ncd_matrix[0, 1]:.4f}")
    print(f"  NCD between response 0 and 2: {ncd_matrix[0, 2]:.4f}")

    # Visualize
    print("\n[TEST 5] Visualization:")
    engine.visualize_metrics(
        analyses,
        "data/processed/results/complexity_metrics.png",
        "Complexity Analysis Demo"
    )

    print("\n" + "="*70)
    print("✓ Demo complete!")
    print("="*70)
