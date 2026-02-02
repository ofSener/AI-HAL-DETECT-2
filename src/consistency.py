"""
Module C: Consistency Engine
Uses DeBERTa-v3 NLI model (cross-encoder) to detect contradictions between LLM responses.
Implements graph-based consistency analysis as specified in plan.md

HYBRID FORMAT (Ocak 2026):
- NLI karşılaştırması sadece ANSWER kısmı için yapılır
- Bu sayede explanation style variance FP'ye yol açmaz
"""

import os
import re
import json
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch


def extract_answer(response_text: str) -> str:
    """
    Extract the ANSWER portion from a structured response.

    Expected format:
        ANSWER: [answer text]
        REASONING: [explanation text]

    If format not found, returns the full response (fallback).

    Args:
        response_text: Full LLM response

    Returns:
        Extracted answer portion only
    """
    if not response_text:
        return ""

    # Try to extract ANSWER portion
    # Pattern 1: ANSWER: ... REASONING: ...
    pattern1 = r'ANSWER:\s*(.+?)(?:REASONING:|REASON:|EXPLANATION:|$)'
    match = re.search(pattern1, response_text, re.IGNORECASE | re.DOTALL)

    if match:
        answer = match.group(1).strip()
        # Clean up any trailing whitespace or newlines
        answer = ' '.join(answer.split())
        return answer

    # Pattern 2: Just ANSWER: ... (no reasoning section)
    pattern2 = r'ANSWER:\s*(.+?)$'
    match = re.search(pattern2, response_text, re.IGNORECASE | re.DOTALL)

    if match:
        answer = match.group(1).strip()
        answer = ' '.join(answer.split())
        return answer

    # Fallback: Return first sentence or first 200 chars
    # This handles free-form responses
    first_sentence = response_text.split('.')[0].strip()
    if len(first_sentence) > 10:
        return first_sentence + '.'

    return response_text[:200].strip()

@dataclass
class ConsistencyAnalysis:
    """Results from consistency analysis"""
    question_id: str
    consistency_score: float
    contradiction_matrix: List[List[float]]
    graph_metrics: Dict[str, float]
    visualization_path: Optional[str] = None

class ConsistencyEngine:
    """
    Analyzes consistency between LLM responses using DeBERTa NLI model (cross-encoder approach).

    Architecture:
    1. Pairwise NLI classification (entailment/neutral/contradiction)
    2. Contradiction graph construction
    3. Graph-based metrics (clustering coefficient, avg path length)
    4. Weighted consistency score

    Model: Uses cross-encoder/nli-deberta-v3-base for bidirectional NLI
    """

    def __init__(self,
                 nli_model_name: str = "cross-encoder/nli-deberta-v3-base",
                 nli_threshold: float = 0.5,
                 output_dir: str = "data/processed/consistency_graphs",
                 device: Optional[str] = None):
        """
        Initialize Consistency Engine with DeBERTa cross-encoder NLI model.

        Args:
            nli_model_name: HuggingFace model identifier for NLI cross-encoder
            nli_threshold: Threshold for contradiction detection (0-1)
            output_dir: Directory to save visualization graphs
            device: Device to use ('cuda', 'cpu', or None for auto-detect)
        """
        self.nli_model_name = nli_model_name
        self.nli_threshold = nli_threshold
        self.output_dir = output_dir

        # Auto-detect device
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        print(f"[ConsistencyEngine] Initializing with {nli_model_name} on {self.device}...")

        # Load DeBERTa NLI cross-encoder model
        try:
            # Try multiple NLI models in order of preference
            nli_models = [
                "cross-encoder/nli-deberta-v3-base",      # Best option: trained specifically for NLI
                "cross-encoder/nli-deberta-v3-large",     # Larger version
                "microsoft/deberta-v3-base",              # Base model (no NLI fine-tuning)
            ]

            model_loaded = False
            for model_name in nli_models:
                try:
                    print(f"[ConsistencyEngine] Trying to load {model_name}...")
                    self.tokenizer = AutoTokenizer.from_pretrained(model_name)
                    self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
                    self.model.to(self.device)
                    self.model.eval()
                    self.nli_model_name = model_name

                    # Check number of labels
                    if hasattr(self.model.config, 'num_labels'):
                        num_labels = self.model.config.num_labels
                        print(f"[ConsistencyEngine] Model has {num_labels} labels")

                        # Set label mapping based on model type
                        if num_labels == 3:
                            # Standard MNLI format: 0=entailment, 1=neutral, 2=contradiction
                            self.label_mapping = {
                                0: "entailment",
                                1: "neutral",
                                2: "contradiction"
                            }
                        elif num_labels == 2:
                            # Binary NLI: 0=entailment, 1=not_entailment
                            self.label_mapping = {
                                0: "entailment",
                                1: "contradiction"
                            }
                        else:
                            raise ValueError(f"Unexpected number of labels: {num_labels}")

                    print(f"[ConsistencyEngine] Successfully loaded {model_name}")
                    model_loaded = True
                    break
                except Exception as e:
                    print(f"[ConsistencyEngine] Failed to load {model_name}: {e}")
                    continue

            if not model_loaded:
                raise Exception("Could not load any DeBERTa NLI model")

        except Exception as e:
            print(f"[ConsistencyEngine] Error loading NLI model: {e}")
            raise

        # Create output directory
        os.makedirs(self.output_dir, exist_ok=True)

    def predict_nli(self, premise: str, hypothesis: str) -> Tuple[str, float]:
        """
        Predict NLI relationship between premise and hypothesis using cross-encoder.

        Args:
            premise: First text (reference)
            hypothesis: Second text (to compare)

        Returns:
            (label, confidence) where label is 'entailment', 'neutral', or 'contradiction'
        """
        # Tokenize inputs (cross-encoder format: premise + [SEP] + hypothesis)
        inputs = self.tokenizer(
            premise,
            hypothesis,
            return_tensors="pt",
            truncation=True,
            max_length=512,
            padding=True
        ).to(self.device)

        # Get predictions
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            probabilities = torch.softmax(logits, dim=1)[0]

        # Get predicted label and confidence
        predicted_label_id = torch.argmax(probabilities).item()
        confidence = probabilities[predicted_label_id].item()

        label = self.label_mapping[predicted_label_id]

        return label, confidence

    def build_contradiction_matrix(self,
                                   responses: List[str],
                                   response_labels: Optional[List[str]] = None,
                                   use_answer_only: bool = True) -> np.ndarray:
        """
        Build pairwise contradiction matrix using NLI model.

        HYBRID FORMAT: When use_answer_only=True, extracts and compares only
        the ANSWER portion of responses. This reduces false positives caused
        by explanation style variance.

        For each pair (i,j):
        - Predict NLI in both directions
        - Calculate contradiction score based on predictions

        Args:
            responses: List of response texts
            response_labels: Optional labels for responses (for visualization)
            use_answer_only: If True, extract and compare only ANSWER portions

        Returns:
            N x N matrix where M[i][j] = contradiction score between response i and j
        """
        n = len(responses)
        matrix = np.zeros((n, n))

        # Extract answer portions if using hybrid format
        if use_answer_only:
            comparison_texts = [extract_answer(r) for r in responses]
            print(f"[ConsistencyEngine] Using ANSWER-only comparison (hybrid format)")
        else:
            comparison_texts = responses

        print(f"[ConsistencyEngine] Building {n}x{n} contradiction matrix...")

        for i in range(n):
            for j in range(i+1, n):
                # Predict NLI in both directions (NLI is not symmetric)
                # Use extracted answers for comparison
                label_ij, conf_ij = self.predict_nli(comparison_texts[i], comparison_texts[j])
                label_ji, conf_ji = self.predict_nli(comparison_texts[j], comparison_texts[i])

                # Calculate contradiction score (0-1)
                # High score = strong contradiction
                contradiction_score = 0.0

                if label_ij == "contradiction" or label_ji == "contradiction":
                    # If either direction shows contradiction, use the max confidence
                    contradiction_score = max(
                        conf_ij if label_ij == "contradiction" else 0,
                        conf_ji if label_ji == "contradiction" else 0
                    )
                elif label_ij == "neutral" and label_ji == "neutral":
                    # Both neutral = some inconsistency (but not full contradiction)
                    contradiction_score = (conf_ij + conf_ji) / 2 * 0.5
                elif label_ij == "neutral" or label_ji == "neutral":
                    # One neutral, one entailment = slight inconsistency
                    neutral_conf = conf_ij if label_ij == "neutral" else conf_ji
                    contradiction_score = neutral_conf * 0.3
                else:
                    # Both entailment = fully consistent
                    contradiction_score = 0.0

                matrix[i][j] = contradiction_score
                matrix[j][i] = contradiction_score

        return matrix

    def build_graph(self,
                   contradiction_matrix: np.ndarray,
                   response_labels: Optional[List[str]] = None) -> nx.Graph:
        """
        Build weighted graph from contradiction matrix.
        Edges represent contradictions (higher weight = stronger contradiction).

        Args:
            contradiction_matrix: N x N contradiction matrix
            response_labels: Optional labels for nodes

        Returns:
            NetworkX graph
        """
        n = contradiction_matrix.shape[0]
        G = nx.Graph()

        # Add nodes
        if response_labels:
            for i, label in enumerate(response_labels):
                G.add_node(i, label=label)
        else:
            for i in range(n):
                G.add_node(i, label=f"R{i+1}")

        # Add edges for contradictions above threshold
        for i in range(n):
            for j in range(i+1, n):
                if contradiction_matrix[i][j] > self.nli_threshold:
                    G.add_edge(i, j, weight=contradiction_matrix[i][j])

        return G

    def calculate_graph_metrics(self, G: nx.Graph) -> Dict[str, float]:
        """
        Calculate graph-based consistency metrics.

        Args:
            G: NetworkX graph

        Returns:
            Dictionary of metrics
        """
        metrics = {}

        # Number of nodes and edges
        num_nodes = G.number_of_nodes()
        num_edges = G.number_of_edges()

        metrics['num_responses'] = num_nodes
        metrics['num_contradictions'] = num_edges

        # Density: ratio of actual edges to possible edges
        # Low density = few contradictions = high consistency
        if num_nodes > 1:
            max_edges = num_nodes * (num_nodes - 1) / 2
            metrics['graph_density'] = num_edges / max_edges if max_edges > 0 else 0
        else:
            metrics['graph_density'] = 0

        # Average clustering coefficient
        # High clustering = contradictions are grouped (might indicate subgroups)
        if num_edges > 0:
            metrics['avg_clustering'] = nx.average_clustering(G)
        else:
            metrics['avg_clustering'] = 0

        # Number of connected components
        # More components = more fragmented responses
        metrics['num_components'] = nx.number_connected_components(G)

        return metrics

    def calculate_consistency_score(self,
                                    contradiction_matrix: np.ndarray,
                                    graph_metrics: Dict[str, float]) -> float:
        """
        Calculate overall consistency score (0-1, higher = more consistent).

        Formula:
        consistency = 1 - (α * density + β * avg_contradiction + γ * fragmentation)

        Where:
        - density: graph density (ratio of contradictions to possible pairs)
        - avg_contradiction: average contradiction strength
        - fragmentation: degree of response fragmentation

        Args:
            contradiction_matrix: N x N contradiction matrix
            graph_metrics: Graph metrics

        Returns:
            Consistency score (0-1)
        """
        # Weights (as specified in plan.md)
        alpha = 0.4  # Importance of graph density
        beta = 0.4   # Importance of average contradiction strength
        gamma = 0.2  # Importance of fragmentation

        # Graph density (0-1)
        density = graph_metrics['graph_density']

        # Average contradiction strength (0-1)
        n = contradiction_matrix.shape[0]
        total_pairs = n * (n - 1) / 2 if n > 1 else 1
        avg_contradiction = np.sum(contradiction_matrix) / (2 * total_pairs) if total_pairs > 0 else 0

        # Fragmentation: normalized number of components
        num_components = graph_metrics['num_components']
        num_nodes = graph_metrics['num_responses']
        fragmentation = (num_components - 1) / (num_nodes - 1) if num_nodes > 1 else 0

        # Inconsistency score (higher = more inconsistent)
        inconsistency = alpha * density + beta * avg_contradiction + gamma * fragmentation

        # Consistency score (higher = better)
        consistency = 1 - inconsistency

        # Clamp to [0, 1]
        consistency = max(0.0, min(1.0, consistency))

        return consistency

    def visualize(self,
                 G: nx.Graph,
                 contradiction_matrix: np.ndarray,
                 question_id: str,
                 response_labels: Optional[List[str]] = None) -> str:
        """
        Visualize contradiction graph and matrix.

        Args:
            G: NetworkX graph
            contradiction_matrix: N x N contradiction matrix
            question_id: Question ID for filename
            response_labels: Optional labels for responses

        Returns:
            Path to saved visualization
        """
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))

        # Plot 1: Contradiction Matrix Heatmap
        ax1 = axes[0]
        n = contradiction_matrix.shape[0]
        labels = response_labels if response_labels else [f"R{i+1}" for i in range(n)]

        sns.heatmap(
            contradiction_matrix,
            annot=True,
            fmt='.2f',
            cmap='RdYlGn_r',
            vmin=0,
            vmax=1,
            xticklabels=labels,
            yticklabels=labels,
            ax=ax1,
            cbar_kws={'label': 'Contradiction Score'}
        )
        ax1.set_title(f'Contradiction Matrix\n{question_id}', fontsize=12, fontweight='bold')

        # Plot 2: Graph Visualization
        ax2 = axes[1]
        pos = nx.spring_layout(G, seed=42, k=2)

        # Node colors based on degree (number of contradictions)
        node_degrees = dict(G.degree())
        node_colors = [node_degrees[node] for node in G.nodes()]

        # Draw nodes
        nx.draw_networkx_nodes(
            G, pos,
            node_color=node_colors,
            node_size=800,
            cmap='YlOrRd',
            ax=ax2,
            alpha=0.9
        )

        # Draw edges with width based on weight
        edges = G.edges()
        if len(edges) > 0:
            weights = [G[u][v]['weight'] for u, v in edges]
            nx.draw_networkx_edges(
                G, pos,
                width=[w * 3 for w in weights],
                alpha=0.6,
                edge_color='red',
                ax=ax2
            )

        # Draw labels
        node_labels = nx.get_node_attributes(G, 'label')
        nx.draw_networkx_labels(
            G, pos,
            labels=node_labels,
            font_size=10,
            font_weight='bold',
            ax=ax2
        )

        ax2.set_title(f'Contradiction Graph\n{question_id}', fontsize=12, fontweight='bold')
        ax2.axis('off')

        plt.tight_layout()

        # Save
        output_path = os.path.join(self.output_dir, f'{question_id}_consistency.png')
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()

        return output_path

    def analyze(self,
               responses: List[str],
               question_id: str,
               response_labels: Optional[List[str]] = None,
               visualize: bool = True) -> ConsistencyAnalysis:
        """
        Main analysis pipeline.

        Args:
            responses: List of response texts to analyze
            question_id: Identifier for this question
            response_labels: Optional labels for responses (e.g., 'Original', 'Variant1')
            visualize: Whether to generate visualization

        Returns:
            ConsistencyAnalysis object with results
        """
        print(f"\n[ConsistencyEngine] Analyzing {len(responses)} responses for {question_id}...")

        if len(responses) < 2:
            print("[ConsistencyEngine] Warning: Less than 2 responses, returning perfect consistency")
            return ConsistencyAnalysis(
                question_id=question_id,
                consistency_score=1.0,
                contradiction_matrix=[],
                graph_metrics={'num_responses': len(responses), 'num_contradictions': 0},
                visualization_path=None
            )

        # Step 1: Build contradiction matrix
        contradiction_matrix = self.build_contradiction_matrix(responses, response_labels)

        # Step 2: Build graph
        G = self.build_graph(contradiction_matrix, response_labels)

        # Step 3: Calculate metrics
        graph_metrics = self.calculate_graph_metrics(G)

        # Step 4: Calculate consistency score
        consistency_score = self.calculate_consistency_score(contradiction_matrix, graph_metrics)

        print(f"[ConsistencyEngine] Consistency Score: {consistency_score:.4f}")
        print(f"[ConsistencyEngine] Graph Density: {graph_metrics['graph_density']:.4f}")
        print(f"[ConsistencyEngine] Contradictions: {graph_metrics['num_contradictions']}")

        # Step 5: Visualize (optional)
        viz_path = None
        if visualize:
            viz_path = self.visualize(G, contradiction_matrix, question_id, response_labels)
            print(f"[ConsistencyEngine] Visualization saved: {viz_path}")

        return ConsistencyAnalysis(
            question_id=question_id,
            consistency_score=consistency_score,
            contradiction_matrix=contradiction_matrix.tolist(),
            graph_metrics=graph_metrics,
            visualization_path=viz_path
        )


if __name__ == "__main__":
    # Test the Consistency Engine with DeBERTa cross-encoder
    print("=" * 70)
    print("Testing Consistency Engine with DeBERTa NLI Cross-Encoder")
    print("=" * 70)

    # Initialize engine
    engine = ConsistencyEngine(device="cpu")  # Use CPU for testing

    # Test Case 1: Consistent responses
    print("\n" + "=" * 70)
    print("TEST 1: Consistent Responses")
    print("=" * 70)

    consistent_responses = [
        "Yes, Socrates is mortal because all humans are mortal and Socrates is human.",
        "Socrates must be mortal since he is human and all humans are mortal.",
        "Definitely yes. Socrates is a human, and humans are mortal, so Socrates is mortal."
    ]

    result1 = engine.analyze(
        responses=consistent_responses,
        question_id="test_consistent",
        response_labels=["R1", "R2", "R3"],
        visualize=True
    )

    print(f"\n✓ Consistency Score: {result1.consistency_score:.4f}")
    print(f"✓ Expected: High score (>0.8)")

    # Test Case 2: Inconsistent responses
    print("\n" + "=" * 70)
    print("TEST 2: Inconsistent Responses")
    print("=" * 70)

    inconsistent_responses = [
        "Yes, penguins can fly because all birds can fly.",
        "No, penguins cannot fly. They are flightless birds.",
        "Penguins are excellent flyers and migrate long distances."
    ]

    result2 = engine.analyze(
        responses=inconsistent_responses,
        question_id="test_inconsistent",
        response_labels=["R1", "R2", "R3"],
        visualize=True
    )

    print(f"\n✓ Consistency Score: {result2.consistency_score:.4f}")
    print(f"✓ Expected: Low score (<0.5)")

    print("\n" + "=" * 70)
    print("✓ All tests completed!")
    print("Check data/processed/consistency_graphs/ for visualizations")
    print("=" * 70)
