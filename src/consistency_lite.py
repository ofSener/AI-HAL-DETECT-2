"""
MODULE C: Consistency Engine (Lightweight Version)
Uses sentence similarity instead of NLI for faster prototyping.
"""

import yaml
import logging
import numpy as np
import networkx as nx
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
import matplotlib.pyplot as plt
from sentence_transformers import SentenceTransformer, util


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class ConsistencyAnalysis:
    """Results of consistency analysis"""
    question_id: str
    num_responses: int
    consistency_score: float  # 0 to 1, higher = more consistent
    contradiction_matrix: np.ndarray  # Pairwise contradiction scores
    edge_weights: Dict[Tuple[int, int], float]  # Graph edge weights
    graph: Optional[nx.Graph] = None
    visualization_path: Optional[str] = None


class ConsistencyEngineLite:
    """
    Lightweight consistency analyzer using semantic similarity.

    Logic:
    - High similarity = consistent responses (entailment/paraphrase)
    - Low similarity = contradictory or neutral responses
    - We invert similarity to get "contradiction score"
    """

    def __init__(self, config_path: str = "config/config.yaml"):
        """Initialize the Consistency Engine (Lite)"""
        # Load configuration
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)['consistency']

        # Initialize sentence similarity model
        logger.info("Loading Sentence-BERT model for consistency analysis...")
        self.similarity_model = SentenceTransformer('all-MiniLM-L6-v2')

        logger.info("✓ Consistency Engine (Lite) initialized")

    def compare_responses(self, response1: str, response2: str) -> Dict[str, float]:
        """
        Compare two responses using semantic similarity.

        Returns:
            Dict with estimated probabilities for entailment/neutral/contradiction
        """
        # Calculate semantic similarity
        embeddings = self.similarity_model.encode([response1, response2], convert_to_tensor=True)
        similarity = float(util.cos_sim(embeddings[0], embeddings[1]).item())

        # Heuristic mapping:
        # similarity > 0.9 → entailment
        # similarity 0.5-0.9 → neutral
        # similarity < 0.5 → contradiction

        if similarity > 0.9:
            scores = {
                'entailment': similarity,
                'neutral': 0.1,
                'contradiction': max(0, 1 - similarity - 0.1)
            }
        elif similarity > 0.5:
            scores = {
                'entailment': similarity * 0.3,
                'neutral': similarity,
                'contradiction': 1 - similarity
            }
        else:
            scores = {
                'entailment': similarity * 0.1,
                'neutral': 0.3,
                'contradiction': 1 - similarity
            }

        # Normalize
        total = sum(scores.values())
        scores = {k: v/total for k, v in scores.items()}

        return scores

    def build_contradiction_matrix(self, responses: List[str]) -> np.ndarray:
        """Build matrix of pairwise contradiction scores"""
        n = len(responses)
        matrix = np.zeros((n, n))

        # Compare all pairs
        for i in range(n):
            for j in range(i + 1, n):
                scores_ij = self.compare_responses(responses[i], responses[j])
                scores_ji = self.compare_responses(responses[j], responses[i])

                # Average contradiction scores
                contradiction_score = (
                    scores_ij.get('contradiction', 0) +
                    scores_ji.get('contradiction', 0)
                ) / 2.0

                matrix[i, j] = contradiction_score
                matrix[j, i] = contradiction_score

        return matrix

    def calculate_consistency_score(self, contradiction_matrix: np.ndarray) -> float:
        """Calculate overall consistency score"""
        n = contradiction_matrix.shape[0]

        if n <= 1:
            return 1.0

        # Get upper triangle (excluding diagonal)
        upper_triangle = np.triu(contradiction_matrix, k=1)

        # Calculate average contradiction
        num_pairs = n * (n - 1) / 2
        total_contradiction = np.sum(upper_triangle)
        avg_contradiction = total_contradiction / num_pairs

        # Consistency = 1 - contradiction
        consistency_score = 1.0 - avg_contradiction

        return consistency_score

    def build_graph(self, responses: List[str], contradiction_matrix: np.ndarray,
                   response_labels: Optional[List[str]] = None) -> nx.Graph:
        """Build NetworkX graph from contradiction matrix"""
        n = len(responses)
        G = nx.Graph()

        # Add nodes
        for i in range(n):
            label = response_labels[i] if response_labels else f"R{i}"
            # Truncate text for display
            text_preview = responses[i][:50] + "..." if len(responses[i]) > 50 else responses[i]
            G.add_node(i, label=label, text=text_preview)

        # Add edges for significant contradictions
        min_weight = self.config.get('min_edge_weight', 0.3)

        for i in range(n):
            for j in range(i + 1, n):
                weight = contradiction_matrix[i, j]

                if weight >= min_weight:
                    G.add_edge(i, j, weight=weight)

        return G

    def visualize_graph(self, graph: nx.Graph, output_path: str,
                       title: str = "Response Consistency Graph"):
        """Visualize the consistency graph"""
        plt.figure(figsize=(12, 8))

        # Layout
        pos = nx.spring_layout(graph, k=1, iterations=50, seed=42)

        # Draw nodes
        node_labels = nx.get_node_attributes(graph, 'label')
        nx.draw_networkx_nodes(
            graph, pos,
            node_size=3000,
            node_color='lightblue',
            alpha=0.9,
            edgecolors='navy',
            linewidths=2
        )

        # Draw edges (thicker = more contradiction)
        edges = list(graph.edges())
        if edges:
            weights = [graph[u][v]['weight'] for u, v in edges]
            nx.draw_networkx_edges(
                graph, pos,
                width=[w * 5 for w in weights],  # Scale for visibility
                alpha=0.6,
                edge_color=weights,
                edge_cmap=plt.cm.Reds,
                edge_vmin=0,
                edge_vmax=1
            )

            # Add edge labels (contradiction scores)
            edge_labels = {(u, v): f"{graph[u][v]['weight']:.2f}"
                          for u, v in edges}
            nx.draw_networkx_edge_labels(graph, pos, edge_labels, font_size=8)

        # Draw node labels
        nx.draw_networkx_labels(graph, pos, node_labels, font_size=10, font_weight='bold')

        plt.title(title, fontsize=16, fontweight='bold', pad=20)
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()

        logger.info(f"✓ Graph visualization saved to {output_path}")

    def analyze(self, responses: List[str], question_id: str,
               response_labels: Optional[List[str]] = None,
               visualize: bool = True,
               output_dir: str = "data/processed/results") -> ConsistencyAnalysis:
        """Perform full consistency analysis"""
        logger.info(f"Analyzing consistency for {question_id} ({len(responses)} responses)")

        # Build contradiction matrix
        contradiction_matrix = self.build_contradiction_matrix(responses)

        # Calculate consistency score
        consistency_score = self.calculate_consistency_score(contradiction_matrix)
        logger.info(f"  Consistency score: {consistency_score:.4f}")

        # Build graph
        graph = self.build_graph(responses, contradiction_matrix, response_labels)

        # Extract edge weights
        edge_weights = {(u, v): graph[u][v]['weight'] for u, v in graph.edges()}

        # Visualize
        visualization_path = None
        if visualize:
            visualization_path = f"{output_dir}/{question_id}_consistency_graph.png"
            self.visualize_graph(
                graph,
                visualization_path,
                title=f"Consistency Graph: {question_id}\nScore: {consistency_score:.3f}"
            )

        return ConsistencyAnalysis(
            question_id=question_id,
            num_responses=len(responses),
            consistency_score=consistency_score,
            contradiction_matrix=contradiction_matrix,
            edge_weights=edge_weights,
            graph=graph,
            visualization_path=visualization_path
        )


# Test/Demo function
if __name__ == "__main__":
    print("="*70)
    print("MODULE C: Consistency Engine (Lite) - Demo")
    print("="*70)

    # Initialize engine
    engine = ConsistencyEngineLite()

    # Test with sample responses
    print("\n[TEST 1] Consistent responses (Socrates):")
    consistent_responses = [
        "Yes, Socrates is mortal because all humans are mortal.",
        "Affirmative. Since all humans are mortal and Socrates is human, he is mortal.",
        "Yes. Given the premises, Socrates must be mortal."
    ]

    analysis1 = engine.analyze(
        consistent_responses,
        "test_consistent",
        response_labels=["original", "paraphrase", "reordered"],
        visualize=True
    )

    print(f"  ✓ Consistency Score: {analysis1.consistency_score:.4f}")
    print(f"  ✓ Graph edges (contradictions): {len(analysis1.edge_weights)}")
    print(f"  ✓ Visualization: {analysis1.visualization_path}")

    print("\n[TEST 2] Inconsistent responses (Penguins):")
    inconsistent_responses = [
        "Yes, penguins can fly because all birds can fly.",
        "No, penguins cannot fly. They are flightless birds.",
        "The premise is wrong. Not all birds can fly.",
        "Based on pure logic yes, but factually no."
    ]

    analysis2 = engine.analyze(
        inconsistent_responses,
        "test_inconsistent",
        response_labels=["original", "negation", "factual", "conditional"],
        visualize=True
    )

    print(f"  ✓ Consistency Score: {analysis2.consistency_score:.4f}")
    print(f"  ✓ Graph edges (contradictions): {len(analysis2.edge_weights)}")
    print(f"  ✓ Contradiction matrix:\n{analysis2.contradiction_matrix}")

    print("\n" + "="*70)
    print("✓ Demo complete!")
    print("  Check data/processed/results/ for visualizations")
    print("="*70)
