"""
MODULE E: Fusion Layer - Final Decision Mechanism
Combines consistency, entropy, and compression signals to detect hallucinations.
"""

import yaml
import logging
import json
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
import numpy as np
import matplotlib.pyplot as plt


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class HallucinationDetection:
    """Final hallucination detection result"""
    question_id: str
    variant_type: str

    # Individual component scores
    consistency_score: float  # 0-1, higher = more consistent
    entropy_score: float  # 0-1 normalized
    ncd_score: float  # 0-1 normalized

    # Fusion
    hallucination_risk: float  # 0-1, higher = more likely hallucination
    is_hallucination: bool  # Binary decision
    confidence: str  # 'high', 'medium', 'low'

    # Explainability
    explanation: str
    contributing_factors: List[str]

    def to_dict(self):
        """Convert to dictionary"""
        return asdict(self)


class FusionLayer:
    """
    Combines signals from all modules to make final hallucination decision.

    Formula:
    HallucinationRisk = α·(1 - S_cons) + β·Entropy_norm + γ·NCD_norm

    where:
    - α, β, γ are fusion weights (sum to 1)
    - S_cons is consistency score (high = consistent)
    - Entropy_norm is normalized entropy
    - NCD_norm is normalized compression distance
    """

    def __init__(self, config_path: str = "config/config.yaml"):
        """
        Initialize the Fusion Layer.

        Args:
            config_path: Path to config.yaml
        """
        # Load configuration
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)['fusion']

        # Fusion weights
        self.alpha = self.config['alpha']  # Consistency weight
        self.beta = self.config['beta']  # Entropy weight
        self.gamma = self.config['gamma']  # NCD weight

        # Decision threshold
        self.threshold = self.config['hallucination_threshold']

        # Confidence thresholds
        self.high_confidence = self.config['high_confidence']
        self.low_confidence = self.config['low_confidence']

        # Validate weights
        weight_sum = self.alpha + self.beta + self.gamma
        if not np.isclose(weight_sum, 1.0, atol=0.01):
            logger.warning(f"Fusion weights sum to {weight_sum}, not 1.0. Normalizing...")
            self.alpha /= weight_sum
            self.beta /= weight_sum
            self.gamma /= weight_sum

        logger.info(f"Fusion Layer initialized (α={self.alpha:.2f}, β={self.beta:.2f}, γ={self.gamma:.2f}, τ={self.threshold:.2f})")

    def calculate_hallucination_risk(self, consistency_score: float,
                                     entropy_norm: float,
                                     ncd_norm: float) -> float:
        """
        Calculate hallucination risk score.

        Args:
            consistency_score: Consistency score (0-1, high = consistent)
            entropy_norm: Normalized entropy (0-1)
            ncd_norm: Normalized NCD (0-1)

        Returns:
            Hallucination risk (0-1, high = likely hallucination)
        """
        # Convert consistency to inconsistency
        inconsistency = 1.0 - consistency_score

        # Weighted combination
        risk = (
            self.alpha * inconsistency +
            self.beta * entropy_norm +
            self.gamma * ncd_norm
        )

        # Clamp to [0, 1]
        risk = np.clip(risk, 0.0, 1.0)

        return float(risk)

    def determine_confidence(self, risk_score: float) -> str:
        """
        Determine confidence level of the decision.

        Args:
            risk_score: Hallucination risk score

        Returns:
            Confidence level: 'high', 'medium', or 'low'
        """
        # Distance from threshold
        distance = abs(risk_score - self.threshold)

        if distance >= (self.high_confidence - self.threshold):
            return 'high'
        elif distance >= (self.threshold - self.low_confidence):
            return 'medium'
        else:
            return 'low'

    def generate_explanation(self, consistency_score: float,
                            entropy_norm: float,
                            ncd_norm: float,
                            risk_score: float) -> Tuple[str, List[str]]:
        """
        Generate human-readable explanation of the decision.

        Args:
            consistency_score: Consistency score
            entropy_norm: Normalized entropy
            ncd_norm: Normalized NCD
            risk_score: Final risk score

        Returns:
            Tuple of (explanation string, list of contributing factors)
        """
        factors = []
        inconsistency = 1.0 - consistency_score

        # Analyze each component
        if inconsistency > 0.5:
            factors.append(f"High inconsistency ({inconsistency:.2f}) - responses contradict each other")

        if entropy_norm > 0.6:
            factors.append(f"High uncertainty ({entropy_norm:.2f}) - model is not confident in its answer")

        if ncd_norm > 0.6:
            factors.append(f"Abnormal compression pattern ({ncd_norm:.2f}) - response structure is irregular")

        # Generate explanation
        if risk_score > self.threshold:
            if len(factors) == 0:
                explanation = "Hallucination detected based on overall risk score, though individual signals are moderate."
            elif len(factors) == 1:
                explanation = f"Hallucination likely due to: {factors[0]}"
            else:
                explanation = f"Hallucination likely due to multiple factors: {', '.join(factors[:2])}"
        else:
            explanation = "No hallucination detected. Response appears consistent and confident."

        return explanation, factors

    def detect(self, question_id: str, variant_type: str,
              consistency_score: float,
              entropy_norm: float,
              ncd_norm: float) -> HallucinationDetection:
        """
        Perform final hallucination detection.

        Args:
            question_id: Question identifier
            variant_type: Type of response variant
            consistency_score: Consistency score from MODULE C
            entropy_norm: Normalized entropy from MODULE D
            ncd_norm: Normalized NCD from MODULE D

        Returns:
            HallucinationDetection object
        """
        # Calculate risk
        risk = self.calculate_hallucination_risk(consistency_score, entropy_norm, ncd_norm)

        # Make decision
        is_hallucination = risk > self.threshold

        # Determine confidence
        confidence = self.determine_confidence(risk)

        # Generate explanation
        explanation, factors = self.generate_explanation(
            consistency_score, entropy_norm, ncd_norm, risk
        )

        logger.info(f"[{question_id}/{variant_type}] Risk: {risk:.4f}, Hallucination: {is_hallucination}, Confidence: {confidence}")

        return HallucinationDetection(
            question_id=question_id,
            variant_type=variant_type,
            consistency_score=consistency_score,
            entropy_score=entropy_norm,
            ncd_score=ncd_norm,
            hallucination_risk=risk,
            is_hallucination=is_hallucination,
            confidence=confidence,
            explanation=explanation,
            contributing_factors=factors
        )

    def batch_detect(self, analyses: List[Dict]) -> List[HallucinationDetection]:
        """
        Detect hallucinations for multiple responses.

        Args:
            analyses: List of dicts with keys:
                     'question_id', 'variant_type',
                     'consistency_score', 'entropy_norm', 'ncd_norm'

        Returns:
            List of HallucinationDetection objects
        """
        results = []

        for analysis in analyses:
            detection = self.detect(
                question_id=analysis['question_id'],
                variant_type=analysis.get('variant_type', 'original'),
                consistency_score=analysis['consistency_score'],
                entropy_norm=analysis['entropy_norm'],
                ncd_norm=analysis['ncd_norm']
            )
            results.append(detection)

        return results

    def save_results(self, detections: List[HallucinationDetection],
                    output_path: str):
        """
        Save detection results to JSON file.

        Args:
            detections: List of HallucinationDetection objects
            output_path: Path to save JSON
        """
        data = [d.to_dict() for d in detections]

        with open(output_path, 'w') as f:
            json.dump(data, f, indent=2)

        logger.info(f"✓ Saved {len(detections)} detection results to {output_path}")

    def visualize_decision(self, detection: HallucinationDetection,
                          output_path: str):
        """
        Visualize the fusion decision for a single response.

        Args:
            detection: HallucinationDetection object
            output_path: Path to save visualization
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

        # Plot 1: Component scores
        components = ['Inconsistency\n(1 - Consistency)', 'Entropy\n(normalized)', 'NCD\n(normalized)']
        scores = [
            1.0 - detection.consistency_score,
            detection.entropy_score,
            detection.ncd_score
        ]
        weights = [self.alpha, self.beta, self.gamma]

        x_pos = np.arange(len(components))
        bars = ax1.bar(x_pos, scores, color=['coral', 'steelblue', 'seagreen'], alpha=0.7)

        # Add weight labels on bars
        for i, (bar, weight) in enumerate(zip(bars, weights)):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                    f'w={weight:.2f}',
                    ha='center', va='bottom', fontsize=10, fontweight='bold')

        ax1.set_ylabel('Score', fontsize=12)
        ax1.set_title('Component Scores', fontsize=14, fontweight='bold')
        ax1.set_xticks(x_pos)
        ax1.set_xticklabels(components)
        ax1.set_ylim(0, 1.1)
        ax1.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, label='Mid-point')
        ax1.grid(axis='y', alpha=0.3)
        ax1.legend()

        # Plot 2: Final risk gauge
        ax2.axis('off')

        # Draw gauge
        theta = np.linspace(np.pi, 0, 100)
        r = 1

        # Background arc
        ax2.plot(r * np.cos(theta), r * np.sin(theta), 'k-', linewidth=3)

        # Color zones
        low_zone = theta[theta > np.pi * 0.7]
        med_zone = theta[(theta >= np.pi * 0.3) & (theta <= np.pi * 0.7)]
        high_zone = theta[theta < np.pi * 0.3]

        ax2.fill_between(r * np.cos(low_zone), 0, r * np.sin(low_zone),
                         color='green', alpha=0.3, label='Low Risk')
        ax2.fill_between(r * np.cos(med_zone), 0, r * np.sin(med_zone),
                         color='yellow', alpha=0.3, label='Medium Risk')
        ax2.fill_between(r * np.cos(high_zone), 0, r * np.sin(high_zone),
                         color='red', alpha=0.3, label='High Risk')

        # Needle (pointer)
        risk_angle = np.pi * (1 - detection.hallucination_risk)
        ax2.plot([0, r * 0.9 * np.cos(risk_angle)],
                [0, r * 0.9 * np.sin(risk_angle)],
                'r-', linewidth=4, label=f'Risk: {detection.hallucination_risk:.3f}')

        # Threshold marker
        threshold_angle = np.pi * (1 - self.threshold)
        ax2.plot([r * 0.7 * np.cos(threshold_angle), r * 1.1 * np.cos(threshold_angle)],
                [r * 0.7 * np.sin(threshold_angle), r * 1.1 * np.sin(threshold_angle)],
                'b--', linewidth=2, label=f'Threshold: {self.threshold:.2f}')

        ax2.set_xlim(-1.3, 1.3)
        ax2.set_ylim(-0.2, 1.3)
        ax2.set_aspect('equal')
        ax2.legend(loc='upper right')

        # Decision text
        decision_text = "HALLUCINATION" if detection.is_hallucination else "NO HALLUCINATION"
        decision_color = "red" if detection.is_hallucination else "green"

        ax2.text(0, -0.15, decision_text,
                ha='center', va='top', fontsize=16, fontweight='bold',
                color=decision_color,
                bbox=dict(boxstyle='round', facecolor='white', edgecolor=decision_color, linewidth=2))

        plt.suptitle(f"Hallucination Detection: {detection.question_id} ({detection.variant_type})\nConfidence: {detection.confidence.upper()}",
                    fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()

        logger.info(f"✓ Decision visualization saved to {output_path}")


# Test/Demo function
if __name__ == "__main__":
    print("="*70)
    print("MODULE E: Fusion Layer - Demo")
    print("="*70)

    # Initialize fusion layer
    fusion = FusionLayer()

    # Test case 1: Consistent, confident response (NO hallucination)
    print("\n[TEST 1] Consistent & confident response:")
    detection1 = fusion.detect(
        question_id="test_001",
        variant_type="original",
        consistency_score=0.95,  # Very consistent
        entropy_norm=0.2,  # Low uncertainty
        ncd_norm=0.3  # Normal compression
    )

    print(f"  Risk: {detection1.hallucination_risk:.4f}")
    print(f"  Hallucination: {detection1.is_hallucination}")
    print(f"  Confidence: {detection1.confidence}")
    print(f"  Explanation: {detection1.explanation}")

    # Test case 2: Inconsistent, uncertain response (HALLUCINATION)
    print("\n[TEST 2] Inconsistent & uncertain response:")
    detection2 = fusion.detect(
        question_id="test_002",
        variant_type="paraphrase",
        consistency_score=0.35,  # Low consistency
        entropy_norm=0.75,  # High uncertainty
        ncd_norm=0.65  # Abnormal compression
    )

    print(f"  Risk: {detection2.hallucination_risk:.4f}")
    print(f"  Hallucination: {detection2.is_hallucination}")
    print(f"  Confidence: {detection2.confidence}")
    print(f"  Explanation: {detection2.explanation}")
    print(f"  Factors: {detection2.contributing_factors}")

    # Batch processing
    print("\n[TEST 3] Batch processing:")
    test_cases = [
        {
            'question_id': 'q1',
            'variant_type': 'original',
            'consistency_score': 0.9,
            'entropy_norm': 0.3,
            'ncd_norm': 0.4
        },
        {
            'question_id': 'q1',
            'variant_type': 'paraphrase',
            'consistency_score': 0.4,
            'entropy_norm': 0.8,
            'ncd_norm': 0.7
        },
        {
            'question_id': 'q2',
            'variant_type': 'original',
            'consistency_score': 0.6,
            'entropy_norm': 0.5,
            'ncd_norm': 0.5
        }
    ]

    detections = fusion.batch_detect(test_cases)

    for d in detections:
        print(f"  [{d.question_id}/{d.variant_type}] Risk: {d.hallucination_risk:.3f}, Hallucination: {d.is_hallucination}")

    # Save results
    fusion.save_results(detections, "data/processed/results/fusion_demo_results.json")

    # Visualize
    print("\n[TEST 4] Visualization:")
    fusion.visualize_decision(detection2, "data/processed/results/fusion_decision_viz.png")

    print("\n" + "="*70)
    print("✓ Fusion Layer demo complete!")
    print("="*70)
