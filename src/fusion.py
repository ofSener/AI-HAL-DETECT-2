"""
MODULE E: Fusion Layer - Final Decision Mechanism
Combines 5 signals to detect hallucinations:
  1. Self-verification score (GT contradiction: response vs ground truth)
  2. Inconsistency score (self-consistency between variants)
  3. Entropy score (normalized token entropy)
  4. NCD score (normalized compression distance)
  5. Minority penalty (answer-validator minority vote penalty)

Formula:
  HallucinationRisk = w1·self_verification + w2·inconsistency
                    + w3·entropy + w4·ncd + w5·minority_penalty

All weights are normalized to sum to 1.0.
"""

import yaml
import logging
import json
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field, asdict
import numpy as np
import matplotlib.pyplot as plt


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Canonical signal names used throughout the module
SIGNAL_NAMES = [
    "self_verification",
    "inconsistency",
    "entropy",
    "ncd",
    "minority_penalty",
]


@dataclass
class HallucinationDetection:
    """Final hallucination detection result with full 5-signal support."""
    question_id: str
    variant_type: str

    # Individual component scores (all 0-1, higher = more suspicious)
    self_verification_score: float   # GT contradiction score
    inconsistency_score: float       # Self-consistency (inverted: 1 - consistency)
    entropy_score: float             # Normalized entropy
    ncd_score: float                 # Normalized compression distance
    minority_penalty: float          # Answer-validator minority penalty

    # Fusion
    hallucination_risk: float        # 0-1, higher = more likely hallucination
    is_hallucination: bool           # Binary decision
    confidence: str                  # 'high', 'medium', 'low'

    # Explainability
    signal_contributions: Dict[str, float] = field(default_factory=dict)
    explanation: str = ""
    contributing_factors: List[str] = field(default_factory=list)

    def to_dict(self):
        """Convert to dictionary."""
        return asdict(self)


class FusionLayer:
    """
    Combines 5 signals from all modules to make final hallucination decision.

    Formula:
      HallucinationRisk = w1·self_verification + w2·inconsistency
                        + w3·entropy + w4·ncd + w5·minority_penalty

    Weights are loaded from config and normalized to sum to 1.0.
    Backward compatible: falls back to legacy alpha/beta/gamma/delta keys
    when new named-weight keys are not present.
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

        # ------------------------------------------------------------------
        # Load named weights (with backward-compatible fallback)
        # ------------------------------------------------------------------
        if 'weight_self_verification' in self.config:
            # New-style named weights
            self.weights = {
                "self_verification": float(self.config['weight_self_verification']),
                "inconsistency":     float(self.config['weight_inconsistency']),
                "entropy":           float(self.config['weight_entropy']),
                "ncd":               float(self.config['weight_ncd']),
                "minority_penalty":  float(self.config['weight_minority_penalty']),
            }
        else:
            # Legacy alpha/beta/gamma/delta keys
            logger.info("New weight keys not found; falling back to legacy alpha/beta/gamma/delta keys")
            self.weights = {
                "self_verification": float(self.config.get('alpha', 0.0)),
                "inconsistency":     float(self.config.get('beta', 0.0)),
                "entropy":           float(self.config.get('gamma', 0.0)),
                "ncd":               float(self.config.get('delta', 0.0)),
                "minority_penalty":  float(self.config.get('epsilon', 0.0)),
            }

        # Normalize weights to sum to 1.0
        weight_sum = sum(self.weights.values())
        if weight_sum == 0:
            raise ValueError("All fusion weights are zero; cannot normalize.")
        if not np.isclose(weight_sum, 1.0, atol=0.01):
            logger.warning(f"Fusion weights sum to {weight_sum:.4f}, not 1.0. Normalizing...")
        for key in self.weights:
            self.weights[key] /= weight_sum

        # Decision threshold
        self.threshold = float(self.config['hallucination_threshold'])

        # Confidence thresholds
        self.high_confidence = float(self.config['high_confidence'])
        self.low_confidence = float(self.config['low_confidence'])

        weight_str = ", ".join(f"{k}={v:.4f}" for k, v in self.weights.items())
        logger.info(f"Fusion Layer initialized ({weight_str}, threshold={self.threshold:.4f})")

    # ------------------------------------------------------------------
    # Core risk calculation
    # ------------------------------------------------------------------

    def calculate_hallucination_risk(
        self, signals: Dict[str, float]
    ) -> Tuple[float, Dict[str, float]]:
        """
        Calculate hallucination risk score from 5 signals.

        Args:
            signals: Dict mapping signal names to scores (all 0-1,
                     higher = more suspicious).  Expected keys:
                     self_verification, inconsistency, entropy, ncd,
                     minority_penalty.

        Returns:
            Tuple of (risk, contributions) where risk is 0-1 and
            contributions maps each signal name to its weighted
            contribution to the total risk.
        """
        contributions: Dict[str, float] = {}
        risk = 0.0

        for name in SIGNAL_NAMES:
            value = float(signals.get(name, 0.0))
            weight = self.weights.get(name, 0.0)
            contrib = weight * value
            contributions[name] = contrib
            risk += contrib

        # Clamp to [0, 1]
        risk = float(np.clip(risk, 0.0, 1.0))

        return risk, contributions

    # ------------------------------------------------------------------
    # Confidence
    # ------------------------------------------------------------------

    def determine_confidence(self, risk_score: float) -> str:
        """
        Determine confidence level of the decision.

        Args:
            risk_score: Hallucination risk score

        Returns:
            Confidence level: 'high', 'medium', or 'low'
        """
        distance = abs(risk_score - self.threshold)

        if distance >= (self.high_confidence - self.threshold):
            return 'high'
        elif distance >= (self.threshold - self.low_confidence):
            return 'medium'
        else:
            return 'low'

    # ------------------------------------------------------------------
    # Explanation
    # ------------------------------------------------------------------

    def generate_explanation(
        self,
        signals: Dict[str, float],
        contributions: Dict[str, float],
        risk_score: float,
    ) -> Tuple[str, List[str]]:
        """
        Generate human-readable explanation using signal contributions.

        Args:
            signals: Raw signal values dict.
            contributions: Per-signal weighted contributions dict.
            risk_score: Final aggregated risk score.

        Returns:
            Tuple of (explanation string, list of contributing factors).
        """
        # Readable labels for each signal
        labels = {
            "self_verification": "GT contradiction",
            "inconsistency":     "response inconsistency",
            "entropy":           "high uncertainty (entropy)",
            "ncd":               "abnormal compression (NCD)",
            "minority_penalty":  "minority-answer penalty",
        }

        # Thresholds above which a signal is considered a contributing factor
        alert_thresholds = {
            "self_verification": 0.5,
            "inconsistency":     0.5,
            "entropy":           0.6,
            "ncd":               0.6,
            "minority_penalty":  0.3,
        }

        factors: List[str] = []
        for name in SIGNAL_NAMES:
            value = signals.get(name, 0.0)
            thresh = alert_thresholds.get(name, 0.5)
            if value > thresh:
                label = labels.get(name, name)
                contrib_pct = (contributions.get(name, 0.0) / risk_score * 100) if risk_score > 0 else 0
                factors.append(f"{label} ({value:.2f}, {contrib_pct:.0f}% of risk)")

        # Build explanation string
        if risk_score > self.threshold:
            if len(factors) == 0:
                explanation = (
                    "Hallucination detected based on overall risk score, "
                    "though individual signals are moderate."
                )
            elif len(factors) == 1:
                explanation = f"Hallucination likely due to: {factors[0]}"
            else:
                explanation = (
                    f"Hallucination likely due to multiple factors: "
                    f"{'; '.join(factors)}"
                )
        else:
            explanation = (
                "No hallucination detected. Response appears consistent and confident."
            )

        return explanation, factors

    # ------------------------------------------------------------------
    # Main detect entry point
    # ------------------------------------------------------------------

    def detect(
        self,
        question_id: str,
        variant_type: str,
        signals: Dict[str, float],
    ) -> HallucinationDetection:
        """
        Perform final hallucination detection from 5 named signals.

        Args:
            question_id: Question identifier.
            variant_type: Type of response variant (e.g. 'original', 'paraphrase').
            signals: Dict with keys:
                self_verification  - GT contradiction score (0-1)
                inconsistency      - Self-inconsistency score (0-1)
                entropy            - Normalized entropy (0-1)
                ncd                - Normalized NCD (0-1)
                minority_penalty   - Answer-validator minority penalty (0-1)

        Returns:
            HallucinationDetection object.
        """
        # Calculate risk and per-signal contributions
        risk, contributions = self.calculate_hallucination_risk(signals)

        # Binary decision
        is_hallucination = risk > self.threshold

        # Confidence
        confidence = self.determine_confidence(risk)

        # Explanation
        explanation, factors = self.generate_explanation(signals, contributions, risk)

        logger.info(
            f"[{question_id}/{variant_type}] Risk: {risk:.4f}, "
            f"Hallucination: {is_hallucination}, Confidence: {confidence}"
        )

        return HallucinationDetection(
            question_id=question_id,
            variant_type=variant_type,
            self_verification_score=signals.get("self_verification", 0.0),
            inconsistency_score=signals.get("inconsistency", 0.0),
            entropy_score=signals.get("entropy", 0.0),
            ncd_score=signals.get("ncd", 0.0),
            minority_penalty=signals.get("minority_penalty", 0.0),
            hallucination_risk=risk,
            is_hallucination=is_hallucination,
            confidence=confidence,
            signal_contributions=contributions,
            explanation=explanation,
            contributing_factors=factors,
        )

    # ------------------------------------------------------------------
    # Batch detect
    # ------------------------------------------------------------------

    def batch_detect(self, analyses: List[Dict]) -> List[HallucinationDetection]:
        """
        Detect hallucinations for multiple responses.

        Args:
            analyses: List of dicts. Each dict must contain:
                'question_id' (str)
                'signals' (Dict[str, float]) with the 5 signal keys.
              Optional:
                'variant_type' (str, defaults to 'original')

        Returns:
            List of HallucinationDetection objects.
        """
        results = []

        for analysis in analyses:
            detection = self.detect(
                question_id=analysis['question_id'],
                variant_type=analysis.get('variant_type', 'original'),
                signals=analysis['signals'],
            )
            results.append(detection)

        return results

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save_results(self, detections: List[HallucinationDetection],
                     output_path: str):
        """
        Save detection results to JSON file.

        Args:
            detections: List of HallucinationDetection objects.
            output_path: Path to save JSON.
        """
        data = [d.to_dict() for d in detections]

        with open(output_path, 'w') as f:
            json.dump(data, f, indent=2)

        logger.info(f"Saved {len(detections)} detection results to {output_path}")

    # ------------------------------------------------------------------
    # Visualization
    # ------------------------------------------------------------------

    def visualize_decision(self, detection: HallucinationDetection,
                           output_path: str):
        """
        Visualize the fusion decision for a single response showing all
        5 signals, their weights, and a risk gauge.

        Args:
            detection: HallucinationDetection object.
            output_path: Path to save visualization.
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

        # ----------------------------------------------------------
        # Plot 1: All 5 signal scores with weight annotations
        # ----------------------------------------------------------
        signal_labels = [
            "Self-Verification\n(GT Contradiction)",
            "Inconsistency\n(Self-Consistency)",
            "Entropy\n(Normalized)",
            "NCD\n(Normalized)",
            "Minority\nPenalty",
        ]
        scores = [
            detection.self_verification_score,
            detection.inconsistency_score,
            detection.entropy_score,
            detection.ncd_score,
            detection.minority_penalty,
        ]
        weight_values = [self.weights[name] for name in SIGNAL_NAMES]
        colors = ['coral', 'steelblue', 'seagreen', 'mediumpurple', 'goldenrod']

        x_pos = np.arange(len(signal_labels))
        bars = ax1.bar(x_pos, scores, color=colors, alpha=0.7)

        # Add weight labels on bars
        for bar, weight in zip(bars, weight_values):
            height = bar.get_height()
            ax1.text(
                bar.get_x() + bar.get_width() / 2.0, height + 0.02,
                f'w={weight:.3f}',
                ha='center', va='bottom', fontsize=9, fontweight='bold',
            )

        ax1.set_ylabel('Score', fontsize=12)
        ax1.set_title('Signal Scores (5-Signal Fusion)', fontsize=14, fontweight='bold')
        ax1.set_xticks(x_pos)
        ax1.set_xticklabels(signal_labels, fontsize=9)
        ax1.set_ylim(0, 1.15)
        ax1.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, label='Mid-point')
        ax1.grid(axis='y', alpha=0.3)
        ax1.legend()

        # ----------------------------------------------------------
        # Plot 2: Risk gauge (same style as original)
        # ----------------------------------------------------------
        ax2.axis('off')

        # Draw gauge arc
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
        ax2.plot(
            [0, r * 0.9 * np.cos(risk_angle)],
            [0, r * 0.9 * np.sin(risk_angle)],
            'r-', linewidth=4,
            label=f'Risk: {detection.hallucination_risk:.3f}',
        )

        # Threshold marker
        threshold_angle = np.pi * (1 - self.threshold)
        ax2.plot(
            [r * 0.7 * np.cos(threshold_angle), r * 1.1 * np.cos(threshold_angle)],
            [r * 0.7 * np.sin(threshold_angle), r * 1.1 * np.sin(threshold_angle)],
            'b--', linewidth=2,
            label=f'Threshold: {self.threshold:.2f}',
        )

        ax2.set_xlim(-1.3, 1.3)
        ax2.set_ylim(-0.2, 1.3)
        ax2.set_aspect('equal')
        ax2.legend(loc='upper right')

        # Decision text
        decision_text = "HALLUCINATION" if detection.is_hallucination else "NO HALLUCINATION"
        decision_color = "red" if detection.is_hallucination else "green"

        ax2.text(
            0, -0.15, decision_text,
            ha='center', va='top', fontsize=16, fontweight='bold',
            color=decision_color,
            bbox=dict(boxstyle='round', facecolor='white',
                      edgecolor=decision_color, linewidth=2),
        )

        plt.suptitle(
            f"Hallucination Detection: {detection.question_id} ({detection.variant_type})\n"
            f"Confidence: {detection.confidence.upper()}",
            fontsize=16, fontweight='bold',
        )
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()

        logger.info(f"Decision visualization saved to {output_path}")


# ======================================================================
# Demo / self-test
# ======================================================================
if __name__ == "__main__":
    print("=" * 70)
    print("MODULE E: Fusion Layer - 5-Signal Demo")
    print("=" * 70)

    # Initialize fusion layer
    fusion = FusionLayer()

    # ------------------------------------------------------------------
    # Test 1: Consistent, confident response (NO hallucination expected)
    # ------------------------------------------------------------------
    print("\n[TEST 1] Consistent & confident response:")
    detection1 = fusion.detect(
        question_id="test_001",
        variant_type="original",
        signals={
            "self_verification": 0.10,  # Low GT contradiction
            "inconsistency":     0.05,  # Very consistent
            "entropy":           0.20,  # Low uncertainty
            "ncd":               0.30,  # Normal compression
            "minority_penalty":  0.00,  # Majority answer
        },
    )
    print(f"  Risk: {detection1.hallucination_risk:.4f}")
    print(f"  Hallucination: {detection1.is_hallucination}")
    print(f"  Confidence: {detection1.confidence}")
    print(f"  Explanation: {detection1.explanation}")
    print(f"  Contributions: {detection1.signal_contributions}")

    # ------------------------------------------------------------------
    # Test 2: Inconsistent, uncertain response (HALLUCINATION expected)
    # ------------------------------------------------------------------
    print("\n[TEST 2] Inconsistent & uncertain response:")
    detection2 = fusion.detect(
        question_id="test_002",
        variant_type="paraphrase",
        signals={
            "self_verification": 0.85,  # High GT contradiction
            "inconsistency":     0.65,  # Inconsistent
            "entropy":           0.75,  # High uncertainty
            "ncd":               0.65,  # Abnormal compression
            "minority_penalty":  0.40,  # Minority answer
        },
    )
    print(f"  Risk: {detection2.hallucination_risk:.4f}")
    print(f"  Hallucination: {detection2.is_hallucination}")
    print(f"  Confidence: {detection2.confidence}")
    print(f"  Explanation: {detection2.explanation}")
    print(f"  Factors: {detection2.contributing_factors}")
    print(f"  Contributions: {detection2.signal_contributions}")

    # ------------------------------------------------------------------
    # Test 3: Batch processing
    # ------------------------------------------------------------------
    print("\n[TEST 3] Batch processing:")
    test_cases = [
        {
            'question_id': 'q1',
            'variant_type': 'original',
            'signals': {
                "self_verification": 0.10,
                "inconsistency":     0.10,
                "entropy":           0.30,
                "ncd":               0.40,
                "minority_penalty":  0.00,
            },
        },
        {
            'question_id': 'q1',
            'variant_type': 'paraphrase',
            'signals': {
                "self_verification": 0.80,
                "inconsistency":     0.60,
                "entropy":           0.80,
                "ncd":               0.70,
                "minority_penalty":  0.35,
            },
        },
        {
            'question_id': 'q2',
            'variant_type': 'original',
            'signals': {
                "self_verification": 0.40,
                "inconsistency":     0.40,
                "entropy":           0.50,
                "ncd":               0.50,
                "minority_penalty":  0.10,
            },
        },
    ]

    detections = fusion.batch_detect(test_cases)

    for d in detections:
        print(
            f"  [{d.question_id}/{d.variant_type}] "
            f"Risk: {d.hallucination_risk:.3f}, Hallucination: {d.is_hallucination}"
        )

    # Save results
    fusion.save_results(detections, "data/processed/results/fusion_demo_results.json")

    # ------------------------------------------------------------------
    # Test 4: Visualization
    # ------------------------------------------------------------------
    print("\n[TEST 4] Visualization:")
    fusion.visualize_decision(detection2, "data/processed/results/fusion_decision_viz.png")

    print("\n" + "=" * 70)
    print("Fusion Layer 5-signal demo complete!")
    print("=" * 70)
