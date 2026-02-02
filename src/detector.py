"""
LOGIC-HALT Main Detector
Integrates all modules (A-E) into a complete hallucination detection pipeline.
"""

# Load environment variables first
from dotenv import load_dotenv
load_dotenv()

import json
import logging
import numpy as np
import yaml
import torch
from typing import Dict, List, Optional
from dataclasses import dataclass, asdict
import argparse

# Import all modules
from src.morpher import MorpherEngine
from src.interrogator import InterrogatorEngine
from src.consistency import ConsistencyEngine
from src.complexity import ComplexityEngine
from src.fusion import FusionLayer, HallucinationDetection


def get_device(config_path: str = "config/config.yaml") -> str:
    """Get device from config with auto-detection fallback"""
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        device = config.get('hardware', {}).get('device', 'auto')
    except:
        device = 'auto'

    if device == 'auto' or device == 'cuda':
        if torch.cuda.is_available():
            return 'cuda'
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            return 'mps'
    elif device == 'mps':
        if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            return 'mps'

    return 'cpu'


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class PipelineResult:
    """Complete pipeline result for a question"""
    question_id: str
    original_question: str
    ground_truth: Optional[str]

    # Module outputs
    num_variants_generated: int
    num_responses_collected: int
    consistency_score: float
    avg_entropy: float
    avg_ncd: float

    # Final detection
    hallucination_detected: bool
    hallucination_risk: float
    confidence: str
    explanation: str

    def to_dict(self):
        return asdict(self)


class HallucinationDetector:
    """
    Main LOGIC-HALT detector class.

    Pipeline:
    1. Morpher → Generate variants
    2. Interrogator → Query LLM
    3. Consistency Engine → Analyze contradictions
    4. Complexity Engine → Measure entropy/NCD
    5. Fusion Layer → Final decision
    """

    def __init__(self, config_path: str = "config/config.yaml",
                 use_mock: bool = True):
        """
        Initialize the detector.

        Args:
            config_path: Path to configuration file
            use_mock: Use mock versions (no API keys required)
        """
        logger.info("Initializing LOGIC-HALT Detector...")

        self.use_mock = use_mock

        # Initialize all modules
        try:
            if use_mock:
                from src.morpher_demo import MockMorpherEngine
                from src.pipeline_demo import MockInterrogatorEngine

                self.morpher = MockMorpherEngine(config_path)
                self.interrogator = MockInterrogatorEngine(config_path)
            else:
                from src.morpher import MorpherEngine
                self.morpher = MorpherEngine(config_path)
                self.interrogator = InterrogatorEngine(config_path)

            # Auto-detect GPU/MPS/CPU
            device = get_device(config_path)
            logger.info(f"Using device: {device}")
            self.consistency_engine = ConsistencyEngine(device=device)
            self.complexity_engine = ComplexityEngine(config_path)
            self.fusion_layer = FusionLayer(config_path)

            logger.info("✓ All modules initialized successfully")

        except Exception as e:
            logger.error(f"Failed to initialize modules: {e}")
            raise

    def detect_single(self, question: str, question_id: str,
                     ground_truth: Optional[str] = None,
                     visualize: bool = True) -> PipelineResult:
        """
        Run full pipeline on a single question.

        Args:
            question: The logic question
            question_id: Unique identifier
            ground_truth: Optional ground truth answer
            visualize: Whether to generate visualizations

        Returns:
            PipelineResult object
        """
        logger.info(f"\n{'='*70}")
        logger.info(f"Processing question: {question_id}")
        logger.info(f"{'='*70}")

        # STEP 1: Generate variants
        logger.info("\n[STEP 1] Generating semantic variants...")
        variants = self.morpher.generate_variants(question, question_id)
        logger.info(f"✓ Generated {len(variants)} variants")

        # STEP 2: Query LLM
        logger.info("\n[STEP 2] Querying LLM...")
        variant_dicts = [
            {'variant_text': v.variant_text, 'transformation_type': v.transformation_type}
            for v in variants
        ]

        responses = self.interrogator.query_variants(
            question, question_id, variant_dicts
        )
        logger.info(f"✓ Collected {len(responses)} responses")

        # STEP 3: Consistency analysis
        logger.info("\n[STEP 3] Analyzing consistency...")
        response_texts = [r.response_text for r in responses]
        response_labels = [r.variant_type for r in responses]

        consistency_analysis = self.consistency_engine.analyze(
            response_texts,
            question_id,
            response_labels,
            visualize=visualize
        )
        logger.info(f"✓ Consistency score: {consistency_analysis.consistency_score:.4f}")

        # STEP 4: Complexity analysis
        logger.info("\n[STEP 4] Analyzing complexity...")
        complexity_analyses = []

        for response in responses:
            analysis = self.complexity_engine.analyze_response(
                response.response_text,
                question_id,
                response.variant_type,
                response.logprobs
            )
            complexity_analyses.append(analysis)

        # Normalize scores
        complexity_analyses = self.complexity_engine.normalize_scores(complexity_analyses)

        # Calculate averages
        avg_entropy = np.mean([
            a.entropy_normalized for a in complexity_analyses
            if a.entropy_normalized is not None
        ])
        avg_ncd = np.mean([
            a.ncd_normalized for a in complexity_analyses
            if a.ncd_normalized is not None
        ])

        logger.info(f"✓ Avg entropy (norm): {avg_entropy:.4f}")
        logger.info(f"✓ Avg NCD (norm): {avg_ncd:.4f}")

        # STEP 5: Fusion & decision
        logger.info("\n[STEP 5] Making final decision...")
        detection = self.fusion_layer.detect(
            question_id,
            "combined",
            consistency_analysis.consistency_score,
            float(avg_entropy),
            float(avg_ncd)
        )

        if visualize:
            self.fusion_layer.visualize_decision(
                detection,
                f"data/processed/results/{question_id}_final_decision.png"
            )

        logger.info(f"\n{'='*70}")
        logger.info(f"RESULT: {'⚠️  HALLUCINATION DETECTED' if detection.is_hallucination else '✓ NO HALLUCINATION'}")
        logger.info(f"Risk: {detection.hallucination_risk:.4f}, Confidence: {detection.confidence}")
        logger.info(f"Explanation: {detection.explanation}")
        logger.info(f"{'='*70}\n")

        # Create result
        return PipelineResult(
            question_id=question_id,
            original_question=question,
            ground_truth=ground_truth,
            num_variants_generated=len(variants),
            num_responses_collected=len(responses),
            consistency_score=consistency_analysis.consistency_score,
            avg_entropy=float(avg_entropy),
            avg_ncd=float(avg_ncd),
            hallucination_detected=detection.is_hallucination,
            hallucination_risk=detection.hallucination_risk,
            confidence=detection.confidence,
            explanation=detection.explanation
        )

    def detect_batch(self, questions: List[Dict],
                    visualize: bool = False) -> List[PipelineResult]:
        """
        Run pipeline on multiple questions.

        Args:
            questions: List of dicts with 'id', 'question', optional 'ground_truth'
            visualize: Whether to generate visualizations

        Returns:
            List of PipelineResult objects
        """
        results = []

        for i, q in enumerate(questions, 1):
            logger.info(f"\n\nProcessing question {i}/{len(questions)}: {q['id']}")

            result = self.detect_single(
                q['question'],
                q['id'],
                q.get('ground_truth'),
                visualize=visualize
            )

            results.append(result)

        return results

    def save_results(self, results: List[PipelineResult], output_path: str):
        """Save results to JSON file"""
        data = [r.to_dict() for r in results]

        with open(output_path, 'w') as f:
            json.dump(data, f, indent=2)

        logger.info(f"\n✓ Saved {len(results)} results to {output_path}")

    def generate_report(self, results: List[PipelineResult]) -> str:
        """Generate summary report"""
        total = len(results)
        hallucinations = sum(1 for r in results if r.hallucination_detected)
        no_hallucinations = total - hallucinations

        avg_risk = np.mean([r.hallucination_risk for r in results])
        avg_consistency = np.mean([r.consistency_score for r in results])

        report = f"""
╔════════════════════════════════════════════════════════════════════╗
║                    LOGIC-HALT DETECTION REPORT                     ║
╚════════════════════════════════════════════════════════════════════╝

SUMMARY:
  Total questions analyzed: {total}
  Hallucinations detected:  {hallucinations} ({hallucinations/total*100:.1f}%)
  No hallucination:         {no_hallucinations} ({no_hallucinations/total*100:.1f}%)

METRICS:
  Average hallucination risk: {avg_risk:.4f}
  Average consistency score:  {avg_consistency:.4f}

BREAKDOWN:
"""

        for r in results:
            status = "⚠️  HALLUCINATION" if r.hallucination_detected else "✓ OK"
            report += f"  [{r.question_id}] {status} (risk: {r.hallucination_risk:.3f}, conf: {r.confidence})\n"

        report += "\n" + "="*70 + "\n"

        return report


# CLI Interface
def main():
    parser = argparse.ArgumentParser(description="LOGIC-HALT Hallucination Detector")
    parser.add_argument('--dataset', type=str, default='data/raw/pilot_dataset.json',
                       help='Path to dataset JSON file')
    parser.add_argument('--output', type=str, default='data/processed/results/detection_results.json',
                       help='Path to save results')
    parser.add_argument('--limit', type=int, default=None,
                       help='Limit number of questions to process')
    parser.add_argument('--visualize', action='store_true',
                       help='Generate visualizations (slower)')
    parser.add_argument('--no-mock', action='store_true',
                       help='Use real API calls instead of mock')

    args = parser.parse_args()

    # Initialize detector
    detector = HallucinationDetector(use_mock=not args.no_mock)

    # Load dataset
    logger.info(f"Loading dataset from {args.dataset}")
    with open(args.dataset, 'r') as f:
        dataset = json.load(f)

    if args.limit:
        dataset = dataset[:args.limit]

    logger.info(f"Processing {len(dataset)} questions...")

    # Run detection
    results = detector.detect_batch(dataset, visualize=args.visualize)

    # Save results
    detector.save_results(results, args.output)

    # Generate report
    report = detector.generate_report(results)
    print(report)

    # Save report
    report_path = args.output.replace('.json', '_report.txt')
    with open(report_path, 'w') as f:
        f.write(report)

    logger.info(f"✓ Report saved to {report_path}")


if __name__ == "__main__":
    main()
