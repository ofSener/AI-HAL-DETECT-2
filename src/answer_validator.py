"""
MODULE F: Answer Validator - Majority Voting based Answer Consistency
Extracts final answers from responses and detects outliers using majority voting.

This module addresses the limitation where semantically similar but factually incorrect
responses can have high consistency scores. By extracting and comparing final answers,
we can identify responses that disagree with the majority.
"""

import re
import logging
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
from collections import Counter
import yaml

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class AnswerValidation:
    """Result of answer validation for a single response"""
    response_id: str
    extracted_answer: Optional[str]
    is_majority: bool
    majority_answer: Optional[str]
    majority_ratio: float  # What percentage agrees with majority
    minority_penalty: float  # 0.0 if majority, penalty value if minority
    confidence: str  # 'high', 'medium', 'low' based on majority agreement


@dataclass
class ValidationSummary:
    """Summary of answer validation across all responses"""
    total_responses: int
    extracted_count: int
    majority_answer: Optional[str]
    majority_count: int
    majority_ratio: float
    unique_answers: int
    outlier_indices: List[int]
    answer_distribution: Dict[str, int]


class AnswerValidator:
    """
    Validates responses by extracting final answers and performing majority voting.

    This helps detect factually incorrect responses that might appear consistent
    in terms of language but give different final answers.
    """

    def __init__(self, config_path: str = "config/config.yaml"):
        """
        Initialize Answer Validator.

        Args:
            config_path: Path to config file
        """
        # Load configuration
        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)

            # Get validator config or use defaults
            self.config = config.get('answer_validator', {})
        except Exception as e:
            logger.warning(f"Could not load config: {e}. Using defaults.")
            self.config = {}

        # Minority penalty (added to risk score for minority answers)
        self.minority_penalty = self.config.get('minority_penalty', 0.30)

        # Minimum responses needed for majority voting
        self.min_responses = self.config.get('min_responses', 3)

        # Confidence thresholds based on majority ratio
        self.high_confidence_threshold = self.config.get('high_confidence_threshold', 0.8)
        self.low_confidence_threshold = self.config.get('low_confidence_threshold', 0.5)

        logger.info(f"Answer Validator initialized (penalty={self.minority_penalty}, min_responses={self.min_responses})")

    def extract_answer(self, response_text: str) -> Optional[str]:
        """
        Extract the final answer from a response text.

        Looks for common patterns:
        - "answer is X"
        - "result is X"
        - "= X" at the end
        - Final number or short phrase
        - Turkish patterns: "sonuç X", "cevap X", "fark X"

        Args:
            response_text: The full response text

        Returns:
            Extracted answer string, or None if not found
        """
        if not response_text:
            return None

        text = response_text.lower().strip()

        # Pattern list (order matters - more specific first)
        patterns = [
            # Turkish patterns - Currency/Unit specific (highest priority)
            r"toplamda\s+(\d+)\s*kuruş",
            r"toplam[:\s]+(\d+)\s*kuruş",
            r"(\d+)\s*kuruş\s*(?:eder|yapar|tutar|olur)",
            r"(\d+)\s*kuruş(?:tur|tür|tir|tır)",

            # Turkish patterns - Result expressions
            r"sonuç(?:\s+olarak)?[:\s]+['\"]?(\d+)['\"]?",
            r"fark[:\s]+(\d+)",
            r"fark\s+(\d+)",
            r"cevap[:\s]+['\"]?(\d+)['\"]?",
            r"sonuç[:\s]+['\"]?(\d+)['\"]?",
            r"(\d+)['\"]?\s*(?:\'?dir|dır|dur|dür|tir|tır|tur|tür)(?:\s|$|\.)",

            # English patterns
            r"(?:the\s+)?answer\s+is[:\s]+['\"]?(\d+)['\"]?",
            r"(?:the\s+)?result\s+is[:\s]+['\"]?(\d+)['\"]?",
            r"(?:the\s+)?difference\s+is[:\s]+['\"]?(\d+)['\"]?",
            r"(?:total|sum)\s*(?:is|=|:)\s*(\d+)",
            r"=\s*['\"]?(\d+)['\"]?\s*$",

            # Boolean/Yes-No patterns (Turkish & English)
            r"(evet|hayır|yes|no|true|false)\s*[.!]?\s*$",

            # Final number in last sentence
            r"[.!?]\s*[^.!?]*?(\d+)[^.!?]*$",
        ]

        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE | re.MULTILINE)
            if match:
                # Return the captured group or the full match
                answer = match.group(1) if match.groups() else match.group(0)
                return answer.strip().lower()

        # Fallback: Look for the last number in the text
        numbers = re.findall(r'\b(\d+)\b', text)
        if numbers:
            # Return the last number (likely the final answer)
            return numbers[-1]

        return None

    def extract_answers_batch(self, responses: List[str]) -> List[Optional[str]]:
        """
        Extract answers from multiple responses.

        Args:
            responses: List of response texts

        Returns:
            List of extracted answers (None for failed extractions)
        """
        return [self.extract_answer(r) for r in responses]

    def majority_vote(self, answers: List[Optional[str]]) -> Tuple[Optional[str], float, Dict[str, int]]:
        """
        Perform majority voting on extracted answers.

        Args:
            answers: List of extracted answers

        Returns:
            Tuple of (majority_answer, majority_ratio, answer_distribution)
        """
        # Filter out None values
        valid_answers = [a for a in answers if a is not None]

        if not valid_answers:
            return None, 0.0, {}

        # Count occurrences
        counter = Counter(valid_answers)
        distribution = dict(counter)

        # Get most common
        most_common = counter.most_common(1)[0]
        majority_answer = most_common[0]
        majority_count = most_common[1]

        # Calculate ratio (against total responses, not just valid ones)
        majority_ratio = majority_count / len(answers) if answers else 0.0

        return majority_answer, majority_ratio, distribution

    def validate_responses(self, responses: List[str],
                          response_ids: Optional[List[str]] = None) -> Tuple[List[AnswerValidation], ValidationSummary]:
        """
        Validate all responses and identify outliers.

        Args:
            responses: List of response texts
            response_ids: Optional list of response identifiers

        Returns:
            Tuple of (list of AnswerValidation, ValidationSummary)
        """
        if response_ids is None:
            response_ids = [f"response_{i+1}" for i in range(len(responses))]

        # Extract all answers
        answers = self.extract_answers_batch(responses)

        # Perform majority voting
        majority_answer, majority_ratio, distribution = self.majority_vote(answers)

        # Determine confidence level based on majority agreement
        if majority_ratio >= self.high_confidence_threshold:
            overall_confidence = 'high'
        elif majority_ratio >= self.low_confidence_threshold:
            overall_confidence = 'medium'
        else:
            overall_confidence = 'low'

        # Validate each response
        validations = []
        outlier_indices = []

        for i, (response_id, answer) in enumerate(zip(response_ids, answers)):
            # Check if this answer matches majority
            is_majority = (answer == majority_answer) if answer is not None else False

            # Calculate penalty
            if answer is None:
                # Could not extract answer - apply partial penalty
                minority_penalty = self.minority_penalty * 0.5
                is_majority = False
            elif is_majority:
                minority_penalty = 0.0
            else:
                minority_penalty = self.minority_penalty
                outlier_indices.append(i)

            # Determine confidence for this response
            if is_majority and majority_ratio >= self.high_confidence_threshold:
                confidence = 'high'
            elif is_majority:
                confidence = 'medium'
            else:
                confidence = 'low'

            validation = AnswerValidation(
                response_id=response_id,
                extracted_answer=answer,
                is_majority=is_majority,
                majority_answer=majority_answer,
                majority_ratio=majority_ratio,
                minority_penalty=minority_penalty,
                confidence=confidence
            )
            validations.append(validation)

        # Create summary
        summary = ValidationSummary(
            total_responses=len(responses),
            extracted_count=sum(1 for a in answers if a is not None),
            majority_answer=majority_answer,
            majority_count=distribution.get(majority_answer, 0) if majority_answer else 0,
            majority_ratio=majority_ratio,
            unique_answers=len(distribution),
            outlier_indices=outlier_indices,
            answer_distribution=distribution
        )

        # Log results
        logger.info(f"Answer Validation: {summary.extracted_count}/{summary.total_responses} answers extracted")
        logger.info(f"Majority answer: '{majority_answer}' ({summary.majority_ratio*100:.1f}% agreement)")
        if outlier_indices:
            logger.info(f"Outliers detected at indices: {outlier_indices}")

        return validations, summary

    def get_adjusted_risk(self, base_risk: float, minority_penalty: float) -> float:
        """
        Adjust risk score based on minority penalty.

        Args:
            base_risk: Original risk score from fusion layer
            minority_penalty: Penalty from answer validation

        Returns:
            Adjusted risk score (clamped to [0, 1])
        """
        adjusted = base_risk + minority_penalty
        return min(max(adjusted, 0.0), 1.0)


# Demo/Test
if __name__ == "__main__":
    print("=" * 70)
    print("MODULE F: Answer Validator - Demo")
    print("=" * 70)

    validator = AnswerValidator()

    # Test case: Multiple responses with different final answers
    test_responses = [
        """Öncelikle sepetteki yumurta sayısını hesaplayalım:
        32 × 4 = 128
        128 ÷ 16 = 8
        Başlangıçta 8 yumurta var. 12 eklendi = 20.
        Elimde 19 yumurta var. Fark: 19 - 8 = 11
        Sonuç olarak fark 11'dir.""",

        """32*4/16 = 8 yumurta başlangıçta.
        12 ekleniyor = 20 yumurta.
        3-2 = 1 yumurta alıyorum.
        Fark: 8 - 1 = 7
        Sonuç olarak fark 7'dir.""",

        """Hesaplama: 32×4÷16 = 8
        Başlangıçta 20 yumurta vardı.
        Elimde 19 var.
        Fark = 20 - 19 = 1
        Sonuç 1'dir.""",

        """32*4/16 = 8 yumurta.
        Sepete 12 ekleniyor, toplam 20.
        3-2 = 1 yumurta aldım.
        Başlangıçtaki 8 ile elimdeki 1 arasındaki fark:
        8 - 1 = 7
        Sonuç: 7""",

        """İlk hesap: 32*4/16 = 8
        12 eklenince 20 oldu ama düştü.
        Ben 3-2=1 yumurta aldım.
        Fark = |1 - 8| = 7
        Cevap 7'dir."""
    ]

    print("\n[TEST] Validating 5 responses with different answers...")
    validations, summary = validator.validate_responses(test_responses)

    print(f"\n--- Summary ---")
    print(f"Total responses: {summary.total_responses}")
    print(f"Answers extracted: {summary.extracted_count}")
    print(f"Unique answers: {summary.unique_answers}")
    print(f"Majority answer: {summary.majority_answer} ({summary.majority_ratio*100:.1f}% agreement)")
    print(f"Answer distribution: {summary.answer_distribution}")
    print(f"Outlier indices: {summary.outlier_indices}")

    print(f"\n--- Individual Validations ---")
    for v in validations:
        status = "MAJORITY" if v.is_majority else "OUTLIER"
        penalty = f"+{v.minority_penalty*100:.0f}%" if v.minority_penalty > 0 else "none"
        print(f"  [{v.response_id}] Answer: {v.extracted_answer} | {status} | Penalty: {penalty}")

    # Test risk adjustment
    print(f"\n--- Risk Adjustment Example ---")
    base_risk = 0.35
    for v in validations:
        adjusted = validator.get_adjusted_risk(base_risk, v.minority_penalty)
        print(f"  [{v.response_id}] Base: {base_risk*100:.1f}% → Adjusted: {adjusted*100:.1f}%")

    print("\n" + "=" * 70)
    print("Answer Validator demo complete!")
    print("=" * 70)
