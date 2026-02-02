"""
Demo script for MODULE A: The Morpher
Tests variant generation with mock data (no API keys required)
"""

import sys
import json
from morpher import MorpherEngine, Variant


class MockMorpherEngine(MorpherEngine):
    """
    Mock version of Morpher that doesn't require API keys.
    Uses predefined transformations for testing.
    """

    def __init__(self, config_path: str = "config/config.yaml",
                 prompts_path: str = "config/prompts.yaml"):
        # Initialize parent but skip LLM client
        super().__init__(config_path, prompts_path)
        print("✓ Mock Morpher initialized (no API keys needed)")

    def _call_llm(self, system_prompt: str, user_prompt: str) -> str:
        """
        Mock LLM call - returns predefined transformations based on prompt type
        """
        # Extract the original question from the prompt
        question_start = user_prompt.find("Question: ") + len("Question: ")
        question_end = user_prompt.find("\n", question_start)
        if question_end == -1:
            question_end = len(user_prompt)

        original_question = user_prompt[question_start:question_end].strip()

        # Determine transformation type from prompt
        if "paraphrase" in user_prompt.lower():
            return self._mock_paraphrase(original_question)
        elif "negation" in user_prompt.lower() or "contrapositive" in user_prompt.lower():
            return self._mock_negation(original_question)
        elif "variable" in user_prompt.lower() or "substitut" in user_prompt.lower():
            return self._mock_variable_substitution(original_question)
        elif "reorder" in user_prompt.lower() or "premise" in user_prompt.lower():
            return self._mock_premise_reorder(original_question)
        elif "redundant" in user_prompt.lower() or "context" in user_prompt.lower():
            return self._mock_redundant_context(original_question)
        else:
            return f'{{"variant": "{original_question}", "type": "unchanged"}}'

    def _mock_paraphrase(self, question: str) -> str:
        """Generate a simple paraphrase"""
        # Simple word replacements
        paraphrased = question.replace("All humans", "Every person")
        paraphrased = paraphrased.replace("mortal", "subject to death")
        paraphrased = paraphrased.replace("Is ", "Can we conclude that ")

        return json.dumps({"variant": paraphrased, "type": "paraphrase"})

    def _mock_negation(self, question: str) -> str:
        """Apply double negation"""
        # Add double negation patterns
        if "All" in question and "are" in question:
            negated = question.replace("All", "There are no")
            negated = negated.replace("are mortal", "that are not mortal")
        else:
            negated = f"It is not false that {question.lower()}"

        return json.dumps({"variant": negated, "type": "negation"})

    def _mock_variable_substitution(self, question: str) -> str:
        """Substitute entity names"""
        substituted = question.replace("humans", "H-type entities")
        substituted = substituted.replace("Socrates", "Entity-S")
        substituted = substituted.replace("mortal", "property-M")

        return json.dumps({"variant": substituted, "type": "variable_substitution"})

    def _mock_premise_reorder(self, question: str) -> str:
        """Reorder premises"""
        # Split by periods, reverse order
        sentences = [s.strip() for s in question.split('.') if s.strip()]
        if len(sentences) > 1:
            # Keep the question at the end, reverse premises
            if '?' in sentences[-1]:
                reordered = sentences[-1] + ' Given that: ' + '. '.join(reversed(sentences[:-1])) + '.'
            else:
                reordered = '. '.join(reversed(sentences)) + '.'
        else:
            reordered = question

        return json.dumps({"variant": reordered, "type": "premise_reordering"})

    def _mock_redundant_context(self, question: str) -> str:
        """Add irrelevant context"""
        context = "In the context of classical logic, which has been studied for centuries, "
        contextualized = context + question.lower()

        return json.dumps({"variant": contextualized, "type": "redundant_context"})


def demo_single_question():
    """Test Morpher on a single question"""
    print("="*70)
    print("DEMO 1: Single Question Variant Generation")
    print("="*70)

    morpher = MockMorpherEngine()

    question = "All humans are mortal. Socrates is a human. Is Socrates mortal?"
    print(f"\n📝 Original Question:\n{question}\n")

    variants = morpher.generate_variants(question, "demo_001", num_variants=5)

    print(f"✓ Generated {len(variants)} valid variants:\n")

    for i, variant in enumerate(variants, 1):
        print(f"\n{i}. TYPE: {variant.transformation_type.upper()}")
        print(f"   SIMILARITY: {variant.similarity_score:.4f}")
        print(f"   VALID: {'✓' if variant.is_valid else '✗'}")
        print(f"   TEXT: {variant.variant_text}")


def demo_batch_processing():
    """Test Morpher on multiple questions from dataset"""
    print("\n" + "="*70)
    print("DEMO 2: Batch Processing (First 3 questions from dataset)")
    print("="*70)

    morpher = MockMorpherEngine()

    # Load pilot dataset
    with open('data/raw/pilot_dataset.json', 'r') as f:
        dataset = json.load(f)

    # Process first 3 questions
    sample_questions = [
        {"id": q["id"], "question": q["question"]}
        for q in dataset[:3]
    ]

    results = morpher.batch_generate(sample_questions)

    for question_id, variants in results.items():
        original = next(q for q in dataset if q['id'] == question_id)
        print(f"\n📌 {question_id}: {original['question'][:60]}...")
        print(f"   Generated {len(variants)} variants")

        for v in variants[:2]:  # Show first 2 variants only
            print(f"   → [{v.transformation_type}] {v.variant_text[:80]}...")


def demo_similarity_filtering():
    """Demonstrate similarity filtering"""
    print("\n" + "="*70)
    print("DEMO 3: Similarity Filtering")
    print("="*70)

    morpher = MockMorpherEngine()

    # Test with two texts
    text1 = "All humans are mortal. Socrates is a human."
    text2 = "Every person is subject to death. Socrates is a person."
    text3 = "The sky is blue and grass is green."

    sim_high = morpher._calculate_similarity(text1, text2)
    sim_low = morpher._calculate_similarity(text1, text3)

    print(f"\n📊 Similarity Scores:")
    print(f"   Similar texts: {sim_high:.4f} {'✓ PASS' if sim_high > 0.85 else '✗ FAIL'}")
    print(f"   Different texts: {sim_low:.4f} {'✓ PASS' if sim_low < 0.85 else '✗ FAIL'}")
    print(f"\n   Threshold: {morpher.config['similarity_threshold']}")


def main():
    """Run all demos"""
    try:
        demo_single_question()
        demo_batch_processing()
        demo_similarity_filtering()

        print("\n" + "="*70)
        print("✓ All demos completed successfully!")
        print("="*70)

    except Exception as e:
        print(f"\n❌ Error during demo: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
