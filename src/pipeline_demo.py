"""
Integrated Pipeline Demo: Morpher → Interrogator
Shows how MODULE A and MODULE B work together
"""

import json
import sys
from morpher_demo import MockMorpherEngine
from interrogator import InterrogatorEngine, LLMResponse


class MockInterrogatorEngine(InterrogatorEngine):
    """
    Mock version of Interrogator for testing without API keys.
    Generates simulated responses based on question type.
    """

    def __init__(self, config_path: str = "config/config.yaml",
                 prompts_path: str = "config/prompts.yaml"):
        super().__init__(config_path, prompts_path)
        print("✓ Mock Interrogator initialized (no API keys needed)")

    def _query_openai(self, question: str, question_id: str,
                     variant_type: str, model_name: str) -> LLMResponse:
        """Mock OpenAI query - generates simulated responses"""
        import time
        import random

        # Simulate different responses based on question
        if "mortal" in question.lower():
            response_text = self._generate_mortal_response(question, variant_type)
        elif "fish" in question.lower() or "whale" in question.lower():
            response_text = self._generate_whale_response(question, variant_type)
        elif "penguin" in question.lower():
            response_text = self._generate_penguin_response(question, variant_type)
        else:
            response_text = f"Based on the logical structure, the answer is: Yes, this follows from the premises."

        # Generate fake logprobs (simulating uncertainty)
        num_tokens = len(response_text.split())
        fake_logprobs = [-random.uniform(0.1, 2.0) for _ in range(num_tokens)]
        fake_tokens = response_text.split()

        return LLMResponse(
            question_id=question_id,
            variant_type=variant_type,
            question_text=question,
            response_text=response_text,
            model_name=model_name,
            logprobs=fake_logprobs,
            tokens=fake_tokens,
            timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
            metadata={
                "finish_reason": "stop",
                "usage": {"prompt_tokens": 50, "completion_tokens": num_tokens, "total_tokens": 50 + num_tokens}
            }
        )

    def _generate_mortal_response(self, question: str, variant_type: str) -> str:
        """Generate response for mortal/Socrates questions"""
        responses = {
            "original": "Yes, Socrates is mortal. Since all humans are mortal and Socrates is a human, it logically follows that Socrates must be mortal.",
            "paraphrase": "Affirmative. Given that every person is subject to death and Socrates is a person, we can conclude Socrates is mortal.",
            "negation": "Yes. The statement 'there are no humans that are not mortal' is equivalent to 'all humans are mortal', so Socrates being human means he is mortal.",
            "premise_reordering": "Yes, Socrates is mortal. The order of premises doesn't affect the validity: Socrates is human, and all humans are mortal.",
            "redundant_context": "Yes. Regardless of the historical context, the logical structure is sound: all humans are mortal, Socrates is human, therefore Socrates is mortal.",
            "variable_substitution": "Yes, Entity-S has property-M. All H-type entities have property-M, and Entity-S is an H-type entity."
        }
        return responses.get(variant_type, responses["original"])

    def _generate_whale_response(self, question: str, variant_type: str) -> str:
        """Generate response for whale/fish questions"""
        responses = {
            "original": "No, whales are not fish. All whales are mammals, and no fish are mammals, therefore whales cannot be fish.",
            "paraphrase": "No. Since the sets of fish and mammals are disjoint, and whales are mammals, whales cannot be fish.",
            "negation": "No. The premise states no fish are mammals, and all whales are mammals, so whales are definitively not fish."
        }
        return responses.get(variant_type, responses["original"])

    def _generate_penguin_response(self, question: str, variant_type: str) -> str:
        """Generate response for penguin questions - HALLUCINATION TEST"""
        import random

        # Simulate inconsistent responses (hallucination)
        if variant_type == "original":
            return "Yes, penguins can fly since all birds can fly and penguins are birds."
        elif variant_type == "paraphrase":
            return "Actually, while the premise states all birds can fly, this is factually incorrect. Penguins are flightless birds."
        elif variant_type == "negation":
            return "The logical structure suggests yes, but empirically penguins cannot fly."
        else:
            # Random inconsistency
            if random.random() > 0.5:
                return "Yes, based on the syllogism, penguins can fly."
            else:
                return "No, penguins are an exception to the premise that all birds fly."

    def _query_anthropic(self, question: str, question_id: str,
                        variant_type: str, model_name: str) -> LLMResponse:
        """Mock Anthropic query"""
        # Reuse OpenAI mock
        response = self._query_openai(question, question_id, variant_type, model_name)
        response.model_name = model_name
        response.logprobs = None  # Claude doesn't provide logprobs
        return response


def demo_integrated_pipeline():
    """
    Full pipeline demo: Dataset → Morpher → Interrogator → Results
    """
    print("="*70)
    print("INTEGRATED PIPELINE DEMO: Morpher + Interrogator")
    print("="*70)

    # Step 1: Initialize modules
    print("\n[STEP 1] Initializing modules...")
    morpher = MockMorpherEngine()
    interrogator = MockInterrogatorEngine()

    # Step 2: Load a test question
    print("\n[STEP 2] Loading test question...")
    with open('data/raw/pilot_dataset.json', 'r') as f:
        dataset = json.load(f)

    test_question = dataset[0]  # "All humans are mortal..."
    print(f"   Question ID: {test_question['id']}")
    print(f"   Question: {test_question['question'][:60]}...")

    # Step 3: Generate variants
    print("\n[STEP 3] Generating variants with Morpher...")
    variants = morpher.generate_variants(
        test_question['question'],
        test_question['id'],
        num_variants=3
    )
    print(f"   ✓ Generated {len(variants)} valid variants")

    # Step 4: Query LLM with all variants
    print("\n[STEP 4] Querying LLM with Interrogator...")
    variant_dicts = [
        {
            'variant_text': v.variant_text,
            'transformation_type': v.transformation_type
        }
        for v in variants
    ]

    responses = interrogator.query_variants(
        test_question['question'],
        test_question['id'],
        variant_dicts,
        model_name="gpt-3.5-turbo"
    )

    print(f"   ✓ Collected {len(responses)} responses")

    # Step 5: Display results
    print("\n[STEP 5] Results:")
    print("-" * 70)

    for i, response in enumerate(responses):
        print(f"\n{i+1}. [{response.variant_type.upper()}]")
        print(f"   Question: {response.question_text[:50]}...")
        print(f"   Response: {response.response_text[:100]}...")

        if response.logprobs:
            avg_logprob = sum(response.logprobs) / len(response.logprobs)
            print(f"   Avg Log Prob: {avg_logprob:.4f}")

    # Step 6: Save to file
    print("\n[STEP 6] Saving results...")
    output_path = "data/processed/results/demo_pipeline_output.json"
    interrogator.save_responses(responses, output_path)
    print(f"   ✓ Saved to {output_path}")


def demo_hallucination_detection():
    """
    Demo showing how inconsistent responses are detected
    """
    print("\n" + "="*70)
    print("HALLUCINATION DETECTION DEMO")
    print("="*70)

    morpher = MockMorpherEngine()
    interrogator = MockInterrogatorEngine()

    # Use the penguin question (has faulty premise)
    with open('data/raw/pilot_dataset.json', 'r') as f:
        dataset = json.load(f)

    penguin_q = dataset[2]  # "All birds can fly. Penguins are birds..."

    print(f"\n📝 Testing: {penguin_q['question']}")
    print(f"   (Note: This has a faulty premise - not all birds fly)")

    # Generate variants
    variants = morpher.generate_variants(
        penguin_q['question'],
        penguin_q['id'],
        num_variants=3
    )

    variant_dicts = [
        {'variant_text': v.variant_text, 'transformation_type': v.transformation_type}
        for v in variants
    ]

    # Get responses
    responses = interrogator.query_variants(
        penguin_q['question'],
        penguin_q['id'],
        variant_dicts,
        model_name="gpt-3.5-turbo"
    )

    # Analyze consistency
    print("\n🔍 Analyzing response consistency:")
    print("-" * 70)

    answers = []
    for response in responses:
        # Simple answer extraction
        answer = "YES" if "yes" in response.response_text.lower()[:20] else "NO"
        answers.append(answer)

        print(f"\n[{response.variant_type}]: {answer}")
        print(f"   {response.response_text[:80]}...")

    # Check consistency
    unique_answers = set(answers)
    if len(unique_answers) > 1:
        print("\n⚠️  INCONSISTENCY DETECTED!")
        print(f"   Found {len(unique_answers)} different answers: {unique_answers}")
        print("   This indicates potential hallucination or logical confusion.")
    else:
        print("\n✓ Responses are consistent")


def main():
    """Run all demos"""
    try:
        demo_integrated_pipeline()
        demo_hallucination_detection()

        print("\n" + "="*70)
        print("✓ Pipeline demo completed successfully!")
        print("="*70)
        print("\nNext steps:")
        print("  - MODULE C: Consistency Engine (NLI + Graph)")
        print("  - MODULE D: Complexity Engine (Entropy + NCD)")
        print("  - MODULE E: Fusion Layer")

    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
