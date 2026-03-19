"""
Bayesian Optimization using Batch API Results
API çağrısı YOK - sadece GPU kullanır - ÇOK HIZLI!
"""

import json
import re
import sys
import gc
import optuna
import numpy as np
import torch
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple
from dataclasses import dataclass
from tqdm import tqdm

sys.path.append(str(Path(__file__).parent.parent))

from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
from sklearn.model_selection import KFold

# ========================================
# PATHS
# ========================================
BASE_DIR = Path(__file__).parent.parent
BATCH_RESULTS_DIR = BASE_DIR / 'data' / 'batch_results'
DATASET_PATH = BASE_DIR / 'data' / 'raw' / 'truthfulqa_dataset.json'
OUTPUT_DIR = BASE_DIR / 'config' / 'optimization_results'

# Batch result files (update these with your actual batch IDs)
VARIANTS_FILE = BATCH_RESULTS_DIR / 'batch_batch_697f1cb8f0fc819080718b3eba0544c2_organized.json'
ORIGINAL_RESPONSES_FILE = BATCH_RESULTS_DIR / 'batch_batch_697f1cc0055c81908d8b8c3d78c4cb80_organized.json'
VARIANT_RESPONSES_FILE = BATCH_RESULTS_DIR / 'batch_batch_697f2470463081908e5842fb09956d58_organized.json'


# ========================================
# CUDA SETUP
# ========================================
def setup_cuda():
    if not torch.cuda.is_available():
        print("[WARN]  CUDA not available, using CPU")
        return 'cpu'

    gpu_name = torch.cuda.get_device_name(0)
    gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"\n[GPU] {gpu_name}")
    print(f"   VRAM: {gpu_mem:.1f} GB")
    print(f"   CUDA: {torch.version.cuda}")

    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.cuda.empty_cache()

    return 'cuda'

DEVICE = setup_cuda()


# ========================================
# DATA LOADING
# ========================================
def extract_variant_text(raw_text):
    """Extract variant text from JSON response"""
    text = re.sub(r'```json\s*', '', raw_text)
    text = re.sub(r'```\s*', '', text)
    text = text.strip()
    try:
        data = json.loads(text)
        return data.get('variant', text)
    except:
        return text


def load_batch_data():
    """Load all batch results"""
    print("\n[*] Loading batch data...")

    # Load variants
    with open(VARIANTS_FILE, 'r', encoding='utf-8') as f:
        variants_data = json.load(f)
    print(f"   Variants: {len(variants_data.get('variants', {}))} questions")

    # Load original responses
    with open(ORIGINAL_RESPONSES_FILE, 'r', encoding='utf-8') as f:
        original_responses = json.load(f)
    print(f"   Original responses: {len(original_responses.get('responses', {}))} questions")

    # Load variant responses
    with open(VARIANT_RESPONSES_FILE, 'r', encoding='utf-8') as f:
        variant_responses = json.load(f)
    print(f"   Variant responses: {len(variant_responses.get('responses', {}))} questions")

    # Load dataset for ground truths
    with open(DATASET_PATH, 'r', encoding='utf-8') as f:
        dataset = json.load(f)

    questions = dataset if isinstance(dataset, list) else dataset.get('questions', [])
    ground_truths = {q.get('id', q.get('question_id')): q.get('ground_truth', '')
                     for q in questions}
    print(f"   Ground truths: {len(ground_truths)} questions")

    return variants_data, original_responses, variant_responses, ground_truths


def prepare_responses_for_question(q_id, original_responses, variant_responses):
    """Combine original and variant responses for a question"""
    responses = []

    # Original response
    orig = original_responses.get('responses', {}).get(q_id, {}).get('original', '')
    if orig:
        responses.append(('original', orig))

    # Variant responses
    var_resps = variant_responses.get('responses', {}).get(q_id, {})
    for var_type, resp in var_resps.items():
        if resp:
            responses.append((var_type, resp))

    return responses


# ========================================
# ANALYSIS MODULES (GPU)
# ========================================
class ConsistencyAnalyzer:
    """NLI-based consistency analysis using GPU with BATCH INFERENCE"""

    def __init__(self, device='cuda', batch_size=64):
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        print("\n[*] Loading NLI model (LARGE)...")
        model_name = "cross-encoder/nli-deberta-v3-large"  # 435M params - daha güçlü!
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.model.to(device)
        self.model.eval()
        self.device = device
        self.batch_size = batch_size  # GPU batch size for NLI

        # Enable FP16 for faster inference and less memory per sample (allows bigger batches)
        if device == 'cuda':
            self.model = self.model.half()  # FP16

        print(f"   [OK] NLI model loaded on {device} (batch_size={batch_size}, FP16=True)")

    def extract_answer(self, response_text):
        """Extract ANSWER part from response"""
        if 'ANSWER:' in response_text:
            parts = response_text.split('ANSWER:')
            if len(parts) > 1:
                answer_part = parts[1].split('REASONING:')[0] if 'REASONING:' in parts[1] else parts[1]
                return answer_part.strip()
        return response_text.strip()

    @torch.no_grad()
    def analyze(self, responses: List[Tuple[str, str]], min_edge_weight=0.3,
                contradiction_weight=0.9, neutral_weight=0.4) -> float:
        """Analyze consistency between responses using BATCH inference"""
        if len(responses) < 2:
            return 1.0

        # Extract answers
        answers = [self.extract_answer(r[1]) for r in responses]
        n = len(answers)

        # Collect ALL pairs first
        pairs_a = []
        pairs_b = []
        for i in range(n):
            for j in range(i + 1, n):
                pairs_a.append(answers[i])
                pairs_b.append(answers[j])

        if not pairs_a:
            return 1.0

        # BATCH INFERENCE - process all pairs at once (or in large batches)
        all_scores = []

        for batch_start in range(0, len(pairs_a), self.batch_size):
            batch_end = min(batch_start + self.batch_size, len(pairs_a))
            batch_a = pairs_a[batch_start:batch_end]
            batch_b = pairs_b[batch_start:batch_end]

            inputs = self.tokenizer(
                batch_a, batch_b,
                return_tensors='pt',
                truncation=True,
                max_length=512,
                padding=True
            ).to(self.device)

            outputs = self.model(**inputs)
            probs = torch.softmax(outputs.logits, dim=-1)

            # Labels: 0=contradiction, 1=neutral, 2=entailment
            contradiction_probs = probs[:, 0]
            neutral_probs = probs[:, 1]

            scores = contradiction_probs * contradiction_weight + neutral_probs * neutral_weight
            all_scores.extend(scores.cpu().tolist())

        # Count contradictions
        contradictions = sum(1 for s in all_scores if s > min_edge_weight)
        total_pairs = len(all_scores)

        consistency = 1.0 - (contradictions / total_pairs)
        return consistency


class ComplexityAnalyzer:
    """Entropy and NCD-based complexity analysis"""

    def __init__(self):
        import zlib
        self.zlib = zlib

    def token_entropy(self, text: str) -> float:
        """Calculate token-level entropy"""
        tokens = text.split()
        if not tokens:
            return 0.0

        from collections import Counter
        counts = Counter(tokens)
        total = len(tokens)
        probs = [c / total for c in counts.values()]
        entropy = -sum(p * np.log2(p) for p in probs if p > 0)
        return entropy

    def ncd(self, text: str) -> float:
        """Normalized Compression Distance"""
        if not text:
            return 0.0

        original = text.encode('utf-8')
        compressed = self.zlib.compress(original)
        return len(compressed) / len(original)

    def calculate_pairwise_ncd(self, responses):
        """Average pairwise NCD across all response pairs."""
        n = len(responses)
        if n < 2:
            return 0.0
        ncds = []
        for i in range(n):
            for j in range(i + 1, n):
                text1 = responses[i].encode('utf-8')
                text2 = responses[j].encode('utf-8')
                c_x = len(self.zlib.compress(text1))
                c_y = len(self.zlib.compress(text2))
                c_xy = len(self.zlib.compress(text1 + text2))
                denom = max(c_x, c_y)
                ncd_val = (c_xy - min(c_x, c_y)) / denom if denom > 0 else 0.0
                ncds.append(ncd_val)
        return float(np.mean(ncds))

    def analyze(self, responses: List[Tuple[str, str]], entropy_max=10.0, ncd_max=1.0) -> Tuple[float, float]:
        """Analyze complexity of responses"""
        if not responses:
            return 0.0, 0.0

        entropies = []
        ncds = []

        for _, resp in responses:
            entropies.append(self.token_entropy(resp))
            ncds.append(self.ncd(resp))

        avg_entropy = np.mean(entropies) / entropy_max
        avg_ncd = np.mean(ncds) / ncd_max

        return min(avg_entropy, 1.0), min(avg_ncd, 1.0)


# ========================================
# FUSION & DECISION
# ========================================
def calculate_risk(self_verification: float, inconsistency: float, entropy: float,
                   ncd: float, minority_penalty: float,
                   alpha: float, beta: float, gamma: float,
                   delta: float, epsilon: float) -> float:
    """5-signal fusion: Risk = a*SV + b*Incon + g*Entropy + d*NCD + e*MinPenalty"""
    risk = (alpha * self_verification +
            beta * inconsistency +
            gamma * entropy +
            delta * ncd +
            epsilon * minority_penalty)
    return min(max(risk, 0.0), 1.0)


# ========================================
# OPTIMIZER
# ========================================
class BatchOptimizer:
    """Bayesian optimization using cached batch data with GPU BATCH INFERENCE"""

    def __init__(self, nli_batch_size=64):
        """
        Args:
            nli_batch_size: Batch size for NLI inference (higher = more GPU usage)
                           RTX 5070 12GB can handle 64-128 easily
        """
        self.variants_data, self.original_responses, self.variant_responses, self.ground_truths = load_batch_data()

        # Get question IDs that have all data
        self.question_ids = list(set(self.original_responses.get('responses', {}).keys()) &
                                  set(self.variant_responses.get('responses', {}).keys()) &
                                  set(self.ground_truths.keys()))

        print(f"\n[*] Questions with complete data: {len(self.question_ids)}")

        # Initialize analyzers with batch size
        self.consistency_analyzer = ConsistencyAnalyzer(DEVICE, batch_size=nli_batch_size)
        self.complexity_analyzer = ComplexityAnalyzer()

        # Cache for computed features
        self.feature_cache = {}

        # Compute dynamic labels by comparing LLM responses to ground truth
        print("\n[*] Computing labels from LLM responses vs ground truth...")
        self._compute_labels()

        # PRE-COMPUTE new 5-signal features
        self._precompute_self_verification_scores()
        self._precompute_minority_penalties()
        self._precompute_pairwise_ncd()

        # PRE-COMPUTE all NLI pair scores (run NLI ONCE, apply thresholds per trial)
        self._precompute_nli_scores()

    def _compute_labels(self):
        """
        Compute labels using NLI ENTAILMENT - much better than cosine similarity!
        Uses a SEPARATE NLI model to avoid data leakage with feature computation.

        Logic:
        - If ground_truth ENTAILS llm_answer → Truthful (0)
        - If ground_truth CONTRADICTS llm_answer → Hallucination (1)
        - If NEUTRAL and low entailment → Hallucination (1)
        """
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        print("   Loading SEPARATE NLI model for labels (microsoft/deberta-large-mnli)...")
        # Use DIFFERENT model than feature extraction to avoid data leakage
        label_model_name = "microsoft/deberta-large-mnli"
        label_tokenizer = AutoTokenizer.from_pretrained(label_model_name)
        label_model = AutoModelForSequenceClassification.from_pretrained(label_model_name)
        label_model.to(DEVICE)
        label_model.eval()

        if DEVICE == 'cuda':
            label_model = label_model.half()  # FP16

        self.labels = {}
        entailment_scores = []

        # Collect all pairs for batch processing
        all_answers = []
        all_gts = []
        valid_qids = []

        for q_id in self.question_ids:
            gt = self.ground_truths.get(q_id, '')
            if not gt:
                self.labels[q_id] = 0
                continue

            orig_resp = self.original_responses.get('responses', {}).get(q_id, {}).get('original', '')
            if not orig_resp:
                var_resps = self.variant_responses.get('responses', {}).get(q_id, {})
                if var_resps:
                    orig_resp = list(var_resps.values())[0]

            if not orig_resp:
                self.labels[q_id] = 0
                continue

            answer = self.consistency_analyzer.extract_answer(orig_resp)
            all_answers.append(answer)
            all_gts.append(gt)
            valid_qids.append(q_id)

        # BATCH NLI inference for labels
        batch_size = 32  # Smaller batch for label model

        with torch.no_grad():
            for batch_start in tqdm(range(0, len(all_answers), batch_size), desc="NLI Label Computation"):
                batch_end = min(batch_start + batch_size, len(all_answers))
                batch_answers = all_answers[batch_start:batch_end]
                batch_gts = all_gts[batch_start:batch_end]
                batch_qids = valid_qids[batch_start:batch_end]

                # NLI: premise=answer, hypothesis=ground_truth
                # "Does the answer entail the ground truth?"
                inputs = label_tokenizer(
                    batch_answers, batch_gts,
                    return_tensors='pt',
                    truncation=True,
                    max_length=512,
                    padding=True
                ).to(DEVICE)

                outputs = label_model(**inputs)
                probs = torch.softmax(outputs.logits.float(), dim=-1)

                # deberta-large-mnli: 0=contradiction, 1=neutral, 2=entailment
                for i, q_id in enumerate(batch_qids):
                    contradiction_prob = probs[i, 0].item()
                    neutral_prob = probs[i, 1].item()
                    entailment_prob = probs[i, 2].item()

                    entailment_scores.append(entailment_prob)

                    # Label logic:
                    # - High entailment (>0.5) → Truthful
                    # - High contradiction (>0.3) → Hallucination
                    # - Low entailment (<0.3) → Hallucination
                    if entailment_prob > 0.5:
                        self.labels[q_id] = 0  # Truthful
                    elif contradiction_prob > 0.3:
                        self.labels[q_id] = 1  # Hallucination
                    elif entailment_prob < 0.3:
                        self.labels[q_id] = 1  # Hallucination
                    else:
                        self.labels[q_id] = 0  # Uncertain → default to truthful

        # Fill missing
        for q_id in self.question_ids:
            if q_id not in self.labels:
                self.labels[q_id] = 0

        # Stats
        n_halluc = sum(self.labels.values())
        n_total = len(self.labels)
        avg_entailment = np.mean(entailment_scores) if entailment_scores else 0

        print(f"   [OK] NLI Labels: {n_halluc}/{n_total} hallucinations ({100*n_halluc/n_total:.1f}%)")
        print(f"   [OK] Average entailment score: {avg_entailment:.3f}")

        # Cleanup
        del label_model, label_tokenizer
        if DEVICE == 'cuda':
            torch.cuda.empty_cache()

    def _precompute_self_verification_scores(self):
        """Pre-compute self-verification scores using NLI between original and variant responses."""
        print("\n[*] Pre-computing self-verification scores (BATCH)...")

        self.self_verification_cache = {}

        originals = []
        variants_list = []
        valid_qids = []

        for q_id in self.question_ids:
            orig_resp = self.original_responses.get('responses', {}).get(q_id, {}).get('original', '')
            if not orig_resp:
                continue

            var_resps = self.variant_responses.get('responses', {}).get(q_id, {})
            if not var_resps:
                continue

            orig_answer = self.consistency_analyzer.extract_answer(orig_resp)
            for var_type, var_resp in var_resps.items():
                var_answer = self.consistency_analyzer.extract_answer(var_resp)
                originals.append(orig_answer)
                variants_list.append(var_answer)
                valid_qids.append(q_id)

        if not originals:
            for q_id in self.question_ids:
                self.self_verification_cache[q_id] = 0.0
            print("   [WARN] No valid pairs for self-verification")
            return

        # Batch NLI inference
        batch_size = self.consistency_analyzer.batch_size
        all_contradiction_scores = []

        with torch.no_grad():
            for batch_start in tqdm(range(0, len(originals), batch_size), desc="Self-Verification Batch"):
                batch_end = min(batch_start + batch_size, len(originals))
                batch_orig = originals[batch_start:batch_end]
                batch_var = variants_list[batch_start:batch_end]

                inputs = self.consistency_analyzer.tokenizer(
                    batch_orig, batch_var,
                    return_tensors='pt', truncation=True, max_length=512, padding=True
                ).to(self.consistency_analyzer.device)

                outputs = self.consistency_analyzer.model(**inputs)
                probs = torch.softmax(outputs.logits.float(), dim=-1)
                contradiction_probs = probs[:, 0].cpu().tolist()
                all_contradiction_scores.extend(contradiction_probs)

        # Aggregate per question (max contradiction)
        temp_scores = {q_id: [] for q_id in self.question_ids}
        for idx, q_id in enumerate(valid_qids):
            temp_scores[q_id].append(all_contradiction_scores[idx])

        for q_id in self.question_ids:
            scores = temp_scores.get(q_id, [])
            self.self_verification_cache[q_id] = max(scores) if scores else 0.0

        print(f"   [OK] Pre-computed self-verification for {len(self.self_verification_cache)} questions")

    def _precompute_minority_penalties(self):
        """Pre-compute minority penalties using answer extraction and majority voting."""
        print("\n[*] Pre-computing minority penalties...")
        import re
        from collections import Counter

        self.minority_penalty_cache = {}

        for q_id in tqdm(self.question_ids, desc="Minority Penalties"):
            responses = prepare_responses_for_question(q_id, self.original_responses, self.variant_responses)
            if len(responses) < 2:
                self.minority_penalty_cache[q_id] = 0.0
                continue

            # Extract answers (simple: last number or key phrase)
            answers = []
            for _, resp in responses:
                answer = self.consistency_analyzer.extract_answer(resp)
                # Get just the core answer (first 50 chars)
                answers.append(answer[:50].lower().strip() if answer else "")

            valid = [a for a in answers if a]
            if not valid:
                self.minority_penalty_cache[q_id] = 0.0
                continue

            counter = Counter(valid)
            most_common = counter.most_common(1)[0]
            majority_ratio = most_common[1] / len(answers)

            # If majority is weak (<60%), apply penalty
            if majority_ratio < 0.6:
                self.minority_penalty_cache[q_id] = 0.30
            elif majority_ratio < 0.8:
                self.minority_penalty_cache[q_id] = 0.10
            else:
                self.minority_penalty_cache[q_id] = 0.0

        print(f"   [OK] Pre-computed minority penalties for {len(self.minority_penalty_cache)} questions")

    def _precompute_pairwise_ncd(self):
        """Pre-compute pairwise NCD for all questions."""
        print("\n[*] Pre-computing pairwise NCD...")
        self.pairwise_ncd_cache = {}

        for q_id in tqdm(self.question_ids, desc="Pairwise NCD"):
            responses = prepare_responses_for_question(q_id, self.original_responses, self.variant_responses)
            if len(responses) < 2:
                self.pairwise_ncd_cache[q_id] = 0.0
                continue

            response_texts = [r[1] for r in responses]
            self.pairwise_ncd_cache[q_id] = self.complexity_analyzer.calculate_pairwise_ncd(response_texts)

        print(f"   [OK] Pre-computed pairwise NCD for {len(self.pairwise_ncd_cache)} questions")

    def _precompute_nli_scores(self):
        """PRE-COMPUTE all NLI pair scores for all questions.
        This runs NLI inference ONCE, then trials just apply different thresholds."""
        print("\n[*] Pre-computing ALL NLI pair scores (BATCH)...")

        self.nli_scores_cache = {}  # q_id -> list of (contradiction_prob, neutral_prob)
        self.complexity_cache = {}  # q_id -> (entropy, ncd) with entropy_max=10, ncd_max=1

        all_pairs_a = []
        all_pairs_b = []
        pair_mapping = []  # (q_id, pair_index)

        # Collect all pairs from all questions
        for q_id in tqdm(self.question_ids, desc="Collecting pairs"):
            responses = prepare_responses_for_question(q_id, self.original_responses, self.variant_responses)

            if len(responses) < 2:
                self.nli_scores_cache[q_id] = []
                self.complexity_cache[q_id] = (0.0, 0.0)
                continue

            # Compute complexity (doesn't need GPU, do it now)
            entropy, ncd = self.complexity_analyzer.analyze(responses, entropy_max=10.0, ncd_max=1.0)
            self.complexity_cache[q_id] = (entropy, ncd)

            # Extract answers for NLI
            answers = [self.consistency_analyzer.extract_answer(r[1]) for r in responses]
            n = len(answers)

            for i in range(n):
                for j in range(i + 1, n):
                    all_pairs_a.append(answers[i])
                    all_pairs_b.append(answers[j])
                    pair_mapping.append(q_id)

        print(f"   Total NLI pairs: {len(all_pairs_a)}")

        # BATCH NLI inference for ALL pairs
        all_contradiction_probs = []
        all_neutral_probs = []
        batch_size = self.consistency_analyzer.batch_size

        with torch.no_grad():
            for batch_start in tqdm(range(0, len(all_pairs_a), batch_size), desc="NLI Batch Inference"):
                batch_end = min(batch_start + batch_size, len(all_pairs_a))
                batch_a = all_pairs_a[batch_start:batch_end]
                batch_b = all_pairs_b[batch_start:batch_end]

                inputs = self.consistency_analyzer.tokenizer(
                    batch_a, batch_b,
                    return_tensors='pt',
                    truncation=True,
                    max_length=512,
                    padding=True
                ).to(self.consistency_analyzer.device)

                outputs = self.consistency_analyzer.model(**inputs)
                probs = torch.softmax(outputs.logits.float(), dim=-1)

                all_contradiction_probs.extend(probs[:, 0].cpu().tolist())
                all_neutral_probs.extend(probs[:, 1].cpu().tolist())

        # Group scores by question
        temp_scores = {q_id: [] for q_id in self.question_ids}
        for idx, q_id in enumerate(pair_mapping):
            temp_scores[q_id].append((all_contradiction_probs[idx], all_neutral_probs[idx]))

        self.nli_scores_cache = temp_scores

        # Cleanup GPU memory
        if DEVICE == 'cuda':
            torch.cuda.empty_cache()

        print(f"   [OK] Pre-computed NLI scores for {len(self.nli_scores_cache)} questions")
        print(f"   [OK] GPU memory freed - trials will be VERY FAST now!")

    def compute_features_fast(self, q_id, min_edge_weight, contradiction_weight, neutral_weight,
                              entropy_max, ncd_max):
        """FAST feature computation using pre-cached scores"""
        nli_scores = self.nli_scores_cache.get(q_id, [])
        if not nli_scores:
            return None

        # Consistency
        contradictions = sum(1 for contr_prob, neut_prob in nli_scores
                             if contr_prob * contradiction_weight + neut_prob * neutral_weight > min_edge_weight)
        consistency = 1.0 - (contradictions / len(nli_scores))

        # Entropy
        base_entropy, _ = self.complexity_cache.get(q_id, (0.0, 0.0))
        entropy = min(base_entropy * 10.0 / entropy_max, 1.0)

        # Pairwise NCD
        raw_ncd = self.pairwise_ncd_cache.get(q_id, 0.0)
        ncd = min(raw_ncd / ncd_max, 1.0)

        # Self-verification
        self_verification = self.self_verification_cache.get(q_id, 0.0)

        # Minority penalty
        minority_penalty = self.minority_penalty_cache.get(q_id, 0.0)

        return (consistency, entropy, ncd, self_verification, minority_penalty)

    def objective(self, trial: optuna.Trial) -> float:
        alpha = trial.suggest_float('alpha', 0.05, 0.50)   # Self-verification
        beta = trial.suggest_float('beta', 0.05, 0.40)     # Inconsistency
        gamma = trial.suggest_float('gamma', 0.05, 0.30)   # Entropy
        delta = trial.suggest_float('delta', 0.01, 0.30)   # NCD
        epsilon = 1.0 - alpha - beta - gamma - delta        # Minority penalty

        if epsilon < 0.0 or epsilon > 0.40:
            return 0.0

        threshold = trial.suggest_float('hallucination_threshold', 0.20, 0.70)
        min_edge_weight = trial.suggest_float('min_edge_weight', 0.15, 0.60)
        contradiction_weight = trial.suggest_float('contradiction_weight', 0.50, 1.00)
        neutral_weight = trial.suggest_float('neutral_weight', 0.20, 0.70)
        entropy_max = trial.suggest_float('entropy_max', 5.0, 15.0)
        ncd_max = trial.suggest_float('ncd_max', 0.50, 1.00)

        predictions = []
        ground_truths = []

        for q_id in self.question_ids:
            features = self.compute_features_fast(
                q_id, min_edge_weight, contradiction_weight, neutral_weight, entropy_max, ncd_max
            )
            if features is None:
                continue

            consistency, entropy, ncd, self_verification, minority_penalty = features
            inconsistency = 1.0 - consistency

            risk = calculate_risk(self_verification, inconsistency, entropy, ncd,
                                  minority_penalty, alpha, beta, gamma, delta, epsilon)
            pred = 1 if risk > threshold else 0

            predictions.append(pred)
            ground_truths.append(self.labels[q_id])

        if len(predictions) < 10:
            return 0.0

        return f1_score(ground_truths, predictions, zero_division=0)

    def optimize(self, n_trials=100, timeout=None):
        """Run Bayesian optimization"""
        print(f"\n{'='*60}")
        print("[*] STARTING BAYESIAN OPTIMIZATION")
        print(f"{'='*60}")
        print(f"Questions: {len(self.question_ids)}")
        print(f"Trials: {n_trials}")
        print(f"Device: {DEVICE}")

        study = optuna.create_study(direction='maximize', study_name='logic_halt_batch')

        pbar = tqdm(total=n_trials, desc="Optimization")

        def callback(study, trial):
            pbar.update(1)
            pbar.set_postfix({'best_f1': f"{study.best_value:.4f}"})
            if trial.number % 10 == 0:
                gc.collect()
                if DEVICE == 'cuda':
                    torch.cuda.empty_cache()

        study.optimize(
            self.objective,
            n_trials=n_trials,
            timeout=timeout,
            callbacks=[callback],
            show_progress_bar=False
        )

        pbar.close()

        self._print_results(study)
        self._save_results(study)

        return study

    def _print_results(self, study):
        print(f"\n{'='*60}")
        print("[OK] OPTIMIZATION COMPLETE")
        print(f"{'='*60}")

        best = study.best_params
        epsilon = 1.0 - best['alpha'] - best['beta'] - best['gamma'] - best['delta']

        print(f"\n[*] Best F1 Score: {study.best_value:.4f}")
        print(f"   Total trials: {len(study.trials)}")
        print(f"\n[*] Optimal Hyperparameters (5-weight fusion, NO GT contradiction):")
        print(f"   alpha   (Self-Verification): {best['alpha']:.4f}")
        print(f"   beta    (Inconsistency):     {best['beta']:.4f}")
        print(f"   gamma   (Entropy):           {best['gamma']:.4f}")
        print(f"   delta   (NCD):               {best['delta']:.4f}")
        print(f"   epsilon (Minority Penalty):  {epsilon:.4f}")
        print(f"   Threshold:                   {best['hallucination_threshold']:.4f}")

    def _save_results(self, study):
        OUTPUT_DIR.mkdir(exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

        best_params = study.best_params.copy()
        best_params['epsilon'] = 1.0 - best_params['alpha'] - best_params['beta'] - best_params['gamma'] - best_params['delta']
        best_params['best_f1_score'] = study.best_value
        best_params['n_trials'] = len(study.trials)
        best_params['timestamp'] = timestamp
        best_params['n_questions'] = len(self.question_ids)
        best_params['fusion_type'] = '5-weight without GT contradiction'

        json_path = OUTPUT_DIR / f'batch_optimization_best_params.json'
        with open(json_path, 'w') as f:
            json.dump(best_params, f, indent=2)
        print(f"\n[*] Results saved to: {json_path}")

        # Save trials (handle pandas import errors gracefully)
        try:
            df = study.trials_dataframe()
            csv_path = OUTPUT_DIR / f'batch_optimization_trials.csv'
            df.to_csv(csv_path, index=False)
            print(f"[*] Trials saved to: {csv_path}")
        except ImportError as e:
            print(f"[WARN] Could not save CSV (pandas issue): {e}")
            # Save as JSON instead
            trials_data = [{'number': t.number, 'value': t.value, 'params': t.params}
                          for t in study.trials]
            json_trials_path = OUTPUT_DIR / f'batch_optimization_trials.json'
            with open(json_trials_path, 'w') as f:
                json.dump(trials_data, f, indent=2)
            print(f"[*] Trials saved as JSON: {json_trials_path}")


def main():
    import argparse

    parser = argparse.ArgumentParser(description='Batch Optimization with GPU Acceleration')
    parser.add_argument('--trials', type=int, default=100, help='Number of Optuna trials')
    parser.add_argument('--timeout', type=int, default=None, help='Timeout in seconds')
    parser.add_argument('--batch-size', type=int, default=64,
                        help='NLI batch size (higher=more GPU usage). RTX 5070: 64-128 recommended')

    args = parser.parse_args()

    print(f"\n{'='*60}")
    print(f"GPU BATCH OPTIMIZATION")
    print(f"{'='*60}")
    print(f"NLI Batch Size: {args.batch_size}")
    print(f"Trials: {args.trials}")
    print(f"Expected GPU Usage: ~{2 + args.batch_size * 0.05:.1f} GB VRAM")
    print(f"{'='*60}")

    optimizer = BatchOptimizer(nli_batch_size=args.batch_size)
    study = optimizer.optimize(n_trials=args.trials, timeout=args.timeout)


if __name__ == '__main__':
    main()
