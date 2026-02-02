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
def calculate_risk(gt_contradiction: float, consistency: float, entropy: float, ncd: float,
                   alpha: float, beta: float, gamma: float, delta: float) -> float:
    """
    Calculate hallucination risk score with ground truth signal as primary indicator.

    Args:
        gt_contradiction: NLI contradiction score vs ground truth (PRIMARY signal)
        consistency: Self-consistency score between variants
        entropy: Token entropy score
        ncd: Normalized compression distance
        alpha: Weight for GT contradiction (recommended: 0.50)
        beta: Weight for inconsistency (recommended: 0.25)
        gamma: Weight for entropy (recommended: 0.15)
        delta: Weight for NCD (recommended: 0.10)
    """
    inconsistency = 1.0 - consistency
    risk = alpha * gt_contradiction + beta * inconsistency + gamma * entropy + delta * ncd
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

        # PRE-COMPUTE all GT contradictions in batch (FAST!)
        self._precompute_gt_contradictions()

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

    def _precompute_gt_contradictions(self):
        """BATCH compute all GT contradiction scores at once - much faster!"""
        print("\n[*] Pre-computing GT contradictions (BATCH)...")

        answers = []
        gts = []
        valid_qids = []

        for q_id in self.question_ids:
            gt = self.ground_truths.get(q_id, '')
            if not gt:
                continue

            orig_resp = self.original_responses.get('responses', {}).get(q_id, {}).get('original', '')
            if not orig_resp:
                var_resps = self.variant_responses.get('responses', {}).get(q_id, {})
                if var_resps:
                    orig_resp = list(var_resps.values())[0]

            if not orig_resp:
                continue

            answer = self.consistency_analyzer.extract_answer(orig_resp)
            answers.append(answer)
            gts.append(gt)
            valid_qids.append(q_id)

        # BATCH inference
        self.gt_contradiction_cache = {}
        batch_size = self.consistency_analyzer.batch_size

        with torch.no_grad():
            for batch_start in tqdm(range(0, len(answers), batch_size), desc="GT Contradiction Batch"):
                batch_end = min(batch_start + batch_size, len(answers))
                batch_answers = answers[batch_start:batch_end]
                batch_gts = gts[batch_start:batch_end]
                batch_qids = valid_qids[batch_start:batch_end]

                inputs = self.consistency_analyzer.tokenizer(
                    batch_answers, batch_gts,
                    return_tensors='pt',
                    truncation=True,
                    max_length=512,
                    padding=True
                ).to(self.consistency_analyzer.device)

                outputs = self.consistency_analyzer.model(**inputs)
                probs = torch.softmax(outputs.logits.float(), dim=-1)  # Back to FP32 for softmax

                contradiction_probs = probs[:, 0].cpu().numpy()
                entailment_probs = probs[:, 2].cpu().numpy()

                for i, q_id in enumerate(batch_qids):
                    gt_risk = contradiction_probs[i] * 0.7 + (1 - entailment_probs[i]) * 0.3
                    self.gt_contradiction_cache[q_id] = float(gt_risk)

        # Fill missing with 0
        for q_id in self.question_ids:
            if q_id not in self.gt_contradiction_cache:
                self.gt_contradiction_cache[q_id] = 0.0

        print(f"   [OK] Pre-computed {len(self.gt_contradiction_cache)} GT contradictions")

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

    def compute_gt_contradiction(self, q_id):
        """Get cached GT contradiction score"""
        return self.gt_contradiction_cache.get(q_id, 0.0)

    def compute_features_fast(self, q_id, min_edge_weight, contradiction_weight, neutral_weight,
                              entropy_max, ncd_max):
        """FAST feature computation using pre-cached NLI scores"""
        nli_scores = self.nli_scores_cache.get(q_id, [])

        if not nli_scores:
            return None

        # Apply thresholds to cached scores (NO GPU needed!)
        contradictions = 0
        for contr_prob, neut_prob in nli_scores:
            score = contr_prob * contradiction_weight + neut_prob * neutral_weight
            if score > min_edge_weight:
                contradictions += 1

        consistency = 1.0 - (contradictions / len(nli_scores))

        # Get cached complexity (scale by parameters)
        base_entropy, base_ncd = self.complexity_cache.get(q_id, (0.0, 0.0))
        entropy = min(base_entropy * 10.0 / entropy_max, 1.0)  # Rescale
        ncd = min(base_ncd / ncd_max, 1.0)

        gt_contradiction = self.compute_gt_contradiction(q_id)

        return (consistency, entropy, ncd, gt_contradiction)

    def objective(self, trial: optuna.Trial) -> float:
        """Optuna objective function with ground truth signal as primary feature"""
        # Sample hyperparameters for new 4-weight fusion
        # Weights must sum to 1.0 and all be positive
        alpha = trial.suggest_float('alpha', 0.10, 0.70)  # GT contradiction weight (PRIMARY)
        beta = trial.suggest_float('beta', 0.05, 0.40)    # Inconsistency weight
        gamma = trial.suggest_float('gamma', 0.05, 0.40)  # Entropy weight
        delta = 1.0 - alpha - beta - gamma                # NCD weight (remainder)

        # Ensure all weights are valid (positive and sum to 1)
        if delta < 0.0 or delta > 0.50:
            return 0.0
        if alpha + beta + gamma + delta > 1.01 or alpha + beta + gamma + delta < 0.99:
            return 0.0

        threshold = trial.suggest_float('hallucination_threshold', 0.20, 0.70)
        min_edge_weight = trial.suggest_float('min_edge_weight', 0.15, 0.60)
        contradiction_weight = trial.suggest_float('contradiction_weight', 0.50, 1.00)
        neutral_weight = trial.suggest_float('neutral_weight', 0.20, 0.70)
        entropy_max = trial.suggest_float('entropy_max', 5.0, 15.0)
        ncd_max = trial.suggest_float('ncd_max', 0.50, 1.00)

        # Evaluate on all questions (using PRE-CACHED NLI scores - NO GPU!)
        predictions = []
        ground_truths = []

        for q_id in self.question_ids:
            features = self.compute_features_fast(
                q_id, min_edge_weight, contradiction_weight, neutral_weight, entropy_max, ncd_max
            )

            if features is None:
                continue

            consistency, entropy, ncd, gt_contradiction = features
            risk = calculate_risk(gt_contradiction, consistency, entropy, ncd, alpha, beta, gamma, delta)
            pred = 1 if risk > threshold else 0

            predictions.append(pred)
            ground_truths.append(self.labels[q_id])

        if len(predictions) < 10:
            return 0.0

        f1 = f1_score(ground_truths, predictions, zero_division=0)
        return f1

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
        delta = 1.0 - best['alpha'] - best['beta'] - best['gamma']

        print(f"\n[*] Best F1 Score: {study.best_value:.4f}")
        print(f"   Total trials: {len(study.trials)}")
        print(f"\n[*] Optimal Hyperparameters (4-weight fusion):")
        print(f"   alpha (GT Contradiction): {best['alpha']:.4f}  <- PRIMARY signal")
        print(f"   beta  (Inconsistency):    {best['beta']:.4f}")
        print(f"   gamma (Entropy):          {best['gamma']:.4f}")
        print(f"   delta (NCD):              {delta:.4f}")
        print(f"   Threshold:            {best['hallucination_threshold']:.4f}")

    def _save_results(self, study):
        OUTPUT_DIR.mkdir(exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

        best_params = study.best_params.copy()
        best_params['delta'] = 1.0 - best_params['alpha'] - best_params['beta'] - best_params['gamma']
        best_params['best_f1_score'] = study.best_value
        best_params['n_trials'] = len(study.trials)
        best_params['timestamp'] = timestamp
        best_params['n_questions'] = len(self.question_ids)
        best_params['fusion_type'] = '4-weight with GT contradiction'

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
