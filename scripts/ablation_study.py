"""
Ablation Study: 3 New Signal Improvements
Tests all 8 combinations (2^3) to find the best signal set.

New signals:
1. Negation Contradiction - does the model agree with both a claim and its negation?
2. Logprob Spike (Factual Token Variance) - do factual tokens change across variants?
3. Semantic Answer Matching - sentence-BERT based majority voting instead of string matching
"""

import json
import sys
import gc
import zlib
import re
import numpy as np
import torch
from pathlib import Path
from typing import List, Tuple, Dict
from itertools import product
from tqdm import tqdm
from collections import Counter

sys.path.append(str(Path(__file__).parent.parent))

from sklearn.metrics import f1_score, precision_score, recall_score
import optuna

# ========================================
# PATHS
# ========================================
BASE_DIR = Path(__file__).parent.parent
BATCH_RESULTS_DIR = BASE_DIR / 'data' / 'batch_results'
DATASET_PATH = BASE_DIR / 'data' / 'raw' / 'truthfulqa_dataset.json'
OUTPUT_DIR = BASE_DIR / 'config' / 'optimization_results'

VARIANTS_FILE = BATCH_RESULTS_DIR / 'batch_batch_697f1cb8f0fc819080718b3eba0544c2_organized.json'
ORIGINAL_RESPONSES_FILE = BATCH_RESULTS_DIR / 'batch_batch_697f1cc0055c81908d8b8c3d78c4cb80_organized.json'
VARIANT_RESPONSES_FILE = BATCH_RESULTS_DIR / 'batch_batch_697f2470463081908e5842fb09956d58_organized.json'


# ========================================
# CUDA
# ========================================
def setup_cuda():
    if not torch.cuda.is_available():
        return 'cpu'
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.cuda.empty_cache()
    print(f"[GPU] {torch.cuda.get_device_name(0)}")
    return 'cuda'

DEVICE = setup_cuda()


# ========================================
# DATA LOADING
# ========================================
def load_data():
    with open(VARIANTS_FILE, 'r', encoding='utf-8') as f:
        variants_data = json.load(f)
    with open(ORIGINAL_RESPONSES_FILE, 'r', encoding='utf-8') as f:
        original_responses = json.load(f)
    with open(VARIANT_RESPONSES_FILE, 'r', encoding='utf-8') as f:
        variant_responses = json.load(f)
    with open(DATASET_PATH, 'r', encoding='utf-8') as f:
        dataset = json.load(f)

    questions = dataset if isinstance(dataset, list) else dataset.get('questions', [])
    ground_truths = {q.get('id', q.get('question_id')): q.get('ground_truth', '')
                     for q in questions}

    return variants_data, original_responses, variant_responses, ground_truths


def get_responses(q_id, original_responses, variant_responses):
    responses = []
    orig = original_responses.get('responses', {}).get(q_id, {}).get('original', '')
    if orig:
        responses.append(('original', orig))
    var_resps = variant_responses.get('responses', {}).get(q_id, {})
    for var_type, resp in var_resps.items():
        if resp:
            responses.append((var_type, resp))
    return responses


def extract_answer(text):
    if not text:
        return ""
    if 'ANSWER:' in text:
        parts = text.split('ANSWER:')
        if len(parts) > 1:
            answer_part = parts[1].split('REASONING:')[0] if 'REASONING:' in parts[1] else parts[1]
            return answer_part.strip()
    return text.strip()[:200]


# ========================================
# NLI MODEL
# ========================================
class NLIModel:
    def __init__(self, device='cuda', batch_size=64):
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        print("[*] Loading NLI model (large)...")
        model_name = "cross-encoder/nli-deberta-v3-large"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.model.to(device)
        self.model.eval()
        if device == 'cuda':
            self.model = self.model.half()
        self.device = device
        self.batch_size = batch_size
        print(f"   [OK] NLI model on {device}")

    @torch.no_grad()
    def batch_predict(self, texts_a, texts_b):
        """Returns (contradiction_probs, neutral_probs, entailment_probs) for each pair."""
        all_c, all_n, all_e = [], [], []
        for i in range(0, len(texts_a), self.batch_size):
            ba = texts_a[i:i+self.batch_size]
            bb = texts_b[i:i+self.batch_size]
            inputs = self.tokenizer(ba, bb, return_tensors='pt', truncation=True,
                                     max_length=512, padding=True).to(self.device)
            outputs = self.model(**inputs)
            probs = torch.softmax(outputs.logits.float(), dim=-1)
            all_c.extend(probs[:, 0].cpu().tolist())
            all_n.extend(probs[:, 1].cpu().tolist())
            all_e.extend(probs[:, 2].cpu().tolist())
        return all_c, all_n, all_e


# ========================================
# SENTENCE-BERT MODEL
# ========================================
class SBERTModel:
    def __init__(self, device='cuda'):
        from sentence_transformers import SentenceTransformer
        print("[*] Loading Sentence-BERT...")
        self.model = SentenceTransformer('all-MiniLM-L6-v2', device=device)
        self.device = device
        print("   [OK] SBERT loaded")

    def encode(self, texts, batch_size=256):
        return self.model.encode(texts, batch_size=batch_size, convert_to_numpy=True,
                                  show_progress_bar=False)


# ========================================
# LABEL COMPUTATION (same as batch_optimization.py)
# ========================================
def compute_labels(question_ids, original_responses, variant_responses, ground_truths, nli_model):
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    print("[*] Computing labels...")
    label_tokenizer = AutoTokenizer.from_pretrained("microsoft/deberta-large-mnli")
    label_model = AutoModelForSequenceClassification.from_pretrained("microsoft/deberta-large-mnli")
    label_model.to(DEVICE).eval()
    if DEVICE == 'cuda':
        label_model = label_model.half()

    labels = {}
    answers, gts, valid_qids = [], [], []

    for q_id in question_ids:
        gt = ground_truths.get(q_id, '')
        if not gt:
            labels[q_id] = 0
            continue
        orig = original_responses.get('responses', {}).get(q_id, {}).get('original', '')
        if not orig:
            labels[q_id] = 0
            continue
        answers.append(extract_answer(orig))
        gts.append(gt)
        valid_qids.append(q_id)

    with torch.no_grad():
        for i in range(0, len(answers), 32):
            ba = answers[i:i+32]
            bg = gts[i:i+32]
            bq = valid_qids[i:i+32]
            inputs = label_tokenizer(ba, bg, return_tensors='pt', truncation=True,
                                      max_length=512, padding=True).to(DEVICE)
            probs = torch.softmax(label_model(**inputs).logits.float(), dim=-1)
            for j, q_id in enumerate(bq):
                ent = probs[j, 2].item()
                con = probs[j, 0].item()
                if ent > 0.5:
                    labels[q_id] = 0
                elif con > 0.3 or ent < 0.3:
                    labels[q_id] = 1
                else:
                    labels[q_id] = 0

    for q_id in question_ids:
        if q_id not in labels:
            labels[q_id] = 0

    n_hal = sum(labels.values())
    print(f"   [OK] Labels: {n_hal}/{len(labels)} hallucinations ({100*n_hal/len(labels):.1f}%)")

    del label_model, label_tokenizer
    if DEVICE == 'cuda':
        torch.cuda.empty_cache()

    return labels


# ========================================
# FEATURE PRECOMPUTATION
# ========================================
class FeatureComputer:
    def __init__(self, question_ids, original_responses, variant_responses,
                 variants_data, ground_truths, nli_model, sbert_model):
        self.question_ids = question_ids
        self.original_responses = original_responses
        self.variant_responses = variant_responses
        self.variants_data = variants_data
        self.ground_truths = ground_truths
        self.nli = nli_model
        self.sbert = sbert_model

        # Base features (from batch_optimization.py)
        self._precompute_nli_pairs()
        self._precompute_complexity()
        self._precompute_self_verification()

        # NEW Feature 1: Negation Contradiction
        self._precompute_negation_contradiction()

        # NEW Feature 2: Factual Token Variance (logprob spike proxy)
        self._precompute_factual_token_variance()

        # NEW Feature 3: Semantic Answer Matching
        self._precompute_semantic_answer_matching()

    def _precompute_nli_pairs(self):
        """Standard NLI pair scores."""
        print("\n[*] Pre-computing NLI pair scores...")
        self.nli_cache = {}
        all_a, all_b, mapping = [], [], []

        for q_id in self.question_ids:
            responses = get_responses(q_id, self.original_responses, self.variant_responses)
            if len(responses) < 2:
                self.nli_cache[q_id] = []
                continue
            answers = [extract_answer(r[1]) for r in responses]
            for i in range(len(answers)):
                for j in range(i+1, len(answers)):
                    all_a.append(answers[i])
                    all_b.append(answers[j])
                    mapping.append(q_id)

        c_probs, n_probs, _ = self.nli.batch_predict(all_a, all_b)

        temp = {q: [] for q in self.question_ids}
        for idx, q_id in enumerate(mapping):
            temp[q_id].append((c_probs[idx], n_probs[idx]))
        self.nli_cache = temp
        print(f"   [OK] {len(all_a)} pairs")

    def _precompute_complexity(self):
        """Entropy and NCD."""
        print("[*] Pre-computing complexity...")
        self.entropy_cache = {}
        self.ncd_cache = {}

        for q_id in self.question_ids:
            responses = get_responses(q_id, self.original_responses, self.variant_responses)
            if len(responses) < 2:
                self.entropy_cache[q_id] = 0.0
                self.ncd_cache[q_id] = 0.0
                continue

            # Token entropy
            entropies = []
            for _, resp in responses:
                tokens = resp.split()
                if tokens:
                    counts = Counter(tokens)
                    total = len(tokens)
                    probs = [c/total for c in counts.values()]
                    entropies.append(-sum(p * np.log2(p) for p in probs if p > 0))
            self.entropy_cache[q_id] = np.mean(entropies) if entropies else 0.0

            # Pairwise NCD
            texts = [r[1] for r in responses]
            ncds = []
            for i in range(len(texts)):
                for j in range(i+1, len(texts)):
                    t1 = texts[i].encode('utf-8')
                    t2 = texts[j].encode('utf-8')
                    cx = len(zlib.compress(t1))
                    cy = len(zlib.compress(t2))
                    cxy = len(zlib.compress(t1 + t2))
                    d = max(cx, cy)
                    ncds.append((cxy - min(cx, cy)) / d if d > 0 else 0)
            self.ncd_cache[q_id] = np.mean(ncds) if ncds else 0.0

        print(f"   [OK] complexity for {len(self.question_ids)} questions")

    def _precompute_self_verification(self):
        """Original vs variant NLI contradiction (existing signal)."""
        print("[*] Pre-computing self-verification...")
        self.sv_cache = {}

        originals, variants_list, valid_qids = [], [], []
        for q_id in self.question_ids:
            orig = self.original_responses.get('responses', {}).get(q_id, {}).get('original', '')
            if not orig:
                continue
            var_resps = self.variant_responses.get('responses', {}).get(q_id, {})
            if not var_resps:
                continue
            orig_ans = extract_answer(orig)
            for _, var_resp in var_resps.items():
                originals.append(orig_ans)
                variants_list.append(extract_answer(var_resp))
                valid_qids.append(q_id)

        if originals:
            c_probs, _, _ = self.nli.batch_predict(originals, variants_list)
            temp = {q: [] for q in self.question_ids}
            for idx, q_id in enumerate(valid_qids):
                temp[q_id].append(c_probs[idx])
            for q_id in self.question_ids:
                scores = temp.get(q_id, [])
                self.sv_cache[q_id] = max(scores) if scores else 0.0
        else:
            for q_id in self.question_ids:
                self.sv_cache[q_id] = 0.0

        print(f"   [OK] self-verification for {len(self.sv_cache)} questions")

    # ========================================
    # NEW SIGNAL 1: Negation Contradiction
    # ========================================
    def _precompute_negation_contradiction(self):
        """
        If model agrees with BOTH original AND its negation variant → hallucination.
        Uses NLI entailment between original response and negation response.
        High entailment for both = logical contradiction = hallucination.
        """
        print("[*] Pre-computing negation contradiction...")
        self.negation_cache = {}

        originals, negations, valid_qids = [], [], []
        for q_id in self.question_ids:
            orig = self.original_responses.get('responses', {}).get(q_id, {}).get('original', '')
            if not orig:
                continue
            var_resps = self.variant_responses.get('responses', {}).get(q_id, {})
            # Find negation variant
            neg_resp = var_resps.get('negation', '')
            if not neg_resp:
                # Try other keys that might be negation
                for k, v in var_resps.items():
                    if 'neg' in k.lower():
                        neg_resp = v
                        break
            if not neg_resp:
                continue

            originals.append(extract_answer(orig))
            negations.append(extract_answer(neg_resp))
            valid_qids.append(q_id)

        if originals:
            # Check entailment between original and negation responses
            # If both responses AGREE (high entailment), the model is contradicting logic
            c_probs, _, e_probs = self.nli.batch_predict(originals, negations)

            for idx, q_id in enumerate(valid_qids):
                # High entailment between original and negation = model agrees with both = BAD
                # High contradiction = model disagrees = GOOD (logically consistent)
                entailment = e_probs[idx]
                contradiction = c_probs[idx]
                # Score: high when model agrees with contradictory premises
                self.negation_cache[q_id] = entailment * 0.7 + (1.0 - contradiction) * 0.3

        for q_id in self.question_ids:
            if q_id not in self.negation_cache:
                self.negation_cache[q_id] = 0.5  # Neutral default

        n_with_neg = len(valid_qids)
        print(f"   [OK] negation contradiction for {n_with_neg}/{len(self.question_ids)} questions")

    # ========================================
    # NEW SIGNAL 2: Factual Token Variance
    # ========================================
    def _precompute_factual_token_variance(self):
        """
        Proxy for logprob spikes: check if factual tokens (numbers, proper nouns)
        change across variant responses. If factual content is unstable → hallucination.
        """
        print("[*] Pre-computing factual token variance...")
        self.ftv_cache = {}

        number_pattern = re.compile(r'\b\d+\.?\d*\b')
        proper_noun_pattern = re.compile(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b')

        for q_id in self.question_ids:
            responses = get_responses(q_id, self.original_responses, self.variant_responses)
            if len(responses) < 2:
                self.ftv_cache[q_id] = 0.0
                continue

            # Extract factual tokens from each response
            all_numbers = []
            all_proper_nouns = []

            for _, resp in responses:
                answer = extract_answer(resp)
                numbers = set(number_pattern.findall(answer))
                proper_nouns = set(proper_noun_pattern.findall(answer))
                all_numbers.append(numbers)
                all_proper_nouns.append(proper_nouns)

            # Measure variance: Jaccard distance between factual token sets
            n = len(responses)
            number_distances = []
            noun_distances = []

            for i in range(n):
                for j in range(i+1, n):
                    # Number variance
                    if all_numbers[i] or all_numbers[j]:
                        union = all_numbers[i] | all_numbers[j]
                        intersection = all_numbers[i] & all_numbers[j]
                        number_distances.append(1.0 - len(intersection)/len(union) if union else 0.0)

                    # Proper noun variance
                    if all_proper_nouns[i] or all_proper_nouns[j]:
                        union = all_proper_nouns[i] | all_proper_nouns[j]
                        intersection = all_proper_nouns[i] & all_proper_nouns[j]
                        noun_distances.append(1.0 - len(intersection)/len(union) if union else 0.0)

            # Combine: high variance in factual tokens = hallucination signal
            num_var = np.mean(number_distances) if number_distances else 0.0
            noun_var = np.mean(noun_distances) if noun_distances else 0.0
            self.ftv_cache[q_id] = 0.6 * num_var + 0.4 * noun_var

        print(f"   [OK] factual token variance for {len(self.ftv_cache)} questions")

    # ========================================
    # NEW SIGNAL 3: Semantic Answer Matching
    # ========================================
    def _precompute_semantic_answer_matching(self):
        """
        Use Sentence-BERT to semantically cluster extracted answers,
        then compute minority penalty based on semantic similarity.
        Much better than exact string matching.
        """
        print("[*] Pre-computing semantic answer matching...")
        self.sam_cache = {}

        # Collect all answers
        all_answers = []
        answer_mapping = []  # (q_id, response_idx)

        for q_id in self.question_ids:
            responses = get_responses(q_id, self.original_responses, self.variant_responses)
            if len(responses) < 2:
                self.sam_cache[q_id] = 0.0
                continue
            for idx, (_, resp) in enumerate(responses):
                answer = extract_answer(resp)
                all_answers.append(answer)
                answer_mapping.append((q_id, idx))

        if not all_answers:
            return

        # Batch encode all answers
        embeddings = self.sbert.encode(all_answers, batch_size=256)

        # Group by question and compute semantic majority
        q_embeddings = {}
        for idx, (q_id, resp_idx) in enumerate(answer_mapping):
            if q_id not in q_embeddings:
                q_embeddings[q_id] = []
            q_embeddings[q_id].append(embeddings[idx])

        for q_id in self.question_ids:
            embs = q_embeddings.get(q_id, [])
            if len(embs) < 2:
                self.sam_cache[q_id] = 0.0
                continue

            embs = np.array(embs)
            n = len(embs)

            # Compute pairwise cosine similarity
            norms = np.linalg.norm(embs, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            normalized = embs / norms
            sim_matrix = normalized @ normalized.T

            # For each answer, compute average similarity to others
            avg_sims = []
            for i in range(n):
                others = [sim_matrix[i][j] for j in range(n) if j != i]
                avg_sims.append(np.mean(others))

            # Semantic minority: answers with low avg similarity to others
            # Score: proportion of "outlier" answers (avg_sim < 0.7)
            outlier_threshold = 0.7
            outlier_count = sum(1 for s in avg_sims if s < outlier_threshold)
            outlier_ratio = outlier_count / n

            # Also: overall answer agreement (high = consistent, low = divergent)
            all_pairwise = []
            for i in range(n):
                for j in range(i+1, n):
                    all_pairwise.append(sim_matrix[i][j])
            avg_agreement = np.mean(all_pairwise)

            # Combine: high outlier ratio + low agreement = hallucination
            self.sam_cache[q_id] = 0.5 * outlier_ratio + 0.5 * (1.0 - avg_agreement)

        print(f"   [OK] semantic answer matching for {len(self.sam_cache)} questions")

    def get_features(self, q_id, min_edge_weight, contradiction_weight, neutral_weight,
                     entropy_max, ncd_max, use_negation=True, use_ftv=True, use_sam=True):
        """Get all features for a question."""
        nli_scores = self.nli_cache.get(q_id, [])
        if not nli_scores:
            return None

        # Base: consistency
        contradictions = sum(1 for cp, np_ in nli_scores
                             if cp * contradiction_weight + np_ * neutral_weight > min_edge_weight)
        consistency = 1.0 - (contradictions / len(nli_scores))

        # Base: entropy
        entropy = min(self.entropy_cache.get(q_id, 0.0) * 10.0 / entropy_max, 1.0)

        # Base: NCD
        ncd = min(self.ncd_cache.get(q_id, 0.0) / ncd_max, 1.0)

        # Base: self-verification
        sv = self.sv_cache.get(q_id, 0.0)

        # NEW signals (conditionally included)
        negation = self.negation_cache.get(q_id, 0.5) if use_negation else 0.0
        ftv = self.ftv_cache.get(q_id, 0.0) if use_ftv else 0.0
        sam = self.sam_cache.get(q_id, 0.0) if use_sam else 0.0

        return (consistency, entropy, ncd, sv, negation, ftv, sam)


# ========================================
# ABLATION OPTIMIZER
# ========================================
def run_optimization(features: FeatureComputer, labels: dict, question_ids: list,
                     use_negation: bool, use_ftv: bool, use_sam: bool,
                     n_trials: int = 500) -> dict:
    """Run optimization for a specific signal combination."""

    def objective(trial):
        # Base 5 weights
        w_sv = trial.suggest_float('w_sv', 0.01, 0.40)
        w_incon = trial.suggest_float('w_incon', 0.01, 0.40)
        w_ent = trial.suggest_float('w_ent', 0.01, 0.30)
        w_ncd = trial.suggest_float('w_ncd', 0.01, 0.30)
        w_mp = trial.suggest_float('w_mp', 0.01, 0.40)

        # New signal weights
        w_neg = trial.suggest_float('w_neg', 0.01, 0.40) if use_negation else 0.0
        w_ftv = trial.suggest_float('w_ftv', 0.01, 0.30) if use_ftv else 0.0
        w_sam = trial.suggest_float('w_sam', 0.01, 0.40) if use_sam else 0.0

        # Normalize
        total_w = w_sv + w_incon + w_ent + w_ncd + w_mp + w_neg + w_ftv + w_sam
        if total_w == 0:
            return 0.0
        w_sv /= total_w
        w_incon /= total_w
        w_ent /= total_w
        w_ncd /= total_w
        w_mp /= total_w
        w_neg /= total_w
        w_ftv /= total_w
        w_sam /= total_w

        threshold = trial.suggest_float('threshold', 0.25, 0.70)
        min_edge_weight = trial.suggest_float('min_edge_weight', 0.15, 0.60)
        contradiction_weight = trial.suggest_float('contradiction_weight', 0.50, 1.00)
        neutral_weight = trial.suggest_float('neutral_weight', 0.20, 0.70)
        entropy_max = trial.suggest_float('entropy_max', 5.0, 15.0)
        ncd_max = trial.suggest_float('ncd_max', 0.50, 1.00)

        predictions, ground_truths = [], []
        for q_id in question_ids:
            feats = features.get_features(q_id, min_edge_weight, contradiction_weight,
                                           neutral_weight, entropy_max, ncd_max,
                                           use_negation, use_ftv, use_sam)
            if feats is None:
                continue
            consistency, entropy, ncd, sv, negation, ftv, sam = feats
            inconsistency = 1.0 - consistency

            risk = (w_sv * sv + w_incon * inconsistency + w_ent * entropy +
                    w_ncd * ncd + w_mp * sam +  # Use SAM as minority if available, else basic
                    w_neg * negation + w_ftv * ftv + w_sam * sam)
            risk = min(max(risk, 0.0), 1.0)

            predictions.append(1 if risk > threshold else 0)
            ground_truths.append(labels[q_id])

        if len(predictions) < 10:
            return 0.0

        p = precision_score(ground_truths, predictions, zero_division=0)
        r = recall_score(ground_truths, predictions, zero_division=0)
        f1 = f1_score(ground_truths, predictions, zero_division=0)

        # Precision-balanced objective
        if p < 0.60:
            return f1 * (p / 0.60)
        return f1 * (1.0 + 0.15 * min(p - 0.60, 0.35))

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction='maximize',
        sampler=optuna.samplers.TPESampler(n_startup_trials=30, seed=42))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    # Evaluate best
    best = study.best_params
    w_sv = best['w_sv']
    w_incon = best['w_incon']
    w_ent = best['w_ent']
    w_ncd = best['w_ncd']
    w_mp = best['w_mp']
    w_neg = best.get('w_neg', 0.0)
    w_ftv = best.get('w_ftv', 0.0)
    w_sam = best.get('w_sam', 0.0)
    total_w = w_sv + w_incon + w_ent + w_ncd + w_mp + w_neg + w_ftv + w_sam
    w_sv /= total_w; w_incon /= total_w; w_ent /= total_w
    w_ncd /= total_w; w_mp /= total_w; w_neg /= total_w
    w_ftv /= total_w; w_sam /= total_w

    predictions, ground_truths = [], []
    for q_id in question_ids:
        feats = features.get_features(q_id, best['min_edge_weight'], best['contradiction_weight'],
                                       best['neutral_weight'], best['entropy_max'], best['ncd_max'],
                                       use_negation, use_ftv, use_sam)
        if feats is None:
            continue
        consistency, entropy, ncd, sv, negation, ftv, sam = feats
        inconsistency = 1.0 - consistency
        risk = (w_sv * sv + w_incon * inconsistency + w_ent * entropy +
                w_ncd * ncd + w_mp * sam + w_neg * negation + w_ftv * ftv + w_sam * sam)
        risk = min(max(risk, 0.0), 1.0)
        predictions.append(1 if risk > best['threshold'] else 0)
        ground_truths.append(labels[q_id])

    return {
        'precision': precision_score(ground_truths, predictions, zero_division=0),
        'recall': recall_score(ground_truths, predictions, zero_division=0),
        'f1': f1_score(ground_truths, predictions, zero_division=0),
        'threshold': best['threshold'],
        'best_value': study.best_value,
        'params': best,
    }


# ========================================
# MAIN
# ========================================
def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--trials', type=int, default=500)
    args = parser.parse_args()

    print("=" * 70)
    print("ABLATION STUDY: 3 New Signals x 8 Combinations")
    print("=" * 70)

    # Load data
    variants_data, original_responses, variant_responses, ground_truths = load_data()
    question_ids = list(set(original_responses.get('responses', {}).keys()) &
                        set(variant_responses.get('responses', {}).keys()) &
                        set(ground_truths.keys()))
    print(f"Questions: {len(question_ids)}")

    # Load models
    nli_model = NLIModel(DEVICE)
    sbert_model = SBERTModel(DEVICE)

    # Compute labels
    labels = compute_labels(question_ids, original_responses, variant_responses,
                            ground_truths, nli_model)

    # Compute ALL features
    features = FeatureComputer(question_ids, original_responses, variant_responses,
                                variants_data, ground_truths, nli_model, sbert_model)

    if DEVICE == 'cuda':
        torch.cuda.empty_cache()
        gc.collect()

    # Run 8 combinations
    combinations = list(product([False, True], repeat=3))
    combo_names = ['Negation', 'FactualTokenVar', 'SemanticMatch']

    results = []
    print(f"\n{'=' * 70}")
    print(f"Running {len(combinations)} combinations x {args.trials} trials each")
    print(f"{'=' * 70}\n")

    for combo in combinations:
        use_neg, use_ftv, use_sam = combo
        name_parts = []
        if use_neg: name_parts.append('Neg')
        if use_ftv: name_parts.append('FTV')
        if use_sam: name_parts.append('SAM')
        name = '+'.join(name_parts) if name_parts else 'Baseline'

        print(f"[{name}] Optimizing ({args.trials} trials)...", end=' ', flush=True)
        result = run_optimization(features, labels, question_ids,
                                   use_neg, use_ftv, use_sam, args.trials)
        result['name'] = name
        result['use_negation'] = use_neg
        result['use_ftv'] = use_ftv
        result['use_sam'] = use_sam
        results.append(result)

        print(f"P={result['precision']:.4f} R={result['recall']:.4f} F1={result['f1']:.4f}")

    # Sort by F1
    results.sort(key=lambda x: x['f1'], reverse=True)

    print(f"\n{'=' * 70}")
    print("ABLATION RESULTS (sorted by F1)")
    print(f"{'=' * 70}")
    print(f"{'Config':<25} {'Precision':>10} {'Recall':>10} {'F1':>10}")
    print("-" * 60)
    for r in results:
        print(f"{r['name']:<25} {r['precision']:>10.4f} {r['recall']:>10.4f} {r['f1']:>10.4f}")

    # Save
    output_path = OUTPUT_DIR / 'ablation_study_results.json'
    OUTPUT_DIR.mkdir(exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to {output_path}")


if __name__ == '__main__':
    main()
