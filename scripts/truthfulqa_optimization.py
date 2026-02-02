"""
TruthfulQA Bayesian Hyperparameter Optimization
RTX 5070 (12GB VRAM) + Ryzen 5 7500F + 32GB DDR5 için optimize edildi
"""

import optuna
import numpy as np
import json
import sys
import os
import yaml
import torch
import gc
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from datetime import datetime
from tqdm import tqdm

sys.path.append(str(Path(__file__).parent.parent))

from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                            recall_score, roc_auc_score, confusion_matrix)
from sklearn.model_selection import KFold

# ========================================
# CUDA OPTIMIZATION FOR RTX 5070
# ========================================
def setup_cuda():
    """Setup CUDA optimizations for RTX 5070"""
    if not torch.cuda.is_available():
        print("⚠️  CUDA not available, using CPU")
        return 'cpu'

    # GPU info
    gpu_name = torch.cuda.get_device_name(0)
    gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"\n🚀 GPU: {gpu_name}")
    print(f"   VRAM: {gpu_mem:.1f} GB")
    print(f"   CUDA: {torch.version.cuda}")

    # CUDA optimizations
    torch.backends.cudnn.benchmark = True  # Auto-tune convolutions
    torch.backends.cuda.matmul.allow_tf32 = True  # Tensor core acceleration
    torch.backends.cudnn.allow_tf32 = True

    # Memory optimization
    torch.cuda.empty_cache()

    print("   ✅ CUDA optimizations enabled (cudnn.benchmark, TF32)")

    return 'cuda'

DEVICE = setup_cuda()

# ========================================
# CONFIGURATION
# ========================================
CONFIG_PATH = Path(__file__).parent.parent / 'config' / 'config.yaml'

def load_config():
    """Load configuration from YAML"""
    with open(CONFIG_PATH, 'r') as f:
        return yaml.safe_load(f)

CONFIG = load_config()


@dataclass
class EvaluationResult:
    """Stores evaluation metrics"""
    accuracy: float
    f1: float
    precision: float
    recall: float
    predictions: List[int]
    ground_truths: List[int]


class TruthfulQAOptimizer:
    """
    Bayesian optimization for LOGIC-HALT using TruthfulQA dataset
    Optimized for RTX 5070 (12GB VRAM)

    Optimizes 13 hyperparameters:
    - Fusion weights: alpha, beta, gamma
    - Decision thresholds: hallucination_threshold, high_confidence, low_confidence
    - Variant generation: similarity_threshold, num_variants
    - Consistency: min_edge_weight, contradiction_weight, neutral_weight
    - Complexity: entropy_max, ncd_max
    """

    def __init__(self,
                 dataset_path: str = None,
                 sample_size: int = None,
                 use_cv: bool = True,
                 n_folds: int = 5,
                 seed: int = 42):
        """
        Args:
            dataset_path: Path to TruthfulQA dataset
            sample_size: Number of questions to use (None = all)
            use_cv: Whether to use cross-validation
            n_folds: Number of CV folds
            seed: Random seed
        """
        if dataset_path is None:
            dataset_path = Path(__file__).parent.parent / 'data' / 'raw' / 'truthfulqa_dataset.json'

        self.dataset_path = Path(dataset_path)
        self.sample_size = sample_size
        self.use_cv = use_cv
        self.n_folds = n_folds
        self.seed = seed
        self.device = DEVICE

        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)

        # Load dataset
        self._load_dataset()

        # Pre-load models for GPU (avoid repeated loading)
        self._preload_models()

    def _load_dataset(self):
        """Load and prepare TruthfulQA dataset"""
        with open(self.dataset_path, 'r', encoding='utf-8') as f:
            self.full_dataset = json.load(f)

        print(f"\n📊 Loaded {len(self.full_dataset)} TruthfulQA questions")

        # Sample if needed
        if self.sample_size and self.sample_size < len(self.full_dataset):
            indices = np.random.choice(len(self.full_dataset), self.sample_size, replace=False)
            self.dataset = [self.full_dataset[i] for i in indices]
            print(f"   Sampled: {self.sample_size} questions")
        else:
            self.dataset = self.full_dataset

    def _preload_models(self):
        """Pre-load models to GPU for faster inference"""
        print("\n⏳ Pre-loading models to GPU...")

        from sentence_transformers import SentenceTransformer

        # Sentence embedder (for answer comparison)
        self.embedder = SentenceTransformer(
            'all-MiniLM-L6-v2',
            device=self.device
        )
        print(f"   ✅ SentenceTransformer on {self.device}")

        # Pre-compute ground truth embeddings for all questions
        print("   ⏳ Pre-computing ground truth embeddings...")
        ground_truths = [q['ground_truth'] for q in self.dataset]
        self.gt_embeddings = self.embedder.encode(
            ground_truths,
            batch_size=256,  # RTX 5070 can handle large batches
            show_progress_bar=True,
            convert_to_numpy=True
        )
        print(f"   ✅ Pre-computed {len(self.gt_embeddings)} embeddings")

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _get_detector(self, custom_config: dict = None):
        """Get detector with custom config"""
        from src.detector import HallucinationDetector

        detector = HallucinationDetector(
            config_path=str(CONFIG_PATH),
            use_mock=False
        )

        if custom_config:
            # Override fusion parameters
            if 'fusion' in custom_config:
                for key, value in custom_config['fusion'].items():
                    if hasattr(detector.fusion_layer, 'config'):
                        detector.fusion_layer.config[key] = value
                    if hasattr(detector.fusion_layer, key):
                        setattr(detector.fusion_layer, key, value)

            # Override morpher parameters
            if 'morpher' in custom_config:
                if hasattr(detector.morpher, 'similarity_threshold'):
                    detector.morpher.similarity_threshold = custom_config['morpher'].get(
                        'similarity_threshold', 0.75
                    )

            # Override consistency parameters
            if 'consistency' in custom_config:
                for key, value in custom_config['consistency'].items():
                    if hasattr(detector.consistency_engine, key):
                        setattr(detector.consistency_engine, key, value)

        return detector

    def objective(self, trial: optuna.Trial) -> float:
        """Objective function for Optuna optimization"""

        # ========================================
        # 1. FUSION LAYER WEIGHTS
        # ========================================
        alpha = trial.suggest_float('alpha', 0.05, 0.50)
        beta = trial.suggest_float('beta', 0.25, 0.65)
        gamma = 1.0 - alpha - beta

        if gamma < 0.10 or gamma > 0.55:
            return 0.0

        # ========================================
        # 2. DECISION THRESHOLDS
        # ========================================
        hallucination_threshold = trial.suggest_float('hallucination_threshold', 0.40, 0.75)
        high_confidence = trial.suggest_float('high_confidence', 0.65, 0.90)
        low_confidence = trial.suggest_float('low_confidence', 0.15, 0.40)

        if low_confidence >= high_confidence - 0.15:
            return 0.0

        # ========================================
        # 3. VARIANT GENERATION
        # ========================================
        similarity_threshold = trial.suggest_float('similarity_threshold', 0.60, 0.88)
        num_variants = trial.suggest_int('num_variants', 3, 7)

        # ========================================
        # 4. CONSISTENCY ENGINE
        # ========================================
        min_edge_weight = trial.suggest_float('min_edge_weight', 0.20, 0.50)
        contradiction_weight = trial.suggest_float('contradiction_weight', 0.80, 1.0)
        neutral_weight = trial.suggest_float('neutral_weight', 0.30, 0.70)

        # ========================================
        # 5. COMPLEXITY NORMALIZATION
        # ========================================
        entropy_max = trial.suggest_float('entropy_max', 5.0, 15.0)
        ncd_max = trial.suggest_float('ncd_max', 0.75, 1.0)

        # ========================================
        # BUILD CONFIG
        # ========================================
        custom_config = {
            'fusion': {
                'alpha': alpha,
                'beta': beta,
                'gamma': gamma,
                'hallucination_threshold': hallucination_threshold,
                'high_confidence': high_confidence,
                'low_confidence': low_confidence
            },
            'morpher': {
                'similarity_threshold': similarity_threshold,
                'num_variants': num_variants
            },
            'consistency': {
                'min_edge_weight': min_edge_weight,
                'contradiction_weight': contradiction_weight,
                'neutral_weight': neutral_weight
            },
            'complexity': {
                'entropy_bounds': [0.0, entropy_max],
                'ncd_bounds': [0.0, ncd_max]
            }
        }

        # ========================================
        # EVALUATE
        # ========================================
        try:
            if self.use_cv:
                return self._evaluate_cv(custom_config)
            else:
                return self._evaluate_full(custom_config)
        except Exception as e:
            print(f"⚠️ Trial {trial.number} failed: {e}")
            # Clear GPU memory on error
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            return 0.0

    def _evaluate_cv(self, custom_config: dict) -> float:
        """Cross-validation evaluation"""
        kf = KFold(n_splits=self.n_folds, shuffle=True, random_state=self.seed)
        f1_scores = []

        for fold_idx, (train_idx, val_idx) in enumerate(kf.split(self.dataset)):
            val_questions = [self.dataset[i] for i in val_idx]
            val_gt_embeddings = self.gt_embeddings[val_idx]

            result = self._evaluate_questions(val_questions, custom_config, val_gt_embeddings)
            f1_scores.append(result.f1)

        return np.mean(f1_scores)

    def _evaluate_full(self, custom_config: dict) -> float:
        """Full dataset evaluation"""
        result = self._evaluate_questions(self.dataset, custom_config, self.gt_embeddings)
        return result.f1

    def _evaluate_questions(self,
                           questions: List[dict],
                           custom_config: dict,
                           gt_embeddings: np.ndarray) -> EvaluationResult:
        """
        Evaluate detector on questions (GPU accelerated)

        HYBRID FORMAT: Extracts ANSWER portion from LLM response for
        ground truth comparison. This aligns with how NLI comparison
        works in the Consistency Engine.
        """
        from sklearn.metrics.pairwise import cosine_similarity
        from src.consistency import extract_answer

        detector = self._get_detector(custom_config)

        predictions = []
        ground_truths = []

        for idx, q in enumerate(questions):
            try:
                # Run detection
                result = detector.detect_single(
                    question=q['question'],
                    question_id=q['id'],
                    ground_truth=q['ground_truth'],
                    visualize=False
                )

                # Get LLM's response
                if hasattr(result, 'responses') and result.responses:
                    # Get primary response text
                    if hasattr(result.responses[0], 'response_text'):
                        llm_response = result.responses[0].response_text
                    else:
                        llm_response = str(result.responses[0])
                else:
                    llm_response = ""

                # HYBRID FORMAT: Extract only ANSWER portion for comparison
                answer_portion = extract_answer(llm_response)

                # Compare extracted ANSWER with ground truth using embeddings
                response_embedding = self.embedder.encode([answer_portion], convert_to_numpy=True)
                similarity = cosine_similarity(response_embedding, [gt_embeddings[idx]])[0][0]

                # Ground truth: similar to correct answer = no hallucination
                # Threshold 0.55 for semantic similarity
                actual_hallucination = 0 if similarity >= 0.55 else 1

                # Prediction from detector
                predicted_hallucination = 1 if result.hallucination_detected else 0

                predictions.append(predicted_hallucination)
                ground_truths.append(actual_hallucination)

            except Exception as e:
                continue

        if len(predictions) == 0:
            return EvaluationResult(0, 0, 0, 0, [], [])

        return EvaluationResult(
            accuracy=accuracy_score(ground_truths, predictions),
            f1=f1_score(ground_truths, predictions, zero_division=0),
            precision=precision_score(ground_truths, predictions, zero_division=0),
            recall=recall_score(ground_truths, predictions, zero_division=0),
            predictions=predictions,
            ground_truths=ground_truths
        )

    def optimize(self,
                 n_trials: int = 100,
                 study_name: str = 'truthfulqa_rtx5070',
                 timeout: int = None) -> optuna.Study:
        """Run Bayesian optimization"""

        print("\n" + "="*70)
        print("🚀 TRUTHFULQA BAYESIAN OPTIMIZATION (RTX 5070)")
        print("="*70)
        print(f"\n📊 Configuration:")
        print(f"   Dataset: {len(self.dataset)} questions")
        print(f"   Trials: {n_trials}")
        print(f"   CV Folds: {self.n_folds}")
        print(f"   Device: {self.device}")
        if torch.cuda.is_available():
            print(f"   GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        print(f"\n🎯 Optimizing 13 hyperparameters...")

        # Create study
        study = optuna.create_study(
            direction='maximize',
            study_name=study_name,
            sampler=optuna.samplers.TPESampler(
                n_startup_trials=15,
                multivariate=True,
                seed=self.seed
            ),
            pruner=optuna.pruners.MedianPruner(
                n_startup_trials=10,
                n_warmup_steps=5
            )
        )

        # Progress callback
        pbar = tqdm(total=n_trials, desc="Optimization")

        def callback(study, trial):
            pbar.update(1)
            pbar.set_postfix({'best_f1': f'{study.best_value:.4f}'})

            # Periodic GPU memory cleanup
            if trial.number % 10 == 0 and torch.cuda.is_available():
                torch.cuda.empty_cache()
                gc.collect()

        # Run optimization
        study.optimize(
            self.objective,
            n_trials=n_trials,
            timeout=timeout,
            callbacks=[callback],
            show_progress_bar=False
        )

        pbar.close()

        # Results
        self._print_results(study)
        self._save_results(study, study_name)

        return study

    def _print_results(self, study: optuna.Study):
        """Print optimization results"""
        print("\n" + "="*70)
        print("✅ OPTIMIZATION COMPLETE")
        print("="*70)

        best = study.best_params
        gamma = 1.0 - best['alpha'] - best['beta']

        print(f"\n📊 Best F1 Score: {study.best_value:.4f}")
        print(f"   Total trials: {len(study.trials)}")
        print(f"\n🎯 Optimal Hyperparameters:\n")

        print("  ┌─────────────────────────────────────────┐")
        print("  │ FUSION WEIGHTS                          │")
        print(f"  │   α (Consistency): {best['alpha']:>20.4f} │")
        print(f"  │   β (Entropy):     {best['beta']:>20.4f} │")
        print(f"  │   γ (NCD):         {gamma:>20.4f} │")
        print("  ├─────────────────────────────────────────┤")
        print("  │ DECISION THRESHOLDS                     │")
        print(f"  │   hallucination_threshold: {best['hallucination_threshold']:>12.4f} │")
        print(f"  │   high_confidence:         {best['high_confidence']:>12.4f} │")
        print(f"  │   low_confidence:          {best['low_confidence']:>12.4f} │")
        print("  ├─────────────────────────────────────────┤")
        print("  │ VARIANT GENERATION                      │")
        print(f"  │   similarity_threshold: {best['similarity_threshold']:>15.4f} │")
        print(f"  │   num_variants:         {best['num_variants']:>15d} │")
        print("  ├─────────────────────────────────────────┤")
        print("  │ CONSISTENCY ENGINE                      │")
        print(f"  │   min_edge_weight:      {best['min_edge_weight']:>15.4f} │")
        print(f"  │   contradiction_weight: {best['contradiction_weight']:>15.4f} │")
        print(f"  │   neutral_weight:       {best['neutral_weight']:>15.4f} │")
        print("  ├─────────────────────────────────────────┤")
        print("  │ COMPLEXITY NORMALIZATION                │")
        print(f"  │   entropy_max: {best['entropy_max']:>24.4f} │")
        print(f"  │   ncd_max:     {best['ncd_max']:>24.4f} │")
        print("  └─────────────────────────────────────────┘")

    def _save_results(self, study: optuna.Study, study_name: str):
        """Save optimization results"""
        output_dir = Path(__file__).parent.parent / 'config' / 'optimization_results'
        output_dir.mkdir(exist_ok=True)

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

        # Save best parameters
        best_params = study.best_params.copy()
        best_params['gamma'] = 1.0 - best_params['alpha'] - best_params['beta']
        best_params['best_f1_score'] = study.best_value
        best_params['n_trials'] = len(study.trials)
        best_params['timestamp'] = timestamp
        best_params['device'] = self.device
        best_params['dataset_size'] = len(self.dataset)

        json_path = output_dir / f'{study_name}_best_params.json'
        with open(json_path, 'w') as f:
            json.dump(best_params, f, indent=2)
        print(f"\n💾 Best parameters: {json_path}")

        # Save trial history
        df = study.trials_dataframe()
        csv_path = output_dir / f'{study_name}_trials.csv'
        df.to_csv(csv_path, index=False)
        print(f"💾 Trial history: {csv_path}")

        # Save parameter importance
        try:
            importance = optuna.importance.get_param_importances(study)
            imp_path = output_dir / f'{study_name}_importance.json'
            with open(imp_path, 'w') as f:
                json.dump(importance, f, indent=2)
            print(f"💾 Parameter importance: {imp_path}")

            print("\n📊 Top 5 Important Parameters:")
            for i, (param, imp) in enumerate(sorted(importance.items(), key=lambda x: -x[1])[:5], 1):
                print(f"   {i}. {param}: {imp:.4f}")
        except Exception as e:
            print(f"⚠️ Could not calculate importance: {e}")


def main():
    """Main entry point"""
    import argparse

    parser = argparse.ArgumentParser(description='TruthfulQA Bayesian Optimization (RTX 5070)')
    parser.add_argument('--trials', type=int, default=100, help='Number of trials')
    parser.add_argument('--sample', type=int, default=None, help='Sample size (None=all 817)')
    parser.add_argument('--cv', action='store_true', default=True, help='Use cross-validation')
    parser.add_argument('--folds', type=int, default=5, help='CV folds')
    parser.add_argument('--timeout', type=int, default=None, help='Timeout in seconds')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--name', type=str, default='truthfulqa_rtx5070', help='Study name')
    args = parser.parse_args()

    print("="*70)
    print("🎯 LOGIC-HALT × TruthfulQA Bayesian Optimization")
    print("   Optimized for RTX 5070 (12GB VRAM)")
    print("="*70)

    # GPU warmup
    if torch.cuda.is_available():
        print("\n⏳ GPU warmup...")
        _ = torch.zeros(1000, 1000, device='cuda')
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        print("   ✅ GPU ready")

    optimizer = TruthfulQAOptimizer(
        sample_size=args.sample,
        use_cv=args.cv,
        n_folds=args.folds,
        seed=args.seed
    )

    study = optimizer.optimize(
        n_trials=args.trials,
        study_name=args.name,
        timeout=args.timeout
    )

    print("\n" + "="*70)
    print("✅ Optimization finished!")
    print(f"   Best F1: {study.best_value:.4f}")
    print(f"   Results: config/optimization_results/")
    print("="*70)

    # Final GPU memory report
    if torch.cuda.is_available():
        mem_used = torch.cuda.max_memory_allocated() / 1e9
        mem_total = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"\n📊 GPU Memory: {mem_used:.2f} / {mem_total:.1f} GB used")


if __name__ == "__main__":
    main()
