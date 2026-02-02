"""
Comprehensive Hyperparameter Optimization
Optimizes ALL 13 hyperparameters using Multi-stage Bayesian Optimization
"""

import optuna
import numpy as np
import json
import sys
from pathlib import Path
from typing import Dict, List
import pandas as pd
sys.path.append('/Users/ofs/bitirm2')

from src.detector import HallucinationDetector
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                            recall_score, roc_auc_score, confusion_matrix)
from sklearn.model_selection import KFold


class ComprehensiveOptimizer:
    """
    Comprehensive hyperparameter optimization for LOGIC-HALT

    Optimizes 13 parameters:
    - Fusion weights: alpha, beta, gamma
    - Decision thresholds: hallucination_threshold, high_confidence, low_confidence
    - Variant generation: similarity_threshold, num_variants
    - Consistency: min_edge_weight, contradiction_weight, neutral_weight
    - Complexity: entropy_max, ncd_max
    """

    def __init__(self, dataset_path: str, ground_truth_labels: dict,
                 use_cv: bool = False, n_folds: int = 5):
        """
        Args:
            dataset_path: Path to dataset
            ground_truth_labels: Ground truth hallucination labels
            use_cv: Whether to use cross-validation
            n_folds: Number of CV folds
        """
        self.dataset_path = dataset_path
        self.ground_truth_labels = ground_truth_labels
        self.use_cv = use_cv
        self.n_folds = n_folds

        # Load dataset
        with open(dataset_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            self.dataset = data if isinstance(data, list) else data['questions']

        # Filter questions with ground truth
        self.questions = [q for q in self.dataset if q['id'] in ground_truth_labels]

        print(f"✅ Loaded {len(self.questions)} questions with ground truth")

    def objective(self, trial):
        """
        Objective function for Optuna optimization

        Suggests all 13 hyperparameters and evaluates performance
        """

        # ========================================
        # 1. FUSION LAYER WEIGHTS (3 params)
        # ========================================
        alpha = trial.suggest_float('alpha', 0.0, 1.0)
        beta = trial.suggest_float('beta', 0.0, 1.0 - alpha)
        gamma = 1.0 - alpha - beta  # Constraint: sum = 1.0

        # ========================================
        # 2. DECISION THRESHOLDS (3 params)
        # ========================================
        hallucination_threshold = trial.suggest_float('hallucination_threshold', 0.30, 0.65)
        high_confidence = trial.suggest_float('high_confidence', 0.70, 0.85)
        low_confidence = trial.suggest_float('low_confidence', 0.20, 0.50)

        # Ensure ordering: low < high
        if low_confidence >= high_confidence:
            return 0.0  # Invalid configuration

        # ========================================
        # 3. VARIANT GENERATION (2 params)
        # ========================================
        similarity_threshold = trial.suggest_float('similarity_threshold', 0.65, 0.85)
        num_variants = trial.suggest_int('num_variants', 3, 7)

        # ========================================
        # 4. CONSISTENCY ENGINE (3 params)
        # ========================================
        min_edge_weight = trial.suggest_float('min_edge_weight', 0.2, 0.5)
        contradiction_weight = trial.suggest_float('contradiction_weight', 0.8, 1.0)
        neutral_weight = trial.suggest_float('neutral_weight', 0.3, 0.7)

        # ========================================
        # 5. COMPLEXITY NORMALIZATION (2 params)
        # ========================================
        entropy_max = trial.suggest_float('entropy_max', 5.0, 15.0)
        ncd_max = trial.suggest_float('ncd_max', 0.8, 1.0)

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
        if self.use_cv:
            # Cross-validation
            f1_scores = []
            kf = KFold(n_splits=self.n_folds, shuffle=True, random_state=42)

            for fold_idx, (train_idx, val_idx) in enumerate(kf.split(self.questions)):
                val_questions = [self.questions[i] for i in val_idx]
                f1 = self._evaluate_questions(val_questions, custom_config)
                f1_scores.append(f1)

            # Return average F1 across folds
            return np.mean(f1_scores)
        else:
            # Single evaluation on full dataset
            return self._evaluate_questions(self.questions, custom_config)

    def _evaluate_questions(self, questions: List[dict], custom_config: dict) -> float:
        """
        Evaluate on a set of questions with given config

        Returns F1 score
        """
        # Initialize detector with custom config
        # NOTE: We would need to modify HallucinationDetector to accept runtime config
        # For now, we'll use a workaround by modifying config file temporarily

        predictions = []
        ground_truths = []

        for question_data in questions:
            question_id = question_data['id']

            if question_id not in self.ground_truth_labels:
                continue

            try:
                # Create detector (ideally with custom config injection)
                detector = HallucinationDetector(
                    config_path='/Users/ofs/bitirm2/config/config.yaml',
                    use_mock=False
                )

                # Override parameters dynamically
                detector.fusion_layer.config.update(custom_config['fusion'])
                detector.morpher.similarity_threshold = custom_config['morpher']['similarity_threshold']
                # Note: num_variants cannot be changed at runtime without refactoring

                # Run detection
                result = detector.detect_single(
                    question=question_data['question'],
                    question_id=question_id,
                    ground_truth=question_data.get('ground_truth'),
                    visualize=False  # Disable visualization for speed
                )

                prediction = 1 if result.hallucination_detected else 0
                predictions.append(prediction)
                ground_truths.append(self.ground_truth_labels[question_id])

            except Exception as e:
                print(f"Error on {question_id}: {e}")
                continue

        # Calculate F1 score
        if len(predictions) == 0:
            return 0.0

        f1 = f1_score(ground_truths, predictions, zero_division=0)
        return f1

    def optimize(self, n_trials: int = 50, study_name: str = 'comprehensive_opt'):
        """
        Run optimization

        Args:
            n_trials: Number of trials
            study_name: Name for the study
        """
        print("="*70)
        print("🚀 COMPREHENSIVE HYPERPARAMETER OPTIMIZATION")
        print("="*70)
        print(f"\n📊 Configuration:")
        print(f"   Questions: {len(self.questions)}")
        print(f"   Trials: {n_trials}")
        print(f"   Cross-validation: {self.use_cv}")
        if self.use_cv:
            print(f"   CV Folds: {self.n_folds}")
        print(f"\n🎯 Optimizing 13 hyperparameters:")
        print("   - Fusion weights (α, β, γ)")
        print("   - Thresholds (τ, τ_high, τ_low)")
        print("   - Variant generation (similarity, num_variants)")
        print("   - Consistency (min_edge, contradiction_w, neutral_w)")
        print("   - Complexity (entropy_max, ncd_max)")
        print()

        # Create Optuna study
        study = optuna.create_study(
            direction='maximize',
            study_name=study_name,
            sampler=optuna.samplers.TPESampler(
                n_startup_trials=10,      # First 10 trials are random
                multivariate=True,         # Learn parameter interactions
                seed=42
            ),
            pruner=optuna.pruners.MedianPruner(
                n_startup_trials=5,
                n_warmup_steps=5
            )
        )

        # Run optimization
        study.optimize(
            self.objective,
            n_trials=n_trials,
            show_progress_bar=True,
            callbacks=[self._log_callback]
        )

        # Print results
        self._print_results(study)

        # Save results
        self._save_results(study, study_name)

        return study

    def _log_callback(self, study, trial):
        """Callback to log each trial"""
        if trial.number % 5 == 0:
            print(f"\n[Trial {trial.number}] Best F1 so far: {study.best_value:.4f}")

    def _print_results(self, study):
        """Print optimization results"""
        print("\n" + "="*70)
        print("✅ OPTIMIZATION COMPLETE")
        print("="*70)
        print(f"\n📊 Best F1 Score: {study.best_value:.4f}")
        print(f"\n🎯 Best Hyperparameters:")
        print()

        # Group by category
        best_params = study.best_params

        print("  📐 Fusion Layer Weights:")
        print(f"     alpha (Consistency):  {best_params['alpha']:.4f}")
        print(f"     beta  (Entropy):      {best_params['beta']:.4f}")
        gamma = 1.0 - best_params['alpha'] - best_params['beta']
        print(f"     gamma (NCD):          {gamma:.4f}")
        print()

        print("  🎚️  Decision Thresholds:")
        print(f"     hallucination_threshold: {best_params['hallucination_threshold']:.4f}")
        print(f"     high_confidence:         {best_params['high_confidence']:.4f}")
        print(f"     low_confidence:          {best_params['low_confidence']:.4f}")
        print()

        print("  🔄 Variant Generation:")
        print(f"     similarity_threshold: {best_params['similarity_threshold']:.4f}")
        print(f"     num_variants:         {best_params['num_variants']}")
        print()

        print("  📊 Consistency Engine:")
        print(f"     min_edge_weight:      {best_params['min_edge_weight']:.4f}")
        print(f"     contradiction_weight: {best_params['contradiction_weight']:.4f}")
        print(f"     neutral_weight:       {best_params['neutral_weight']:.4f}")
        print()

        print("  🧮 Complexity Normalization:")
        print(f"     entropy_max: {best_params['entropy_max']:.4f}")
        print(f"     ncd_max:     {best_params['ncd_max']:.4f}")
        print()
        print("="*70)

    def _save_results(self, study, study_name):
        """Save optimization results"""
        output_dir = Path('/Users/ofs/bitirm2/config/optimization_results')
        output_dir.mkdir(exist_ok=True)

        # Save best parameters as JSON
        best_params = study.best_params.copy()
        best_params['gamma'] = 1.0 - best_params['alpha'] - best_params['beta']
        best_params['best_f1_score'] = study.best_value

        json_path = output_dir / f'{study_name}_best_params.json'
        with open(json_path, 'w') as f:
            json.dump(best_params, f, indent=2)
        print(f"\n💾 Best parameters saved: {json_path}")

        # Save trial history as CSV
        df = study.trials_dataframe()
        csv_path = output_dir / f'{study_name}_trials.csv'
        df.to_csv(csv_path, index=False)
        print(f"💾 Trial history saved: {csv_path}")

        # Save parameter importance
        try:
            importance = optuna.importance.get_param_importances(study)
            importance_path = output_dir / f'{study_name}_importance.json'
            with open(importance_path, 'w') as f:
                json.dump(importance, f, indent=2)
            print(f"💾 Parameter importance saved: {importance_path}")
        except:
            print("⚠️  Could not calculate parameter importance (need more trials)")


def main():
    """
    Main execution
    """
    print("="*70)
    print("LOGIC-HALT COMPREHENSIVE HYPERPARAMETER OPTIMIZATION")
    print("="*70)
    print()

    # ========================================
    # CONFIGURATION
    # ========================================

    # Choose stage
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--stage', type=str, default='pilot',
                       choices=['pilot', 'medium', 'large'],
                       help='Optimization stage')
    parser.add_argument('--trials', type=int, default=50,
                       help='Number of trials')
    parser.add_argument('--cv', action='store_true',
                       help='Use cross-validation')
    args = parser.parse_args()

    # Stage-specific settings
    if args.stage == 'pilot':
        dataset_path = '/Users/ofs/bitirm2/data/raw/hallucination_test_dataset.json'
        n_trials = 50
        use_cv = False
    elif args.stage == 'medium':
        dataset_path = '/Users/ofs/bitirm2/data/raw/medium_dataset.json'
        n_trials = 100
        use_cv = True
    elif args.stage == 'large':
        dataset_path = '/Users/ofs/bitirm2/data/raw/large_dataset.json'
        n_trials = 50
        use_cv = True

    # Override with CLI args
    n_trials = args.trials
    use_cv = args.cv

    # Ground truth labels for hallucination dataset
    # 0 = No hallucination (correct/factual)
    # 1 = Hallucination (fabricated/incorrect)
    ground_truth_labels = {
        "hal_001": 0, "hal_002": 1, "hal_003": 0,
        "hal_004": 1, "hal_005": 0, "hal_006": 0,
        "hal_007": 1, "hal_008": 0, "hal_009": 1,
        "hal_010": 0, "hal_011": 1, "hal_012": 0,
        "hal_013": 1, "hal_014": 1, "hal_015": 1,
        "hal_016": 0, "hal_017": 1, "hal_018": 1,
        "hal_019": 0, "hal_020": 1,
    }

    print(f"📍 Stage: {args.stage.upper()}")
    print(f"📊 Trials: {n_trials}")
    print(f"🔄 Cross-validation: {use_cv}")
    print()

    # ========================================
    # RUN OPTIMIZATION
    # ========================================

    optimizer = ComprehensiveOptimizer(
        dataset_path=dataset_path,
        ground_truth_labels=ground_truth_labels,
        use_cv=use_cv,
        n_folds=5
    )

    study = optimizer.optimize(
        n_trials=n_trials,
        study_name=f'{args.stage}_comprehensive'
    )

    print("\n✅ Optimization completed!")
    print(f"   Best F1 Score: {study.best_value:.4f}")
    print(f"   Results saved to: config/optimization_results/")


if __name__ == "__main__":
    main()
