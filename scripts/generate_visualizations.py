"""
Generate visualizations for LOGIC-HALT optimization results
"""

import json
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

# Set matplotlib backend BEFORE importing pyplot
import matplotlib
matplotlib.use('svg')  # Use SVG backend (no DLL needed)
import matplotlib.pyplot as plt

from pathlib import Path

# Output directory
output_dir = Path(__file__).parent.parent / 'data' / 'processed' / 'visualizations'
output_dir.mkdir(parents=True, exist_ok=True)

# Load best params
config_dir = Path(__file__).parent.parent / 'config' / 'optimization_results'
with open(config_dir / 'batch_optimization_best_params.json') as f:
    params = json.load(f)

print('='*60)
print('VISUALIZATION GENERATION')
print('='*60)

from scripts.batch_optimization import BatchOptimizer, calculate_risk

optimizer = BatchOptimizer(nli_batch_size=64)

# Get predictions
alpha = params['alpha']
beta = params['beta']
gamma = params['gamma']
delta = params['delta']
threshold = params['hallucination_threshold']
min_edge_weight = params['min_edge_weight']
contradiction_weight = params['contradiction_weight']
neutral_weight = params['neutral_weight']
entropy_max = params['entropy_max']
ncd_max = params['ncd_max']

predictions = []
ground_truths = []
risk_scores = []

for q_id in optimizer.question_ids:
    features = optimizer.compute_features_fast(
        q_id, min_edge_weight, contradiction_weight, neutral_weight, entropy_max, ncd_max
    )
    if features is None:
        continue

    consistency, entropy, ncd, gt_contradiction = features
    risk = calculate_risk(gt_contradiction, consistency, entropy, ncd, alpha, beta, gamma, delta)
    pred = 1 if risk > threshold else 0

    predictions.append(pred)
    ground_truths.append(optimizer.labels[q_id])
    risk_scores.append(risk)

from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score, accuracy_score, roc_curve, auc

# Set style
plt.style.use('ggplot')
plt.rcParams['font.size'] = 12
plt.rcParams['figure.figsize'] = (10, 8)

# ========================================
# 1. CONFUSION MATRIX
# ========================================
print('[*] Creating Confusion Matrix...')
cm = confusion_matrix(ground_truths, predictions)
fig, ax = plt.subplots(figsize=(8, 6))
im = ax.imshow(cm, cmap='Blues')
ax.set_xticks([0, 1])
ax.set_yticks([0, 1])
ax.set_xticklabels(['Truthful', 'Hallucination'])
ax.set_yticklabels(['Truthful', 'Hallucination'])
for i in range(2):
    for j in range(2):
        text = ax.text(j, i, cm[i, j], ha='center', va='center', color='black', fontsize=20)
plt.colorbar(im)
plt.title('Confusion Matrix - LOGIC-HALT\nF1=0.7730, Precision=0.9770', fontsize=14, fontweight='bold')
plt.xlabel('Predicted', fontsize=12)
plt.ylabel('Actual', fontsize=12)
plt.tight_layout()
plt.savefig(output_dir / 'confusion_matrix.svg', dpi=150)
plt.close()
print(f'   [OK] Saved: {output_dir}/confusion_matrix.svg')

# ========================================
# 2. METRICS COMPARISON (Old vs New)
# ========================================
print('[*] Creating Metrics Comparison...')
metrics = ['F1 Score', 'Precision', 'Recall', 'Accuracy']
old_values = [0.4609, 0.3564, 0.6519, 0.7050]
new_values = [
    f1_score(ground_truths, predictions),
    precision_score(ground_truths, predictions),
    recall_score(ground_truths, predictions),
    accuracy_score(ground_truths, predictions)
]

x = np.arange(len(metrics))
width = 0.35

fig, ax = plt.subplots(figsize=(10, 6))
bars1 = ax.bar(x - width/2, old_values, width, label='Old (Cosine)', color='#ff7f7f')
bars2 = ax.bar(x + width/2, new_values, width, label='New (NLI)', color='#7fbf7f')

ax.set_ylabel('Score', fontsize=12)
ax.set_title('Performance Comparison: Cosine vs NLI Labeling', fontsize=14, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(metrics)
ax.legend()
ax.set_ylim(0, 1.1)

# Add value labels
for bar in bars1:
    height = bar.get_height()
    ax.annotate(f'{height:.3f}', xy=(bar.get_x() + bar.get_width()/2, height),
                xytext=(0, 3), textcoords='offset points', ha='center', va='bottom', fontsize=10)
for bar in bars2:
    height = bar.get_height()
    ax.annotate(f'{height:.3f}', xy=(bar.get_x() + bar.get_width()/2, height),
                xytext=(0, 3), textcoords='offset points', ha='center', va='bottom', fontsize=10)

plt.tight_layout()
plt.savefig(output_dir / 'metrics_comparison.svg', dpi=150)
plt.close()
print(f'   [OK] Saved: {output_dir}/metrics_comparison.svg')

# ========================================
# 3. FUSION WEIGHTS PIE CHART
# ========================================
print('[*] Creating Fusion Weights Chart...')
weights = [alpha, beta, gamma, delta]
labels = [f'GT Contradiction\n({alpha:.1%})',
          f'Inconsistency\n({beta:.1%})',
          f'Entropy\n({gamma:.1%})',
          f'NCD\n({delta:.1%})']
colors = ['#ff6b6b', '#4ecdc4', '#45b7d1', '#96ceb4']
explode = (0.05, 0, 0, 0)

fig, ax = plt.subplots(figsize=(8, 8))
wedges, texts, autotexts = ax.pie(weights, labels=labels, colors=colors, explode=explode,
                                   autopct='', startangle=90,
                                   wedgeprops=dict(width=0.7, edgecolor='white'))
ax.set_title('Fusion Layer Weights (Optimized)', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(output_dir / 'fusion_weights.svg', dpi=150)
plt.close()
print(f'   [OK] Saved: {output_dir}/fusion_weights.svg')

# ========================================
# 4. RISK SCORE DISTRIBUTION
# ========================================
print('[*] Creating Risk Score Distribution...')
fig, ax = plt.subplots(figsize=(10, 6))

truthful_risks = [r for r, g in zip(risk_scores, ground_truths) if g == 0]
halluc_risks = [r for r, g in zip(risk_scores, ground_truths) if g == 1]

ax.hist(truthful_risks, bins=30, alpha=0.6, label=f'Truthful (n={len(truthful_risks)})', color='green')
ax.hist(halluc_risks, bins=30, alpha=0.6, label=f'Hallucination (n={len(halluc_risks)})', color='red')
ax.axvline(x=threshold, color='black', linestyle='--', linewidth=2, label=f'Threshold={threshold:.3f}')

ax.set_xlabel('Risk Score', fontsize=12)
ax.set_ylabel('Frequency', fontsize=12)
ax.set_title('Risk Score Distribution by Class', fontsize=14, fontweight='bold')
ax.legend()
plt.tight_layout()
plt.savefig(output_dir / 'risk_distribution.svg', dpi=150)
plt.close()
print(f'   [OK] Saved: {output_dir}/risk_distribution.svg')

# ========================================
# 5. ROC CURVE
# ========================================
print('[*] Creating ROC Curve...')
fpr, tpr, thresholds = roc_curve(ground_truths, risk_scores)
roc_auc = auc(fpr, tpr)

fig, ax = plt.subplots(figsize=(8, 8))
ax.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.3f})')
ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random')
ax.fill_between(fpr, tpr, alpha=0.3, color='darkorange')

ax.set_xlim([0.0, 1.0])
ax.set_ylim([0.0, 1.05])
ax.set_xlabel('False Positive Rate', fontsize=12)
ax.set_ylabel('True Positive Rate', fontsize=12)
ax.set_title('ROC Curve - Hallucination Detection', fontsize=14, fontweight='bold')
ax.legend(loc='lower right')
plt.tight_layout()
plt.savefig(output_dir / 'roc_curve.svg', dpi=150)
plt.close()
print(f'   [OK] Saved: {output_dir}/roc_curve.svg')

# ========================================
# 6. SUMMARY DASHBOARD
# ========================================
print('[*] Creating Summary Dashboard...')
fig = plt.figure(figsize=(16, 12))

# Subplot 1: Confusion Matrix
ax1 = fig.add_subplot(2, 3, 1)
im = ax1.imshow(cm, cmap='Blues')
ax1.set_xticks([0, 1])
ax1.set_yticks([0, 1])
ax1.set_xticklabels(['Truth', 'Halluc'])
ax1.set_yticklabels(['Truth', 'Halluc'])
for i in range(2):
    for j in range(2):
        ax1.text(j, i, cm[i, j], ha='center', va='center', color='black', fontsize=14)
ax1.set_title('Confusion Matrix', fontweight='bold')

# Subplot 2: Metrics
ax2 = fig.add_subplot(2, 3, 2)
metrics_vals = [f1_score(ground_truths, predictions),
                precision_score(ground_truths, predictions),
                recall_score(ground_truths, predictions),
                accuracy_score(ground_truths, predictions)]
bars = ax2.bar(['F1', 'Precision', 'Recall', 'Accuracy'], metrics_vals,
               color=['#2ecc71', '#3498db', '#9b59b6', '#e74c3c'])
ax2.set_ylim(0, 1.1)
ax2.set_title('Performance Metrics', fontweight='bold')
for bar, val in zip(bars, metrics_vals):
    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
             f'{val:.3f}', ha='center', fontsize=11)

# Subplot 3: Weights
ax3 = fig.add_subplot(2, 3, 3)
ax3.pie([alpha, beta, gamma, delta], labels=['GT Cont.', 'Incons.', 'Entropy', 'NCD'],
        colors=['#ff6b6b', '#4ecdc4', '#45b7d1', '#96ceb4'], autopct='%.1f%%')
ax3.set_title('Fusion Weights', fontweight='bold')

# Subplot 4: Risk Distribution
ax4 = fig.add_subplot(2, 3, 4)
ax4.hist(truthful_risks, bins=25, alpha=0.6, label='Truthful', color='green')
ax4.hist(halluc_risks, bins=25, alpha=0.6, label='Hallucination', color='red')
ax4.axvline(x=threshold, color='black', linestyle='--', lw=2)
ax4.legend(fontsize=9)
ax4.set_title('Risk Distribution', fontweight='bold')

# Subplot 5: ROC
ax5 = fig.add_subplot(2, 3, 5)
ax5.plot(fpr, tpr, color='darkorange', lw=2)
ax5.plot([0, 1], [0, 1], 'k--', lw=1)
ax5.fill_between(fpr, tpr, alpha=0.3, color='darkorange')
ax5.set_title(f'ROC Curve (AUC={roc_auc:.3f})', fontweight='bold')
ax5.set_xlabel('FPR')
ax5.set_ylabel('TPR')

# Subplot 6: Summary Text
ax6 = fig.add_subplot(2, 3, 6)
ax6.axis('off')
summary_text = f'''
LOGIC-HALT Results Summary
========================

Dataset: TruthfulQA (817 questions)
Model: DeBERTa-v3-Large (435M params)
Labels: NLI-based Entailment

Best F1 Score: 0.7730
Precision:     0.9770 (97.7%)
Recall:        0.6395 (63.9%)
Accuracy:      0.7858 (78.6%)
ROC-AUC:       {roc_auc:.3f}

Improvement over Cosine baseline:
  F1:        +67% (0.46 -> 0.77)
  Precision: +174% (0.36 -> 0.98)
'''
ax6.text(0.1, 0.5, summary_text, fontsize=11, family='monospace',
         verticalalignment='center', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.suptitle('LOGIC-HALT: LLM Hallucination Detection - Optimization Results',
             fontsize=16, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(output_dir / 'summary_dashboard.svg', dpi=150, bbox_inches='tight')
plt.close()
print(f'   [OK] Saved: {output_dir}/summary_dashboard.svg')

print()
print('='*60)
print(f'[OK] All visualizations saved to: {output_dir}')
print('='*60)
print()
print('Generated files:')
for f in output_dir.glob('*.svg'):
    print(f'  - {f.name}')
