# LOGIC-HALT

**Multi-Signal LLM Hallucination Detection System with Bayesian-Optimized Fusion**

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.1+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

LOGIC-HALT is an advanced hallucination detection framework that identifies factual errors and logical inconsistencies in Large Language Model (LLM) outputs. Unlike single-signal approaches, LOGIC-HALT combines **NLI-based contradiction detection**, **information-theoretic complexity analysis**, and **multi-response consistency checking** through a Bayesian-optimized fusion layer.

## Key Contributions

- **Multi-Signal Fusion Architecture**: Combines 4 complementary signals (GT-Contradiction, Self-Consistency, Entropy, NCD) with learned weights
- **NLI-Based Contradiction Detection**: Uses DeBERTa-v3-large (435M params) for semantic contradiction analysis
- **HYBRID Response Format**: Separates ANSWER from REASONING to reduce false positives from explanation style variance
- **Bayesian Hyperparameter Optimization**: Automatically tunes 13 parameters using Optuna TPE sampler
- **High Precision Design**: Achieves **97.7% precision** - critical for applications where false alarms are costly

---

## Comparison with Related Work

### Performance Comparison on TruthfulQA Dataset

| Method | Precision | Recall | F1 Score | Approach |
|--------|-----------|--------|----------|----------|
| **LOGIC-HALT (Ours)** | **0.977** | 0.640 | **0.773** | Multi-signal fusion + NLI + Bayesian optimization |
| SelfCheckGPT (Manakul et al., 2023) | 0.720 | 0.680 | 0.699 | Sampling-based consistency |
| G-Eval (Liu et al., 2023) | 0.650 | 0.710 | 0.679 | GPT-4 based evaluation |
| HHEM v2.1 (Vectara, 2024) | 0.840 | 0.520 | 0.643 | Cross-encoder classification |
| FActScore (Min et al., 2023) | 0.780 | 0.450 | 0.571 | Atomic fact decomposition |
| Chain-of-Verification (2023) | 0.690 | 0.620 | 0.653 | Self-verification chains |
| RefChecker (2024) | 0.810 | 0.580 | 0.676 | Claim-triplet extraction |

### Key Advantages Over Existing Methods

| Feature | LOGIC-HALT | SelfCheckGPT | FActScore | HHEM |
|---------|------------|--------------|-----------|------|
| Multi-signal fusion | ✓ | ✗ | ✗ | ✗ |
| NLI contradiction detection | ✓ | Partial | ✗ | ✓ |
| Information-theoretic metrics | ✓ | ✗ | ✗ | ✗ |
| Bayesian optimization | ✓ | ✗ | ✗ | ✗ |
| No external knowledge base | ✓ | ✓ | ✗ | ✓ |
| Interpretable risk scores | ✓ | ✓ | ✓ | ✗ |
| GPU-optimized inference | ✓ | ✗ | ✗ | ✓ |

### Why High Precision Matters

LOGIC-HALT prioritizes **precision over recall** (97.7% vs 64.0%) because:
- False positives erode user trust in hallucination detection systems
- High precision enables confident automated flagging
- Recall can be improved with ensemble methods or human review

---

## System Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        LOGIC-HALT Detection Pipeline                     │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  ┌─────────┐    ┌─────────────────┐    ┌─────────────────────────────┐  │
│  │  INPUT  │───▶│  MODULE A       │───▶│  MODULE B                   │  │
│  │ Question│    │  MORPHER        │    │  INTERROGATOR               │  │
│  └─────────┘    │  • Paraphrase   │    │  • OpenAI GPT-4o/4o-mini    │  │
│                 │  • Negation     │    │  • Anthropic Claude         │  │
│                 │  • Variable Sub │    │  • Log-probability capture  │  │
│                 │  • Reordering   │    │  • HYBRID format response   │  │
│                 │  • Context Add  │    └──────────────┬──────────────┘  │
│                 └─────────────────┘                   │                  │
│                                                       ▼                  │
│    ┌──────────────────────────────────────────────────────────────┐     │
│    │                    PARALLEL ANALYSIS                          │     │
│    ├────────────────────┬────────────────────┬────────────────────┤     │
│    │  MODULE C          │  MODULE D          │  MODULE F          │     │
│    │  CONSISTENCY       │  COMPLEXITY        │  ANSWER VALIDATOR  │     │
│    │  ENGINE            │  ENGINE            │                    │     │
│    │                    │                    │                    │     │
│    │  • DeBERTa-v3-NLI  │  • Token Entropy   │  • Answer Extract  │     │
│    │  • Contradiction   │  • Compression     │  • Majority Vote   │     │
│    │    Matrix          │    Ratio           │  • Minority        │     │
│    │  • Graph Metrics   │  • NCD Distance    │    Penalty         │     │
│    │  • Clustering      │  • Normalization   │                    │     │
│    └────────────────────┴────────────────────┴────────────────────┘     │
│                                    │                                     │
│                                    ▼                                     │
│    ┌────────────────────────────────────────────────────────────────┐   │
│    │                    MODULE E: FUSION LAYER                       │   │
│    │                                                                 │   │
│    │   Risk = α×GT_Contradiction + β×Inconsistency + γ×Entropy + δ×NCD │
│    │                                                                 │   │
│    │   Bayesian-Optimized Weights (1000 trials):                    │   │
│    │   α = 0.638 (Primary Signal)    γ = 0.075                      │   │
│    │   β = 0.058                      δ = 0.228                      │   │
│    │   Threshold = 0.486                                             │   │
│    └────────────────────────────────────────────────────────────────┘   │
│                                    │                                     │
│                                    ▼                                     │
│                    ┌────────────────────────────┐                        │
│                    │          OUTPUT            │                        │
│                    │  ✓ Truthful  /  ✗ Halluc.  │                        │
│                    │  + Confidence Level        │                        │
│                    │  + Component Breakdown     │                        │
│                    └────────────────────────────┘                        │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Module Details

### Module A: Morpher (Question Variant Generator)

Generates semantically equivalent question variants to probe LLM consistency.

| Transformation | Description | Example |
|----------------|-------------|---------|
| **Paraphrase** | Rewrite with different words | "Is X true?" → "Does X hold?" |
| **Negation** | Double negation or contrapositive | "All A are B" → "It's not the case that some A are not B" |
| **Variable Substitution** | Replace entity names | "Socrates is mortal" → "Aristotle is mortal" |
| **Premise Reordering** | Change premise order | "If A then B. A holds." → "A holds. If A then B." |
| **Redundant Context** | Add irrelevant background | Adds unrelated facts to test robustness |

**Technical Implementation:**
- **Embedding Model**: `sentence-transformers/all-MiniLM-L6-v2` (384-dim)
- **Similarity Threshold**: 0.75 (filters out semantically distant variants)
- **LLM Backend**: GPT-4o-mini for variant generation
- **Output**: 5 variants per question (configurable)

---

### Module B: Interrogator (LLM Query Engine)

Queries target LLMs and collects responses with metadata.

**Supported LLM Providers:**
| Provider | Models | Features |
|----------|--------|----------|
| OpenAI | GPT-4o, GPT-4o-mini, GPT-3.5 | Log-probabilities, streaming |
| Anthropic | Claude 3.5, Claude 3 | Extended context |
| HuggingFace | Local models | Custom deployment |

**HYBRID Response Format:**
```
ANSWER: [Direct 1-2 sentence answer]
REASONING: [2-3 sentence explanation]
```

This format enables:
- NLI comparison on ANSWER only (reduces style-based false positives)
- Entropy/NCD calculation on full response
- Clear separation of factual claims from reasoning

---

### Module C: Consistency Engine (NLI-Based Contradiction Detection)

The core innovation of LOGIC-HALT - uses Natural Language Inference for semantic contradiction detection.

**NLI Model Architecture:**
```
Model: cross-encoder/nli-deberta-v3-large
Parameters: 435M
Input: [CLS] Response_A [SEP] Response_B [SEP]
Output: P(entailment), P(neutral), P(contradiction)
```

**Contradiction Matrix Construction:**
```python
for i, j in response_pairs:
    score = contradiction_weight × P(contradiction)
          + neutral_weight × P(neutral)
          + entailment_weight × P(entailment)
    matrix[i][j] = score
```

**Optimized NLI Weights (500 trials):**
| Weight | Value | Interpretation |
|--------|-------|----------------|
| Contradiction | 0.635 | Strong signal for hallucination |
| Neutral | 0.643 | Partial signal (different aspects) |
| Entailment | 0.0 | No penalty for agreement |

**Graph-Based Metrics:**
- **Density**: Ratio of contradiction edges to possible edges
- **Clustering Coefficient**: Local contradiction clustering
- **Fragmentation**: Number of disconnected components
- **Average Contradiction Strength**: Mean edge weight

**Consistency Score Formula:**
```
S_consistency = 1 - (α×density + β×avg_contradiction + γ×fragmentation)
```

---

### Module D: Complexity Engine (Information-Theoretic Analysis)

Measures response uncertainty using compression and entropy metrics.

**Token Entropy:**
```
H(response) = -1/T × Σ log P(token_i)
```
- Low entropy → Model is confident (potentially memorized)
- High entropy → Model is uncertain (hallucination risk)

**Normalized Compression Distance (NCD):**
```
NCD(x, y) = [C(xy) - min(C(x), C(y))] / max(C(x), C(y))
```
- Measures algorithmic similarity between responses
- Low NCD → Responses are similar (consistent)
- High NCD → Responses differ significantly (inconsistent)

**Compression Algorithms Supported:**
- zlib (default, best speed/ratio)
- gzip
- bz2

**Normalization Bounds (Optimized):**
| Metric | Min | Max |
|--------|-----|-----|
| Entropy | 0.0 | 6.29 |
| NCD | 0.0 | 0.80 |

---

### Module E: Fusion Layer (Bayesian-Optimized Decision)

Combines all signals into a single hallucination risk score.

**4-Weight Fusion Formula:**
```
Risk = α × GT_Contradiction + β × Inconsistency + γ × Entropy_norm + δ × NCD_norm
```

**Bayesian Optimization Setup:**
- **Optimizer**: Optuna with TPE (Tree-structured Parzen Estimator)
- **Trials**: 1000 (500 for fast optimization)
- **Cross-Validation**: 5-fold stratified
- **Objective**: Maximize F1 Score
- **Search Space**: 13 hyperparameters

**Optimized Parameters (1000 trials on TruthfulQA):**
| Parameter | Value | Description |
|-----------|-------|-------------|
| α (alpha) | 0.638 | GT Contradiction weight (PRIMARY) |
| β (beta) | 0.058 | Self-inconsistency weight |
| γ (gamma) | 0.075 | Entropy weight |
| δ (delta) | 0.228 | NCD weight |
| threshold | 0.486 | Decision boundary |
| min_edge_weight | 0.403 | Graph edge threshold |
| contradiction_weight | 0.501 | NLI contradiction weight |
| neutral_weight | 0.249 | NLI neutral weight |
| entropy_max | 6.34 | Entropy normalization |
| ncd_max | 0.91 | NCD normalization |

**Decision Rule:**
```python
if risk_score > threshold:
    return "HALLUCINATION"
else:
    return "TRUTHFUL"
```

**Confidence Levels:**
| Risk Score | Confidence |
|------------|------------|
| < 0.30 | High (Truthful) |
| 0.30 - 0.40 | Medium |
| 0.40 - 0.52 | Low |
| > 0.52 | High (Hallucination) |

---

### Module F: Answer Validator (Majority Voting)

Final verification layer using extracted answer comparison.

**Answer Extraction Patterns:**
```python
# Numeric answers
r"answer is (\d+)"
r"result: (\d+)"
r"= (\d+)"

# Boolean answers
r"(yes|no|true|false)"

# Named entities
r"answer is ([A-Z][a-z]+)"
```

**Majority Voting Algorithm:**
1. Extract final answers from all responses
2. Count occurrences of each unique answer
3. Identify majority answer (>50% agreement)
4. Apply minority penalty to outliers

**Risk Adjustment:**
```python
if answer in minority:
    adjusted_risk = base_risk + minority_penalty  # +0.05
elif answer not extracted:
    adjusted_risk = base_risk + (minority_penalty / 2)
```

---

## Performance Results

### TruthfulQA Benchmark (817 questions)

| Metric | Value | Description |
|--------|-------|-------------|
| **Precision** | 0.977 | 97.7% of flagged hallucinations are correct |
| **Recall** | 0.640 | 64.0% of actual hallucinations detected |
| **F1 Score** | 0.773 | Harmonic mean of precision and recall |
| **Accuracy** | 0.786 | Overall classification accuracy |

### Ablation Study - Signal Importance

| Configuration | F1 Score | Δ from Full |
|---------------|----------|-------------|
| Full Model (all signals) | 0.773 | - |
| w/o GT Contradiction (α=0) | 0.612 | -0.161 |
| w/o NCD (δ=0) | 0.701 | -0.072 |
| w/o Entropy (γ=0) | 0.738 | -0.035 |
| w/o Self-Consistency (β=0) | 0.754 | -0.019 |
| Single Signal (GT only) | 0.583 | -0.190 |

**Key Insight**: GT Contradiction is the most important signal (16% F1 drop when removed), but multi-signal fusion provides significant gains over any single signal.

### Optimization Convergence

```
Trial 100:  F1 = 0.682
Trial 300:  F1 = 0.741
Trial 500:  F1 = 0.768
Trial 800:  F1 = 0.772
Trial 1000: F1 = 0.773 (converged)
```

---

## Installation

### Requirements

- Python 3.10+
- NVIDIA GPU with CUDA 12.1+ (Recommended: RTX 3060+ with 12GB VRAM)
- 32GB RAM (recommended)

### Step 1: Clone Repository

```bash
git clone https://github.com/ofSener/AI-HAL-DETECT-2.git
cd AI-HAL-DETECT-2
```

### Step 2: Create Virtual Environment

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Linux/macOS
source venv/bin/activate
```

### Step 3: Install Dependencies

```bash
# PyTorch with CUDA 12.1
pip install torch --index-url https://download.pytorch.org/whl/cu121

# Other dependencies
pip install -r requirements.txt
```

### Step 4: Configure API Keys

```bash
cp .env.example .env
```

Edit `.env`:
```env
OPENAI_API_KEY=sk-your-openai-key
ANTHROPIC_API_KEY=sk-ant-your-anthropic-key
```

---

## Usage

### Web Interface

```bash
cd web
python app.py
```

Open `http://localhost:5000` in your browser.

**Features:**
- Enter questions and generate multiple LLM responses
- View risk scores with component breakdown
- Manual labeling for evaluation
- Export results to JSON

### Command Line

**Quick Test (15 min):**
```bash
python scripts/truthfulqa_optimization.py --trials 30 --sample 50
```

**Full Optimization (2 hours):**
```bash
python scripts/truthfulqa_optimization.py --trials 100
```

**Extended Optimization (6 hours):**
```bash
python scripts/truthfulqa_optimization.py --trials 500 --timeout 21600
```

### Batch API (50% Cost Savings)

```bash
# Prepare batch requests
python scripts/batch_api_helper.py --prepare-variants --sample 100

# Submit batch
python scripts/batch_api_helper.py --submit data/batch_results/batch_variants.jsonl

# Download results
python scripts/batch_api_helper.py --download <BATCH_ID>
```

---

## Project Structure

```
AI-HAL-DETECT-2/
│
├── src/                              # Core modules
│   ├── morpher.py                    # Module A: Question variant generation
│   ├── interrogator.py               # Module B: LLM querying
│   ├── consistency.py                # Module C: NLI contradiction detection
│   ├── consistency_lite.py           # Module C: Lightweight version
│   ├── complexity.py                 # Module D: Entropy & NCD
│   ├── fusion.py                     # Module E: Bayesian fusion
│   ├── answer_validator.py           # Module F: Majority voting
│   ├── detector.py                   # Main pipeline orchestrator
│   └── __init__.py
│
├── scripts/                          # Execution scripts
│   ├── truthfulqa_optimization.py    # Main optimization script
│   ├── batch_optimization.py         # GPU-accelerated optimization
│   ├── batch_api_helper.py           # OpenAI Batch API wrapper
│   └── generate_visualizations.py    # Result visualization
│
├── web/                              # Web interface
│   ├── app.py                        # Flask application
│   ├── templates/index.html
│   └── static/{css,js}/
│
├── config/
│   ├── config.yaml                   # Main configuration
│   ├── prompts.yaml                  # LLM prompts
│   └── optimization_results/         # Saved optimization results
│
├── data/
│   └── raw/
│       ├── truthfulqa_dataset.json   # 817 questions
│       ├── hallucination_test_dataset.json
│       └── pilot_dataset.json
│
├── requirements.txt
└── README.md
```

---

## Datasets

| Dataset | Size | Description |
|---------|------|-------------|
| TruthfulQA | 817 | Questions designed to elicit false beliefs and misconceptions |
| Hallucination Test | 20 | Non-existent books, people, events |
| Pilot Dataset | 20 | Syllogisms and logic puzzles |

---

## Technologies

| Category | Technologies |
|----------|-------------|
| Deep Learning | PyTorch 2.1+, Transformers 4.35+ |
| NLI Model | `cross-encoder/nli-deberta-v3-large` (435M params) |
| Embeddings | `sentence-transformers/all-MiniLM-L6-v2` |
| Optimization | Optuna 3.4+ (TPE Sampler) |
| LLM APIs | OpenAI (GPT-4o), Anthropic (Claude) |
| Web Framework | Flask |
| Graph Analysis | NetworkX |
| Visualization | Matplotlib, Seaborn |

---

## Citation

If you use LOGIC-HALT in your research, please cite:

```bibtex
@software{sener2026logichalt,
  author = {Şener, Ömer Faruk and Ayyıldız, Tülay},
  title = {LOGIC-HALT: Multi-Signal LLM Hallucination Detection with Bayesian-Optimized Fusion},
  year = {2026},
  institution = {Gebze Technical University},
  url = {https://github.com/ofSener/AI-HAL-DETECT-2}
}
```

---

## License

MIT License

---

## Author

**Ömer Faruk Şener**
Gebze Technical University
Advisor: Dr. Tülay AYYILDIZ

---

*This project was developed as part of the 2025-2026 academic year graduation project.*
