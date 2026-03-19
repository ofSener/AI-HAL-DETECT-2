# LOGIC-HALT Fusion Redesign — 5-Signal Architecture

## Problem

The current system's reported metrics (P=0.977, R=0.640, F1=0.773) are based on a GT Contradiction signal that uses ground truth at evaluation time. This signal is unavailable at inference, making the reported numbers unreliable. The runtime `fusion.py` maps the optimized 4-weight config to a 3-signal system with incorrect signal assignments. Additional bugs (wrong NLI model loaded, NCD not actually computed, Answer Validator not integrated) further degrade real-world performance.

## Goal

Replace GT Contradiction with inference-time-available proxy signals, fix all bugs, and re-optimize to maximize F1 with honest metrics.

## Architecture: 5-Signal Fusion

```
Risk = α×SelfVerification + β×Inconsistency + γ×Entropy + δ×NCD + ε×MinorityPenalty
```

### Signal Definitions

| Signal | Source | Description | Cost |
|--------|--------|-------------|------|
| SelfVerification | New (Faz 2) | NLI contradiction between original answer and model's self-verification response | +1 API call/question |
| Inconsistency | Module C (fixed) | 1 - consistency_score from pairwise NLI on ANSWER portions | Existing |
| Entropy | Module D | Normalized token entropy from logprobs | Existing |
| NCD | Module D (fixed) | Pairwise Normalized Compression Distance between responses | Existing |
| MinorityPenalty | Module F | Binary penalty when extracted answer differs from majority | Existing |

### Phase 1: Bug Fixes

1. **fusion.py**: Rewrite to accept N signals dynamically. Remove hardcoded 3-weight system.
2. **consistency.py**: Fix model loading — try large first, fall back to base. Remove fallback chain that starts with base.
3. **complexity.py**: Add `calculate_pairwise_ncd()` method that computes average NCD across all response pairs (not single-text compression ratio).
4. **detector.py**: Import and call AnswerValidator. Pass minority_penalty to fusion layer.

### Phase 2: Self-Verification Signal

New method in `interrogator.py`:
```python
def self_verify(self, question: str, answer: str, model_name: str) -> LLMResponse:
    """Ask the model to verify its own answer. Returns verification response."""
    prompt = f"Question: {question}\nProposed answer: {answer}\n\nIs this answer correct? If not, what is the correct answer?"
    # Single API call, temperature=0.0 for deterministic verification
```

In `consistency.py`, add:
```python
def compute_verification_score(self, original_answer: str, verification: str) -> float:
    """NLI between original answer and verification response. Returns contradiction score."""
```

### Phase 3: Re-optimization

Update `batch_optimization.py`:
- Remove GT Contradiction signal
- Add SelfVerification signal (pre-compute for all questions)
- Add MinorityPenalty signal
- Optimize 5 weights (α,β,γ,δ,ε) + threshold + NLI params
- Config updated with new optimal weights

### Files Changed

| File | Change |
|------|--------|
| `src/fusion.py` | Rewrite: N-signal support, proper weight loading |
| `src/consistency.py` | Fix model loading order, add verification_score method |
| `src/complexity.py` | Add pairwise NCD calculation |
| `src/detector.py` | Integrate AnswerValidator + SelfVerification into pipeline |
| `src/interrogator.py` | Add self_verify() method |
| `config/config.yaml` | Add epsilon weight, self_verification section |
| `scripts/batch_optimization.py` | Update to 5-signal system |

### Success Criteria

- All 5 signals computed at inference time (no ground truth dependency)
- F1 score re-optimized and honestly reported
- Existing web UI and CLI continue to work
- Backward-compatible config (old 3-signal configs still load with defaults)
