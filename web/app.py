"""
LOGIC-HALT Web Interface
Flask-based UI for demonstrating hallucination detection
"""

import os
import sys
import json
import logging
import numpy as np
from flask import Flask, render_template, request, jsonify
from dotenv import load_dotenv

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

load_dotenv()

from openai import OpenAI

app = Flask(__name__)
app.secret_key = os.urandom(24)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize OpenAI client
client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

# Import detection modules
try:
    from src.consistency import ConsistencyEngine
    from src.complexity import ComplexityEngine
    from src.fusion import FusionLayer
    from src.answer_validator import AnswerValidator
    MODULES_AVAILABLE = True

    # Initialize engines
    consistency_engine = ConsistencyEngine(device="cpu")
    complexity_engine = ComplexityEngine("config/config.yaml")
    fusion_layer = FusionLayer("config/config.yaml")
    answer_validator = AnswerValidator("config/config.yaml")
    logger.info("Detection modules loaded successfully (including AnswerValidator)")
except Exception as e:
    MODULES_AVAILABLE = False
    answer_validator = None
    logger.warning(f"Detection modules not available: {e}")


def generate_responses(question: str, num_responses: int = 10) -> list:
    """
    Generate multiple responses from LLM for the same question.
    Uses direct answer format for better consistency measurement.

    NOTE: Step-by-step prompts were causing high false positive rates because
    different explanation styles led to low consistency scores even for correct answers.
    Direct answer format produces more comparable responses.
    """
    responses = []

    # Fixed low temperature for consistent responses
    temperatures = [0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3]

    # Direct answer prompt - asks for concise answer first, then brief explanation
    # This reduces variation in explanation style while still getting reasoning
    direct_prompt = f"""Aşağıdaki soruyu cevapla:

Soru: {question}

Önce kısa ve net cevabı ver, sonra kısaca açıkla.
Format:
Cevap: [cevabın]
Açıklama: [kısa açıklama]"""

    # All prompts use the same direct format
    prompts = [direct_prompt] * num_responses

    for i in range(min(num_responses, len(temperatures))):
        try:
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant. Provide concise, accurate answers."},
                    {"role": "user", "content": prompts[i]}
                ],
                temperature=temperatures[i],
                max_tokens=600,
                logprobs=True,
                top_logprobs=5
            )

            response_text = response.choices[0].message.content

            # Extract logprobs
            logprobs_data = None
            if response.choices[0].logprobs and response.choices[0].logprobs.content:
                logprobs_data = [token.logprob for token in response.choices[0].logprobs.content]

            responses.append({
                'id': i + 1,
                'text': response_text,
                'temperature': temperatures[i],
                'logprobs': logprobs_data
            })

        except Exception as e:
            logger.error(f"Error generating response {i+1}: {e}")
            responses.append({
                'id': i + 1,
                'text': f"Error: {str(e)}",
                'temperature': temperatures[i],
                'logprobs': None
            })

    return responses


def extract_final_answer(text: str) -> str:
    """
    Cevap metninden final cevabı çıkar.
    Sayısal cevaplar veya anahtar kelimeler aranır.
    """
    import re

    # "Sonuç:" veya "4. Sonuç:" gibi bölümleri ara
    sonuc_patterns = [
        r'[Ss]onuç[:\s]+(.+?)(?:\n|$)',
        r'[Cc]evap[:\s]+(.+?)(?:\n|$)',
        r'toplam[da]?\s*(\d+)',
        r'(\d+)\s*(?:tane|adet|elma|yumurta|kişi)',
    ]

    for pattern in sonuc_patterns:
        match = re.search(pattern, text)
        if match:
            return match.group(1).strip()[:50]  # İlk 50 karakter

    # Sayı ara
    numbers = re.findall(r'\b(\d+)\b', text)
    if numbers:
        return numbers[-1]  # Son sayıyı al

    # İlk 100 karakteri hash'le
    return str(hash(text[:100]) % 10000)


def calculate_diversity_penalty(responses: list) -> float:
    """
    Cevap çeşitliliği penalty'si hesapla.
    Farklı cevap sayısı arttıkça penalty artar.
    """
    if len(responses) < 2:
        return 0.0

    # Final cevapları çıkar
    answers = [extract_final_answer(r['text']) for r in responses]

    # Benzersiz cevap sayısı
    unique_answers = len(set(answers))

    # Diversity penalty: (unique - 1) * 0.05
    # 1 cevap = 0 penalty
    # 2 cevap = 0.05 penalty
    # 3 cevap = 0.10 penalty
    diversity_penalty = (unique_answers - 1) * 0.05

    logger.info(f"Answer diversity: {unique_answers} unique answers, penalty: {diversity_penalty:.2f}")

    return min(diversity_penalty, 0.15)  # Max 0.15 penalty


def analyze_responses(responses: list, question_id: str = "web_query") -> list:
    """
    Analyze each response for hallucination using TWO-PHASE detection:

    PHASE 1: Answer-level consistency (fast check)
    - Extract final answers from all responses
    - If all answers agree → likely safe (skip detailed analysis)
    - If answers disagree → proceed to Phase 2

    PHASE 2: Detailed analysis (only if Phase 1 shows disagreement)
    - Full NLI-based consistency analysis
    - Entropy and NCD complexity metrics
    - Fusion layer decision

    This two-phase approach reduces false positives caused by different
    explanation styles while maintaining sensitivity to actual hallucinations.
    """
    if not MODULES_AVAILABLE:
        return simple_analysis(responses)

    analyzed = []
    response_texts = [r['text'] for r in responses]
    n = len(responses)

    # Get threshold from fusion layer config
    threshold = fusion_layer.threshold
    logger.info(f"Using threshold: {threshold}")

    # ============================================================
    # PHASE 1: Answer-Level Consistency Check
    # ============================================================
    answer_consistency_high = False
    answer_validations = None
    answer_summary = None

    if answer_validator is not None:
        try:
            answer_validations, answer_summary = answer_validator.validate_responses(
                response_texts,
                [f"response_{r['id']}" for r in responses]
            )

            logger.info(f"[PHASE 1] Answer Validation: {answer_summary.unique_answers} unique answers, "
                       f"majority ratio: {answer_summary.majority_ratio:.2%}")

            # If >80% of responses give the same final answer, consider it highly consistent
            if answer_summary.majority_ratio >= 0.80:
                answer_consistency_high = True
                logger.info(f"[PHASE 1] HIGH answer consistency detected - applying fast-track safe classification")

        except Exception as e:
            logger.error(f"Answer validation error: {e}")

    # ============================================================
    # PHASE 2: Detailed Analysis (if needed)
    # ============================================================

    # Build contradiction matrix (needed for per-response consistency)
    per_response_consistency = [0.5] * n  # Default

    # Skip detailed NLI analysis if answer consistency is high (optimization)
    if not answer_consistency_high:
        try:
            contradiction_matrix = consistency_engine.build_contradiction_matrix(
                response_texts,
                [f"response_{r['id']}" for r in responses]
            )

            for i in range(n):
                if n > 1:
                    avg_contradiction = np.mean([contradiction_matrix[i][j] for j in range(n) if j != i])
                    consistency_i = 1.0 - avg_contradiction
                else:
                    consistency_i = 1.0
                per_response_consistency[i] = float(np.clip(consistency_i, 0.0, 1.0))

            logger.info(f"[PHASE 2] Per-response consistency: {[f'{c:.2f}' for c in per_response_consistency]}")
        except Exception as e:
            logger.error(f"Consistency analysis error: {e}")

    # Analyze each response
    for i, response in enumerate(responses):
        try:
            # Get answer validation for this response
            answer_is_majority = True
            answer_penalty = 0.0
            extracted_answer = None

            if answer_validations is not None and i < len(answer_validations):
                av = answer_validations[i]
                answer_is_majority = av.is_majority
                answer_penalty = av.minority_penalty
                extracted_answer = av.extracted_answer

            # If answer consistency is high AND this response agrees with majority
            # → Fast-track to safe classification with reduced risk
            if answer_consistency_high and answer_is_majority:
                # Use boosted consistency (since answers match)
                response_consistency = 0.85  # High consistency for matching answers
                entropy_norm = 0.25  # Assume low entropy for consistent answers
                ncd_norm = 0.4  # Moderate NCD

                # Direct low-risk classification
                risk = 0.30  # Base low risk for consistent answers
                category = 'safe'
                is_hallucination = False
                confidence = 'high'
                explanation = f"High answer consistency ({answer_summary.majority_ratio:.0%} agreement). Answer: {extracted_answer}"

            else:
                # Full Phase 2 analysis
                response_consistency = per_response_consistency[i]

                # Complexity analysis
                complexity = complexity_engine.analyze_response(
                    response['text'],
                    question_id,
                    f"response_{response['id']}",
                    response.get('logprobs')
                )

                # Normalize entropy
                if complexity.token_entropy and complexity.token_entropy > 0:
                    entropy_norm = 1 / (1 + np.exp(-(complexity.token_entropy - 0.25) / 0.15))
                    entropy_norm = float(np.clip(entropy_norm, 0.15, 0.85))
                else:
                    entropy_norm = 0.3

                ncd_norm = complexity.compression_ratio if complexity.compression_ratio else 0.5

                # Fusion decision
                detection = fusion_layer.detect(
                    question_id,
                    f"response_{response['id']}",
                    response_consistency,
                    entropy_norm,
                    ncd_norm
                )

                # Apply answer minority penalty
                risk = detection.hallucination_risk + answer_penalty
                risk = min(risk, 1.0)

                # Classification
                if risk < threshold:
                    category = 'safe'
                    is_hallucination = False
                else:
                    category = 'hallucination'
                    is_hallucination = True

                confidence = detection.confidence
                explanation = detection.explanation

                if not answer_is_majority and extracted_answer:
                    explanation += f" [Minority answer: {extracted_answer}]"

            # Calculate raw entropy for display
            raw_entropy = 0
            if not answer_consistency_high:
                complexity = complexity_engine.analyze_response(
                    response['text'], question_id, f"response_{response['id']}", response.get('logprobs')
                )
                raw_entropy = complexity.token_entropy if complexity.token_entropy else 0

            analyzed.append({
                'id': response['id'],
                'text': response['text'],
                'temperature': response['temperature'],
                'is_hallucination': is_hallucination,
                'category': category,
                'risk_score': round(risk, 4),
                'confidence': confidence,
                'explanation': explanation,
                'extracted_answer': extracted_answer,
                'answer_is_majority': answer_is_majority,
                'metrics': {
                    'consistency': round(response_consistency, 4),
                    'entropy': round(entropy_norm, 4),
                    'entropy_raw': round(raw_entropy, 2),
                    'ncd': round(ncd_norm, 4),
                    'answer_penalty': round(answer_penalty, 4)
                }
            })

        except Exception as e:
            logger.error(f"Analysis error for response {response['id']}: {e}")
            analyzed.append({
                'id': response['id'],
                'text': response['text'],
                'temperature': response['temperature'],
                'is_hallucination': None,
                'risk_score': None,
                'confidence': 'unknown',
                'explanation': f'Analysis error: {str(e)}',
                'metrics': {}
            })

    return analyzed


def simple_analysis(responses: list) -> list:
    """
    Simplified analysis when full modules aren't available.
    Uses basic heuristics.
    """
    analyzed = []

    # Calculate basic consistency (how similar are responses)
    texts = [r['text'].lower() for r in responses]

    for response in responses:
        # Simple heuristics
        text = response['text']

        # Check for uncertainty markers
        uncertainty_markers = ['i think', 'maybe', 'possibly', 'i\'m not sure',
                              'might be', 'could be', 'perhaps', 'allegedly']
        uncertainty_score = sum(1 for marker in uncertainty_markers if marker in text.lower())

        # Check response length (very short or very long can be suspicious)
        length_score = 0.5
        if len(text) < 50:
            length_score = 0.7
        elif len(text) > 500:
            length_score = 0.6

        # Simple risk calculation
        risk = min(0.3 + (uncertainty_score * 0.1) + (length_score - 0.5), 1.0)

        analyzed.append({
            'id': response['id'],
            'text': text,
            'temperature': response['temperature'],
            'is_hallucination': risk > 0.5,
            'risk_score': round(risk, 4),
            'confidence': 'low' if 0.4 < risk < 0.6 else 'medium',
            'explanation': 'Simplified analysis (full modules not loaded)',
            'metrics': {
                'uncertainty_markers': uncertainty_score,
                'length': len(text)
            }
        })

    return analyzed


@app.route('/')
def index():
    """Main page"""
    return render_template('index.html', modules_available=MODULES_AVAILABLE)


@app.route('/analyze', methods=['POST'])
def analyze():
    """API endpoint for analysis"""
    data = request.json
    question = data.get('question', '')
    num_responses = data.get('num_responses', 10)

    if not question:
        return jsonify({'error': 'Question is required'}), 400

    try:
        # Generate responses
        logger.info(f"Generating {num_responses} responses for: {question[:50]}...")
        responses = generate_responses(question, num_responses)

        # Analyze responses
        logger.info("Analyzing responses for hallucinations...")
        analyzed = analyze_responses(responses)

        # Calculate summary statistics (2 categories)
        safe_count = sum(1 for r in analyzed if r.get('category') == 'safe')
        hallucination_count = sum(1 for r in analyzed if r.get('category') == 'hallucination')
        avg_risk = sum(r.get('risk_score', 0) or 0 for r in analyzed) / len(analyzed)

        return jsonify({
            'success': True,
            'question': question,
            'responses': analyzed,
            'summary': {
                'total_responses': len(analyzed),
                'safe_count': safe_count,
                'hallucinations_detected': hallucination_count,
                'average_risk': round(avg_risk, 4),
                'modules_available': MODULES_AVAILABLE
            }
        })

    except Exception as e:
        logger.error(f"Analysis error: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/analyze_qa', methods=['POST'])
def analyze_qa():
    """
    Analyze a given question-answer pair for hallucination.
    User provides both the question and the answer to check.
    """
    data = request.json
    question = data.get('question', '')
    answer = data.get('answer', '')

    if not question or not answer:
        return jsonify({'error': 'Both question and answer are required'}), 400

    try:
        logger.info(f"Analyzing Q&A pair: {question[:50]}...")

        if not MODULES_AVAILABLE:
            return jsonify({'error': 'Detection modules not available'}), 500

        # Strategy: Generate variant responses and compare with user's answer
        # This tests if the answer is consistent with what the model would say

        # Step 1: Generate multiple responses from LLM for comparison
        variant_responses = []
        temperatures = [0.3, 0.5, 0.7]

        for temp in temperatures:
            try:
                response = client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[
                        {"role": "system", "content": "You are a helpful assistant. Answer concisely and accurately."},
                        {"role": "user", "content": question}
                    ],
                    temperature=temp,
                    max_tokens=500,
                    logprobs=True,
                    top_logprobs=5
                )

                response_text = response.choices[0].message.content
                logprobs_data = None
                if response.choices[0].logprobs and response.choices[0].logprobs.content:
                    logprobs_data = [token.logprob for token in response.choices[0].logprobs.content]

                variant_responses.append({
                    'text': response_text,
                    'temperature': temp,
                    'logprobs': logprobs_data
                })
            except Exception as e:
                logger.error(f"Error generating variant: {e}")

        # Step 2: Add user's answer to the response list for consistency analysis
        all_responses = [answer] + [r['text'] for r in variant_responses]

        # Step 3: Consistency analysis - compare user's answer with LLM variants
        try:
            contradiction_matrix = consistency_engine.build_contradiction_matrix(
                all_responses,
                ['user_answer'] + [f"variant_{i}" for i in range(len(variant_responses))]
            )

            # User answer consistency = how consistent is it with LLM variants
            n = len(all_responses)
            if n > 1:
                avg_contradiction = np.mean([contradiction_matrix[0][j] for j in range(1, n)])
                user_consistency = 1.0 - avg_contradiction
            else:
                user_consistency = 0.5

            user_consistency = float(np.clip(user_consistency, 0.0, 1.0))
            logger.info(f"User answer consistency: {user_consistency:.4f}")

        except Exception as e:
            logger.error(f"Consistency analysis error: {e}")
            user_consistency = 0.5

        # Step 4: Complexity analysis on user's answer
        try:
            complexity = complexity_engine.analyze_response(
                answer,
                "qa_check",
                "user_answer",
                None  # No logprobs for user input
            )

            # Use compression ratio as complexity metric
            ncd_norm = complexity.compression_ratio if complexity.compression_ratio else 0.5

            # Estimate entropy based on text characteristics
            word_count = len(answer.split())
            unique_words = len(set(answer.lower().split()))
            if word_count > 0:
                lexical_diversity = unique_words / word_count
                entropy_norm = 1.0 - lexical_diversity  # High diversity = low entropy estimate
                entropy_norm = float(np.clip(entropy_norm, 0.1, 0.9))
            else:
                entropy_norm = 0.5

        except Exception as e:
            logger.error(f"Complexity analysis error: {e}")
            ncd_norm = 0.5
            entropy_norm = 0.5

        # Step 5: Fusion decision
        detection = fusion_layer.detect(
            "qa_check",
            "user_answer",
            user_consistency,
            entropy_norm,
            ncd_norm
        )

        risk = detection.hallucination_risk
        threshold = fusion_layer.threshold  # Use config threshold

        if risk < threshold:
            category = 'safe'
            is_hallucination = False
            verdict = "Bu cevap güvenilir görünüyor. LLM'in ürettiği cevaplarla tutarlı."
        else:
            category = 'hallucination'
            is_hallucination = True
            verdict = "Bu cevap halüsinasyon olabilir. LLM'in ürettiği cevaplarla tutarsızlık tespit edildi."

        # Prepare variant summaries
        variant_summaries = []
        for i, v in enumerate(variant_responses):
            variant_summaries.append({
                'id': i + 1,
                'temperature': v['temperature'],
                'text_preview': v['text'][:200] + '...' if len(v['text']) > 200 else v['text']
            })

        return jsonify({
            'success': True,
            'question': question,
            'answer': answer,
            'result': {
                'is_hallucination': is_hallucination,
                'category': category,
                'risk_score': round(risk, 4),
                'confidence': detection.confidence,
                'verdict': verdict,
                'explanation': detection.explanation
            },
            'metrics': {
                'consistency': round(user_consistency, 4),
                'entropy': round(entropy_norm, 4),
                'ncd': round(ncd_norm, 4)
            },
            'variants': variant_summaries
        })

    except Exception as e:
        logger.error(f"Q&A Analysis error: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/health')
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'modules_available': MODULES_AVAILABLE,
        'openai_configured': bool(os.getenv('OPENAI_API_KEY'))
    })


@app.route('/save_labels', methods=['POST'])
def save_labels():
    """Save labeled results to JSON file for statistics"""
    data = request.json

    labels_file = 'data/labeled_results.json'

    try:
        # Load existing data
        existing_data = []
        if os.path.exists(labels_file):
            with open(labels_file, 'r', encoding='utf-8') as f:
                existing_data = json.load(f)

        # Add new entry
        existing_data.append(data)

        # Save back
        os.makedirs(os.path.dirname(labels_file), exist_ok=True)
        with open(labels_file, 'w', encoding='utf-8') as f:
            json.dump(existing_data, f, ensure_ascii=False, indent=2)

        # Calculate statistics
        total_responses = sum(len(entry.get('responses', [])) for entry in existing_data)

        logger.info(f"Saved labels: {len(data.get('responses', []))} responses, total: {total_responses}")

        return jsonify({
            'success': True,
            'total_records': total_responses,
            'total_questions': len(existing_data)
        })

    except Exception as e:
        logger.error(f"Error saving labels: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/statistics')
def statistics():
    """Get statistics from labeled data"""
    labels_file = 'data/labeled_results.json'

    if not os.path.exists(labels_file):
        return jsonify({'error': 'No labeled data yet'}), 404

    try:
        with open(labels_file, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # Calculate metrics
        total_responses = 0
        true_positives = 0  # Predicted hallucination, actually hallucination
        false_positives = 0  # Predicted hallucination, actually correct
        true_negatives = 0  # Predicted safe, actually correct
        false_negatives = 0  # Predicted safe, actually hallucination

        for entry in data:
            for r in entry.get('responses', []):
                total_responses += 1
                predicted = r.get('predicted', False)
                ground_truth = r.get('ground_truth', False)

                if predicted and ground_truth:
                    true_positives += 1
                elif predicted and not ground_truth:
                    false_positives += 1
                elif not predicted and not ground_truth:
                    true_negatives += 1
                else:
                    false_negatives += 1

        # Calculate metrics
        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        accuracy = (true_positives + true_negatives) / total_responses if total_responses > 0 else 0

        return jsonify({
            'total_questions': len(data),
            'total_responses': total_responses,
            'confusion_matrix': {
                'true_positives': true_positives,
                'false_positives': false_positives,
                'true_negatives': true_negatives,
                'false_negatives': false_negatives
            },
            'metrics': {
                'accuracy': round(accuracy, 4),
                'precision': round(precision, 4),
                'recall': round(recall, 4),
                'f1_score': round(f1, 4)
            }
        })

    except Exception as e:
        logger.error(f"Error calculating statistics: {e}")
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    print("="*60)
    print("LOGIC-HALT Hallucination Detection Demo")
    print("="*60)
    print(f"Modules available: {MODULES_AVAILABLE}")
    print(f"OpenAI configured: {bool(os.getenv('OPENAI_API_KEY'))}")
    print("Starting server at http://127.0.0.1:5000")
    print("="*60)

    app.run(debug=True, host='127.0.0.1', port=5000)
