"""
OpenAI Batch API Helper for LOGIC-HALT
Tüm API çağrılarını batch olarak gönderir, rate limit sorunu olmaz.
%50 daha ucuz, 1-6 saat içinde sonuç.
"""

import json
import os
import sys
import time
import yaml
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

# Load .env file
env_path = Path(__file__).parent.parent / '.env'
load_dotenv(env_path)

from openai import OpenAI

# ========================================
# CONFIGURATION
# ========================================
CONFIG_PATH = Path(__file__).parent.parent / 'config' / 'config.yaml'
PROMPTS_PATH = Path(__file__).parent.parent / 'config' / 'prompts.yaml'
DATASET_PATH = Path(__file__).parent.parent / 'data' / 'raw' / 'truthfulqa_dataset.json'
OUTPUT_DIR = Path(__file__).parent.parent / 'data' / 'batch_results'

def load_config():
    with open(CONFIG_PATH, 'r') as f:
        return yaml.safe_load(f)

def load_prompts():
    with open(PROMPTS_PATH, 'r') as f:
        return yaml.safe_load(f)

def load_dataset(sample_size=None):
    with open(DATASET_PATH, 'r', encoding='utf-8') as f:
        data = json.load(f)

    questions = data if isinstance(data, list) else data.get('questions', [])

    if sample_size:
        questions = questions[:sample_size]

    print(f"Loaded {len(questions)} questions")
    return questions


class BatchAPIHelper:
    """OpenAI Batch API wrapper for LOGIC-HALT"""

    def __init__(self):
        self.client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
        self.config = load_config()
        self.prompts = load_prompts()
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    def prepare_variant_requests(self, questions, transformations=['paraphrase', 'negation', 'variable_substitution', 'premise_reordering', 'redundant_context']):
        """
        Prepare JSONL for variant generation (Morpher)
        """
        requests = []
        morpher_prompts = self.prompts['morpher']

        for q in questions:
            q_id = q.get('id', q.get('question_id', f"q_{len(requests)}"))
            q_text = q.get('question', q.get('text', ''))

            for transform_type in transformations:
                prompt_template = morpher_prompts['transformation_prompts'].get(transform_type, '')
                if not prompt_template:
                    continue

                user_prompt = prompt_template.replace('{question}', q_text)

                request = {
                    "custom_id": f"{q_id}__variant__{transform_type}",
                    "method": "POST",
                    "url": "/v1/chat/completions",
                    "body": {
                        "model": self.config['morpher']['model'],
                        "messages": [
                            {"role": "system", "content": morpher_prompts['system_prompt']},
                            {"role": "user", "content": user_prompt}
                        ],
                        "temperature": self.config['morpher']['temperature'],
                        "max_tokens": self.config['morpher']['max_tokens']
                    }
                }
                requests.append(request)

        return requests

    def prepare_interrogation_requests(self, questions, variants_data=None):
        """
        Prepare JSONL for LLM interrogation
        If variants_data is provided, also query variants
        """
        requests = []
        interr_prompts = self.prompts['interrogator']
        model = self.config['interrogator']['target_models'][0]

        for q in questions:
            q_id = q.get('id', q.get('question_id', f"q_{len(requests)}"))
            q_text = q.get('question', q.get('text', ''))

            # Original question
            user_prompt = interr_prompts['user_template'].format(question=q_text)

            request = {
                "custom_id": f"{q_id}__response__original",
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": {
                    "model": model,
                    "messages": [
                        {"role": "system", "content": interr_prompts['system_prompt']},
                        {"role": "user", "content": user_prompt}
                    ],
                    "temperature": self.config['interrogator']['temperature'],
                    "max_tokens": self.config['interrogator']['max_tokens']
                }
            }
            requests.append(request)

            # Variants (if provided)
            if variants_data and q_id in variants_data:
                for var_type, var_text in variants_data[q_id].items():
                    user_prompt = interr_prompts['user_template'].format(question=var_text)

                    request = {
                        "custom_id": f"{q_id}__response__{var_type}",
                        "method": "POST",
                        "url": "/v1/chat/completions",
                        "body": {
                            "model": model,
                            "messages": [
                                {"role": "system", "content": interr_prompts['system_prompt']},
                                {"role": "user", "content": user_prompt}
                            ],
                            "temperature": self.config['interrogator']['temperature'],
                            "max_tokens": self.config['interrogator']['max_tokens']
                        }
                    }
                    requests.append(request)

        return requests

    def save_jsonl(self, requests, filename):
        """Save requests to JSONL file"""
        filepath = OUTPUT_DIR / filename
        with open(filepath, 'w', encoding='utf-8') as f:
            for req in requests:
                f.write(json.dumps(req, ensure_ascii=False) + '\n')
        print(f"Saved {len(requests)} requests to {filepath}")
        return filepath

    def upload_and_create_batch(self, jsonl_path, description="LOGIC-HALT batch"):
        """Upload JSONL and create batch job"""

        # Upload file
        print(f"\nUploading {jsonl_path}...")
        with open(jsonl_path, 'rb') as f:
            file_obj = self.client.files.create(file=f, purpose='batch')

        print(f"File uploaded: {file_obj.id}")

        # Create batch
        print("Creating batch job...")
        batch = self.client.batches.create(
            input_file_id=file_obj.id,
            endpoint="/v1/chat/completions",
            completion_window="24h",
            metadata={"description": description}
        )

        print(f"\n{'='*60}")
        print(f"BATCH JOB CREATED!")
        print(f"{'='*60}")
        print(f"Batch ID: {batch.id}")
        print(f"Status: {batch.status}")
        print(f"Created: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"{'='*60}")
        print(f"\nCheck status with:")
        print(f"  python scripts/batch_api_helper.py --check {batch.id}")
        print(f"\nDownload results with:")
        print(f"  python scripts/batch_api_helper.py --download {batch.id}")

        # Save batch info
        batch_info = {
            "batch_id": batch.id,
            "file_id": file_obj.id,
            "status": batch.status,
            "created_at": datetime.now().isoformat(),
            "jsonl_path": str(jsonl_path)
        }

        info_path = OUTPUT_DIR / f"batch_{batch.id}_info.json"
        with open(info_path, 'w') as f:
            json.dump(batch_info, f, indent=2)

        return batch

    def check_status(self, batch_id):
        """Check batch job status"""
        batch = self.client.batches.retrieve(batch_id)

        print(f"\n{'='*60}")
        print(f"BATCH STATUS: {batch.status.upper()}")
        print(f"{'='*60}")
        print(f"Batch ID: {batch_id}")
        print(f"Status: {batch.status}")

        if batch.request_counts:
            print(f"Total: {batch.request_counts.total}")
            print(f"Completed: {batch.request_counts.completed}")
            print(f"Failed: {batch.request_counts.failed}")

            if batch.request_counts.total > 0:
                pct = (batch.request_counts.completed / batch.request_counts.total) * 100
                print(f"Progress: {pct:.1f}%")

        if batch.status == 'completed':
            print(f"\nOutput file: {batch.output_file_id}")
            print(f"\nDownload with:")
            print(f"  python scripts/batch_api_helper.py --download {batch_id}")

        return batch

    def download_results(self, batch_id):
        """Download batch results"""
        batch = self.client.batches.retrieve(batch_id)

        if batch.status != 'completed':
            print(f"Batch not complete yet. Status: {batch.status}")
            return None

        if not batch.output_file_id:
            print("No output file available")
            return None

        print(f"Downloading results...")
        content = self.client.files.content(batch.output_file_id)

        output_path = OUTPUT_DIR / f"batch_{batch_id}_results.jsonl"
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(content.text)

        print(f"Results saved to: {output_path}")

        # Parse and organize results
        results = self.parse_results(output_path)

        # Save organized results
        organized_path = OUTPUT_DIR / f"batch_{batch_id}_organized.json"
        with open(organized_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

        print(f"Organized results saved to: {organized_path}")

        return results

    def parse_results(self, results_path):
        """Parse JSONL results into organized structure"""
        results = {
            'variants': {},      # q_id -> {transform_type: variant_text}
            'responses': {},     # q_id -> {variant_type: response_text}
            'errors': []
        }

        with open(results_path, 'r', encoding='utf-8') as f:
            for line in f:
                data = json.loads(line)
                custom_id = data['custom_id']

                if data.get('error'):
                    results['errors'].append({
                        'custom_id': custom_id,
                        'error': data['error']
                    })
                    continue

                # Parse custom_id: "q_id__type__subtype"
                parts = custom_id.split('__')
                q_id = parts[0]
                req_type = parts[1]  # 'variant' or 'response'
                subtype = parts[2]   # transform type or 'original'

                # Extract response text
                response_text = data['response']['body']['choices'][0]['message']['content']

                if req_type == 'variant':
                    if q_id not in results['variants']:
                        results['variants'][q_id] = {}
                    results['variants'][q_id][subtype] = response_text

                elif req_type == 'response':
                    if q_id not in results['responses']:
                        results['responses'][q_id] = {}
                    results['responses'][q_id][subtype] = response_text

        print(f"\nParsed results:")
        print(f"  Variants: {len(results['variants'])} questions")
        print(f"  Responses: {len(results['responses'])} questions")
        print(f"  Errors: {len(results['errors'])}")

        return results


def extract_variant_text(raw_text):
    """Extract variant text from JSON response (handles markdown code blocks)"""
    import re

    # Remove markdown code blocks
    text = re.sub(r'```json\s*', '', raw_text)
    text = re.sub(r'```\s*', '', text)
    text = text.strip()

    try:
        data = json.loads(text)
        return data.get('variant', text)
    except:
        return text


def prepare_variant_responses_batch(variants_file, prompts_path=PROMPTS_PATH, config_path=CONFIG_PATH):
    """
    Prepare batch for getting LLM responses to variants
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    with open(prompts_path, 'r') as f:
        prompts = yaml.safe_load(f)

    with open(variants_file, 'r', encoding='utf-8') as f:
        variants_data = json.load(f)

    requests = []
    interr_prompts = prompts['interrogator']
    model = config['interrogator']['target_models'][0]

    for q_id, variants in variants_data.get('variants', {}).items():
        for var_type, raw_variant in variants.items():
            variant_text = extract_variant_text(raw_variant)

            user_prompt = interr_prompts['user_template'].format(question=variant_text)

            request = {
                "custom_id": f"{q_id}__response__{var_type}",
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": {
                    "model": model,
                    "messages": [
                        {"role": "system", "content": interr_prompts['system_prompt']},
                        {"role": "user", "content": user_prompt}
                    ],
                    "temperature": config['interrogator']['temperature'],
                    "max_tokens": config['interrogator']['max_tokens']
                }
            }
            requests.append(request)

    # Save JSONL
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    filepath = OUTPUT_DIR / 'batch_variant_responses.jsonl'
    with open(filepath, 'w', encoding='utf-8') as f:
        for req in requests:
            f.write(json.dumps(req, ensure_ascii=False) + '\n')

    print(f"Saved {len(requests)} variant response requests to {filepath}")
    return filepath


def main():
    import argparse

    parser = argparse.ArgumentParser(description='OpenAI Batch API Helper')
    parser.add_argument('--prepare-variants', action='store_true',
                        help='Prepare variant generation batch')
    parser.add_argument('--prepare-responses', action='store_true',
                        help='Prepare response collection batch')
    parser.add_argument('--prepare-all', action='store_true',
                        help='Prepare both variants and responses')
    parser.add_argument('--submit', type=str, metavar='JSONL_FILE',
                        help='Submit a JSONL file as batch job')
    parser.add_argument('--check', type=str, metavar='BATCH_ID',
                        help='Check batch job status')
    parser.add_argument('--download', type=str, metavar='BATCH_ID',
                        help='Download batch results')
    parser.add_argument('--sample', type=int, default=None,
                        help='Number of questions to sample')

    args = parser.parse_args()

    helper = BatchAPIHelper()

    if args.prepare_variants or args.prepare_all:
        questions = load_dataset(args.sample)
        requests = helper.prepare_variant_requests(questions)
        jsonl_path = helper.save_jsonl(requests, 'batch_variants.jsonl')
        print(f"\nTo submit: python scripts/batch_api_helper.py --submit {jsonl_path}")

    if args.prepare_responses or args.prepare_all:
        questions = load_dataset(args.sample)
        requests = helper.prepare_interrogation_requests(questions)
        jsonl_path = helper.save_jsonl(requests, 'batch_responses.jsonl')
        print(f"\nTo submit: python scripts/batch_api_helper.py --submit {jsonl_path}")

    if args.submit:
        batch = helper.upload_and_create_batch(args.submit)

    if args.check:
        helper.check_status(args.check)

    if args.download:
        helper.download_results(args.download)

    if not any([args.prepare_variants, args.prepare_responses, args.prepare_all,
                args.submit, args.check, args.download]):
        parser.print_help()
        print("\n" + "="*60)
        print("HIZLI BAŞLANGIÇ:")
        print("="*60)
        print("\n1. Variant batch hazırla (50 soru ile test):")
        print("   python scripts/batch_api_helper.py --prepare-variants --sample 50")
        print("\n2. Batch'i gönder:")
        print("   python scripts/batch_api_helper.py --submit data/batch_results/batch_variants.jsonl")
        print("\n3. Durumu kontrol et:")
        print("   python scripts/batch_api_helper.py --check <BATCH_ID>")
        print("\n4. Sonuçları indir:")
        print("   python scripts/batch_api_helper.py --download <BATCH_ID>")


if __name__ == '__main__':
    main()
