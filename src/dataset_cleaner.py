import json
import logging
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, Any, List, Set
import requests
from tqdm import tqdm

from quality_checker import evaluate_rewrite

logger = logging.getLogger(__name__)

# The exact prompt requested by the user
SYSTEM_PROMPT = """You are rewriting examples for a supervised instruction-tuning dataset.

You will receive:
1. An instruction
2. A document excerpt

Rewrite ONLY the answer.

Rules:
- Never invent facts.
- Use ONLY the provided document.
- Remove OCR mistakes.
- Rewrite naturally.
- Do NOT copy long sentences.
- Explain tables instead of reproducing them.
- Explain figures instead of listing them.
- Preserve all numbers.
- Preserve all medical terminology.
- Remove broken sentences.
- Complete partially cut sentences whenever possible.
- Do not mention "the document says" or "the passage states".
- Produce a concise answer between 50 and 150 words.
- Make the response suitable for training a healthcare assistant.

Return ONLY the rewritten answer."""

def call_ollama(instruction: str, text: str) -> str:
    """
    Call local Ollama model to rewrite the answer.
    """
    user_prompt = f"Instruction:\n{instruction}\n\nDocument Excerpt:\n{text}"
    
    url = "http://localhost:11434/api/generate"
    payload = {
        "model": "gemma3:4b",
        "system": SYSTEM_PROMPT,
        "prompt": user_prompt,
        "stream": False,
        "options": {
            "temperature": 0.3
        }
    }
    
    try:
        response = requests.post(url, json=payload, timeout=60)
        response.raise_for_status()
        result = response.json()
        return result.get('response', '').strip()
    except Exception as e:
        logger.error(f"Ollama API call failed: {e}")
        return ""

class DatasetCleaner:
    def __init__(self, input_file: str, output_file: str, rejected_file: str, workers: int = 4):
        self.input_file = Path(input_file)
        self.output_file = Path(output_file)
        self.rejected_file = Path(rejected_file)
        self.workers = workers
        self.lock = threading.Lock()
        
        # Ensure parent directories exist
        self.output_file.parent.mkdir(parents=True, exist_ok=True)
        self.rejected_file.parent.mkdir(parents=True, exist_ok=True)
        
    def _load_processed_ids(self) -> Set[int]:
        """Load chunk_ids that have already been processed (either accepted or rejected)."""
        processed = set()
        
        for file_path in [self.output_file, self.rejected_file]:
            if file_path.exists():
                with open(file_path, 'r', encoding='utf-8') as f:
                    for line in f:
                        if not line.strip():
                            continue
                        try:
                            record = json.loads(line)
                            if 'chunk_id' in record:
                                processed.add(record['chunk_id'])
                        except json.JSONDecodeError:
                            continue
        return processed

    def _process_single_sample(self, record: Dict[str, Any]) -> Dict[str, Any]:
        """Process a single record: call Ollama, evaluate, return result dict."""
        chunk_id = record.get('chunk_id')
        instruction = record.get('instruction', '')
        original_output = record.get('output', '')
        
        rewritten = call_ollama(instruction, original_output)
        
        if not rewritten:
            return {
                'record': record,
                'status': 'error',
                'reason': 'Ollama returned empty or failed'
            }
            
        score, reason = evaluate_rewrite(original_output, rewritten)
        
        result_record = record.copy()
        result_record['output'] = rewritten
        result_record['quality_score'] = score
        
        status = 'accepted' if score >= 4 else 'rejected'
        
        return {
            'record': result_record,
            'status': status,
            'reason': reason
        }

    def _write_record(self, record: Dict[str, Any], filepath: Path):
        """Thread-safe file write."""
        with self.lock:
            with open(filepath, 'a', encoding='utf-8') as f:
                f.write(json.dumps(record, ensure_ascii=False) + '\n')

    def run(self):
        """Run the rewriting pipeline."""
        if not self.input_file.exists():
            logger.error(f"Input file not found: {self.input_file}")
            return
            
        processed_ids = self._load_processed_ids()
        logger.info(f"Loaded {len(processed_ids)} already processed chunk IDs.")
        
        to_process = []
        with open(self.input_file, 'r', encoding='utf-8') as f:
            for line in f:
                if not line.strip():
                    continue
                record = json.loads(line)
                if record.get('chunk_id') not in processed_ids:
                    to_process.append(record)
                    
        total = len(to_process)
        if total == 0:
            logger.info("No new records to process.")
            return
            
        logger.info(f"Starting processing of {total} records with {self.workers} workers.")
        
        accepted_count = 0
        rejected_count = 0
        error_count = 0
        
        with ThreadPoolExecutor(max_workers=self.workers) as executor:
            # Submit all tasks
            future_to_record = {executor.submit(self._process_single_sample, rec): rec for rec in to_process}
            
            with tqdm(total=total, desc="Rewriting", unit="samples") as pbar:
                for future in as_completed(future_to_record):
                    try:
                        result = future.result()
                        status = result['status']
                        record = result['record']
                        reason = result['reason']
                        
                        if status == 'accepted':
                            self._write_record(record, self.output_file)
                            accepted_count += 1
                        elif status == 'rejected':
                            record['reject_reason'] = reason
                            self._write_record(record, self.rejected_file)
                            rejected_count += 1
                            logger.info(f"Rejected chunk_id {record.get('chunk_id')}: {reason}")
                        else:
                            error_count += 1
                            logger.error(f"Error processing chunk_id {record.get('chunk_id')}: {reason}")
                            
                        # Update tqdm postfix with stats
                        pbar.set_postfix({'Acc': accepted_count, 'Rej': rejected_count, 'Err': error_count})
                        pbar.update(1)
                        
                    except Exception as e:
                        logger.error(f"Exception during processing: {e}")
                        error_count += 1
                        pbar.update(1)

        logger.info("Processing complete.")
        logger.info(f"Accepted: {accepted_count}, Rejected: {rejected_count}, Errors: {error_count}")
