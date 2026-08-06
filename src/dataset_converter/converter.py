import json
import logging
from pathlib import Path
from typing import Dict, Any, Tuple

from .validator import ChatValidator

logger = logging.getLogger(__name__)

class DatasetConverter:
    """Converts instruction tuning dataset into chat format for Phi-3 Mini fine-tuning."""
    
    def __init__(self, output_dir: str = "data/chat"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def convert_record(self, record: Dict[str, Any]) -> Dict[str, Any]:
        """Transforms a single record into the chat format with separated metadata."""
        chat_format = {
            "messages": [
                {
                    "role": "user",
                    "content": str(record.get("instruction", ""))
                },
                {
                    "role": "assistant",
                    "content": str(record.get("output", ""))
                }
            ],
            "metadata": {
                "source": record.get("source", ""),
                "chunk_id": record.get("chunk_id", ""),
                "category": record.get("category", ""),
                "difficulty": record.get("difficulty", ""),
                "quality_score": record.get("quality_score", 0)
            }
        }
        return chat_format
        
    def convert_file(self, input_path: Path, output_filename: str) -> Tuple[int, int]:
        """
        Converts a single JSONL file and saves it to the output directory.
        Returns a tuple of (converted_count, rejected_count).
        """
        if not input_path.exists():
            logger.error(f"Input file not found: {input_path}")
            return 0, 0
            
        output_path = self.output_dir / output_filename
        converted_count = 0
        rejected_count = 0
        
        logger.info(f"Converting {input_path} -> {output_path}")
        
        with open(input_path, 'r', encoding='utf-8') as infile, \
             open(output_path, 'w', encoding='utf-8') as outfile:
             
            for line_idx, line in enumerate(infile):
                if not line.strip():
                    continue
                    
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    logger.warning(f"Invalid JSON at line {line_idx+1} in {input_path}")
                    rejected_count += 1
                    continue
                    
                chat_record = self.convert_record(record)
                
                is_valid, reason = ChatValidator.validate(chat_record)
                
                if is_valid:
                    outfile.write(json.dumps(chat_record, ensure_ascii=False) + '\n')
                    converted_count += 1
                else:
                    logger.debug(f"Rejected record at line {line_idx+1}: {reason}")
                    rejected_count += 1
                    
        return converted_count, rejected_count
