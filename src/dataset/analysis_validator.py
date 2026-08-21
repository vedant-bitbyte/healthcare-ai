import json
import logging
from pathlib import Path
from typing import Dict, Any, List, Tuple

logger = logging.getLogger(__name__)

class DatasetValidator:
    """Validates instruction tuning datasets for required fields, duplicates, and structural integrity."""
    
    def __init__(self):
        self.total_samples = 0
        self.valid_samples = 0
        self.rejected_samples = 0
        
        self.duplicate_count = 0
        self.missing_values_count = 0
        
        self.seen_instructions = set()
        self.seen_outputs = set()
        self.seen_pairs = set()
        
        self.valid_records = []
        
    def validate_file(self, file_path: Path) -> List[Dict[str, Any]]:
        """
        Validates a single JSONL file.
        Returns a list of valid records.
        """
        logger.info(f"Validating file: {file_path}")
        if not file_path.exists():
            logger.error(f"File not found: {file_path}")
            return []
            
        file_valid_records = []
        
        with open(file_path, 'r', encoding='utf-8') as f:
            for line_idx, line in enumerate(f):
                if not line.strip():
                    continue
                    
                self.total_samples += 1
                
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    logger.warning(f"Invalid JSON at line {line_idx+1} in {file_path}")
                    self.rejected_samples += 1
                    continue
                    
                is_valid, reason = self._validate_record(record)
                if is_valid:
                    self.valid_samples += 1
                    file_valid_records.append(record)
                    self.valid_records.append(record)
                else:
                    self.rejected_samples += 1
                    logger.debug(f"Rejected record at line {line_idx+1} in {file_path}: {reason}")
                    
        return file_valid_records
        
    def _validate_record(self, record: Dict[str, Any]) -> Tuple[bool, str]:
        """Validates a single record according to predefined rules."""
        
        # 1. Check for missing values
        required_keys = [
            "instruction", "output", "source", "chunk_id", 
            "category", "difficulty", "quality_score"
        ]
        
        for key in required_keys:
            if key not in record:
                self.missing_values_count += 1
                return False, f"Missing field: {key}"
                
        # 2. Check for empty strings
        if not str(record["instruction"]).strip():
            self.missing_values_count += 1
            return False, "Empty instruction"
            
        if not str(record["output"]).strip():
            self.missing_values_count += 1
            return False, "Empty output"
            
        if not str(record["source"]).strip():
            self.missing_values_count += 1
            return False, "Empty source"
            
        if not str(record["category"]).strip():
            self.missing_values_count += 1
            return False, "Empty category"
            
        if not str(record["difficulty"]).strip():
            self.missing_values_count += 1
            return False, "Empty difficulty"
            
        # 3. Validate quality score
        try:
            score = float(record["quality_score"])
            if not (1 <= score <= 5):
                return False, f"Invalid quality score: {score}"
        except (ValueError, TypeError):
            return False, f"Quality score is not a number: {record['quality_score']}"
            
        # 4. Check for duplicates
        instruction = str(record["instruction"]).strip()
        output = str(record["output"]).strip()
        pair_hash = hash((instruction, output))
        
        is_duplicate = False
        
        if instruction in self.seen_instructions:
            is_duplicate = True
        if output in self.seen_outputs:
            is_duplicate = True
        if pair_hash in self.seen_pairs:
            is_duplicate = True
            
        if is_duplicate:
            self.duplicate_count += 1
            return False, "Duplicate detected (instruction, output, or pair)"
            
        # If valid, add to seen sets to catch future duplicates
        self.seen_instructions.add(instruction)
        self.seen_outputs.add(output)
        self.seen_pairs.add(pair_hash)
        
        return True, "Valid"

    def get_summary(self) -> Dict[str, int]:
        """Returns the summary statistics of the validation process."""
        return {
            "Total Samples": self.total_samples,
            "Valid Samples": self.valid_samples,
            "Rejected Samples": self.rejected_samples,
            "Duplicates": self.duplicate_count,
            "Missing Values": self.missing_values_count
        }
