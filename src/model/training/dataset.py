import logging
from pathlib import Path
from typing import Any, Dict

from datasets import load_dataset, Dataset

logger = logging.getLogger(__name__)

def is_valid_sample(sample: Dict[str, Any]) -> bool:
    """
    Validates a dataset sample to ensure it contains exactly 'messages'
    with 'role' and 'content'.
    
    Args:
        sample (Dict[str, Any]): A single record from the dataset.
        
    Returns:
        bool: True if the sample is valid, False otherwise.
    """
    messages = sample.get("messages", [])
    if not isinstance(messages, list):
        return False
        
    if len(messages) == 0:
        return False
        
    for msg in messages:
        if not isinstance(msg, dict):
            return False
        if "role" not in msg or "content" not in msg:
            return False
            
    return True

def load_and_validate_dataset(data_path: str) -> Dataset:
    """
    Loads a JSONL dataset using HuggingFace datasets and validates records.
    
    Args:
        data_path (str): The path to the JSONL dataset.
        
    Returns:
        Dataset: The validated HuggingFace Dataset object.
        
    Raises:
        FileNotFoundError: If the dataset file does not exist.
        ValueError: If the dataset is empty after validation.
    """
    if not Path(data_path).exists():
        logger.error(f"Dataset file not found: {data_path}")
        raise FileNotFoundError(f"Dataset file not found: {data_path}")
        
    logger.info(f"Loading dataset from {data_path}")
    
    # Load dataset using Hugging Face's load_dataset
    dataset = load_dataset("json", data_files=data_path, split="train")
    
    initial_count = len(dataset)
    logger.info(f"Loaded {initial_count} samples. Validating...")
    
    # Filter out invalid records
    dataset = dataset.filter(is_valid_sample)
    
    final_count = len(dataset)
    rejected = initial_count - final_count
    
    if final_count == 0:
        logger.error("All samples were rejected. Dataset is empty.")
        raise ValueError("Empty dataset after validation.")
        
    logger.info(f"Validation complete: {final_count} valid samples, {rejected} rejected.")
    
    return dataset
