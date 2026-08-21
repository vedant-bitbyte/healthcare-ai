from typing import Dict, Any, List

def validate_dataset(dataset: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Validate generated dataset pairs.
    Reject examples if:
    - instruction empty
    - output empty
    - output shorter than 30 characters
    - instruction shorter than 10 characters
    """
    valid_dataset = []
    
    for record in dataset:
        instruction = record.get('instruction', '').strip()
        output = record.get('output', '').strip()
        
        if not instruction:
            continue
        if not output:
            continue
        if len(output) < 30:
            continue
        if len(instruction) < 10:
            continue
            
        valid_dataset.append(record)
        
    return valid_dataset
