from typing import Dict, Any, List

def deduplicate_dataset(dataset: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Remove duplicate instructions and duplicate outputs.
    Keep the highest quality example (e.g., longest output/instruction combination).
    """
    # Sort dataset by quality (length of output + instruction) descending
    # so we keep the highest quality one when deduplicating.
    sorted_dataset = sorted(
        dataset, 
        key=lambda x: len(x.get('output', '')) + len(x.get('instruction', '')),
        reverse=True
    )
    
    seen_instructions = set()
    seen_outputs = set()
    
    deduplicated = []
    
    for record in sorted_dataset:
        instruction = record.get('instruction', '').strip().lower()
        output = record.get('output', '').strip().lower()
        
        # We need to ensure we don't have exact duplicate instructions
        # OR exact duplicate outputs (if an output is identical, it's redundant to train on it again)
        # However, the user states: "Remove duplicate instructions. Remove duplicate outputs."
        if instruction in seen_instructions or output in seen_outputs:
            continue
            
        seen_instructions.add(instruction)
        seen_outputs.add(output)
        
        deduplicated.append(record)
        
    return deduplicated
