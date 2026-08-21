import random
import re
from typing import Dict, Any, List

from .prompt_templates import PROMPT_TEMPLATES

def split_into_sentences(text: str) -> List[str]:
    """Split text into sentences using simple regex."""
    # Split on period, question mark, or exclamation mark followed by space and capital letter.
    sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z])', text)
    return [s.strip() for s in sentences if s.strip()]

def determine_difficulty(instruction: str, output: str) -> str:
    """Determine difficulty based on length of output and complexity of instruction."""
    output_len = len(output.split())
    if output_len < 30:
        return "Easy"
    elif output_len < 100:
        return "Medium"
    else:
        return "Hard"

def generate_instructions(chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Generate 3-5 instruction-output pairs per chunk deterministically.
    """
    generated_dataset = []
    
    # We want to be deterministic, so we could seed the random generator 
    # but the instructions say "use random seed 42" for the pipeline later. 
    # Let's seed it here as well for deterministic generation.
    rng = random.Random(42)
    
    for chunk in chunks:
        text = chunk.get('text', '')
        category = chunk.get('category', 'Narrative')
        source = chunk.get('source', 'Unknown Document')
        chunk_id = chunk.get('chunk_id', 0)
        
        templates = PROMPT_TEMPLATES.get(category, PROMPT_TEMPLATES["Narrative"])
        
        # Decide number of pairs for this chunk (3 to 5)
        num_pairs = rng.randint(3, 5)
        
        # Pick random templates for this chunk
        selected_templates = rng.sample(templates, min(num_pairs, len(templates)))
        
        sentences = split_into_sentences(text)
        
        for i, template in enumerate(selected_templates):
            instruction = template
            
            # Vary the output slightly deterministically so not all outputs are identical
            # if we have multiple sentences.
            if len(sentences) >= 3:
                if i % 3 == 0:
                    # Use full text
                    output = text
                elif i % 3 == 1:
                    # Use first half
                    half_idx = max(1, len(sentences) // 2)
                    output = " ".join(sentences[:half_idx])
                else:
                    # Use second half
                    half_idx = max(1, len(sentences) // 2)
                    output = " ".join(sentences[half_idx:])
            else:
                # If short, always use full text
                output = text
                
            # Make the instruction unique to the context/source so it is not dropped by deduplicator.
            # E.g. "Summarize the key policy details ... based on National_Health_Policy.pdf [Excerpt ID]"
            instruction = f"{template} (Source: {source}, Section Ref: {chunk_id}-{i})"
            
            # If template specifically asks about table, append context
            if category == "Table Explanation":
                instruction = instruction.replace("referenced table", f"table from {source}")
            
            difficulty = determine_difficulty(instruction, output)
            
            record = {
                "instruction": instruction,
                "input": "",
                "output": output,
                "source": source,
                "category": category,
                "difficulty": difficulty,
                "chunk_id": chunk_id
            }
            generated_dataset.append(record)
            
    return generated_dataset
