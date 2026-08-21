import json
import logging
import random
from collections import Counter
from pathlib import Path
from typing import Dict, Any, List

logger = logging.getLogger(__name__)

def shuffle_and_split(dataset: List[Dict[str, Any]], seed: int = 42):
    """
    Shuffle dataset and split into 80% train, 10% validation, 10% test.
    """
    rng = random.Random(seed)
    rng.shuffle(dataset)
    
    total = len(dataset)
    train_end = int(0.8 * total)
    val_end = int(0.9 * total)
    
    train = dataset[:train_end]
    val = dataset[train_end:val_end]
    test = dataset[val_end:]
    
    return train, val, test

def export_jsonl(dataset: List[Dict[str, Any]], filepath: str | Path):
    """Export dataset to JSONL format."""
    path = Path(filepath)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(path, 'w', encoding='utf-8') as f:
        for record in dataset:
            f.write(json.dumps(record, ensure_ascii=False) + '\n')

def generate_report(dataset: List[Dict[str, Any]], report_path: str | Path):
    """
    Generate output/report.md with dataset statistics and ASCII graphs.
    """
    path = Path(report_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    total_examples = len(dataset)
    
    if total_examples == 0:
        with open(path, 'w', encoding='utf-8') as f:
            f.write("# Dataset Quality Report\n\nNo examples generated.\n")
        return
        
    categories = Counter(r.get('category', 'Unknown') for r in dataset)
    documents = Counter(r.get('source', 'Unknown') for r in dataset)
    
    avg_ans_len = sum(len(r.get('output', '').split()) for r in dataset) / total_examples
    avg_inst_len = sum(len(r.get('instruction', '').split()) for r in dataset) / total_examples
    
    # ASCII Graph generation helper
    def generate_ascii_bar(count: int, max_count: int, max_width: int = 40) -> str:
        if max_count == 0:
            return ""
        width = int((count / max_count) * max_width)
        return "█" * width
        
    max_cat = max(categories.values()) if categories else 1
    
    with open(path, 'w', encoding='utf-8') as f:
        f.write("# Dataset Quality Report\n\n")
        f.write(f"**Total examples:** {total_examples}\n")
        f.write(f"**Average instruction length:** {avg_inst_len:.2f} words\n")
        f.write(f"**Average answer length:** {avg_ans_len:.2f} words\n\n")
        
        f.write("## Examples per Category\n\n")
        f.write("```text\n")
        for cat, count in categories.most_common():
            bar = generate_ascii_bar(count, max_cat)
            f.write(f"{cat:<25} | {count:<5} | {bar}\n")
        f.write("```\n\n")
        
        f.write("## Top Source Documents\n\n")
        for doc, count in documents.most_common(10):
            f.write(f"- **{doc}**: {count} examples\n")
