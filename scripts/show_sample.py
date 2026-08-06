import json
import random
import argparse
from pathlib import Path

def show_random_sample(file_path: Path):
    if not file_path.exists():
        print(f"File not found: {file_path}")
        return
        
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = [line.strip() for line in f if line.strip()]
        
    if not lines:
        print(f"File is empty: {file_path}")
        return
        
    random_line = random.choice(lines)
    
    try:
        sample = json.loads(random_line)
    except json.JSONDecodeError:
        print("Error decoding JSON.")
        return
        
    print("===================================================")
    
    messages = sample.get("messages", [])
    for msg in messages:
        role = str(msg.get("role", "")).upper()
        content = msg.get("content", "")
        
        print(f"\n{role}\n")
        print(content)
        print("\n---------------------------------------------------")
        
    print("\nMetadata\n")
    metadata = sample.get("metadata", {})
    print(f"Source: {metadata.get('source', '')}")
    print(f"Category: {metadata.get('category', '')}")
    print(f"Difficulty: {metadata.get('difficulty', '')}")
    print(f"Quality: {metadata.get('quality_score', '')}")
    
    print("\n===================================================")

def main():
    parser = argparse.ArgumentParser(description="Show a random sample from the converted chat dataset.")
    parser.add_argument("--file", default="data/chat/train_chat.jsonl", help="Path to a converted JSONL file")
    
    args = parser.parse_args()
    show_random_sample(Path(args.file))

if __name__ == "__main__":
    main()
