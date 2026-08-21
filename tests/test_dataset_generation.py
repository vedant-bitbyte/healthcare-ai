import pytest
from src.dataset.chunk_filter import filter_chunks
from src.dataset.classifier import classify_chunk, classify_chunks
from src.dataset.instruction_generator import generate_instructions
from src.dataset.generation_validator import validate_dataset
from src.dataset.deduplicator import deduplicate_dataset

def test_filter_chunks():
    chunks = [
        {"text": "A" * 150}, # Too short
        {"text": "This is a valid chunk with multiple words so it passes the heading check. " * 5}, # Valid, 200+ chars
        {"text": "123 456 789 " * 20}, # Mostly numbers
        {"text": "Table of Contents\n1. Introduction 5\n2. Body 10\n"}, # TOC
        {"text": "Page 5\n"} # Page numbers
    ]
    
    filtered = filter_chunks(chunks)
    assert len(filtered) == 1
    assert filtered[0]["text"].startswith("This is a valid chunk")

def test_classify_chunk():
    policy_text = "The new policy guidelines mandate a framework for healthcare."
    assert classify_chunk(policy_text) == "Policy"
    
    stats_text = "The rate is 50% with 10,000 cases of new data."
    assert classify_chunk(stats_text) == "Statistics"
    
    gen_text = "Hello world this is a test."
    # With <100 words and no matches it's General Healthcare
    assert classify_chunk(gen_text) == "General Healthcare"

def test_generate_instructions():
    chunks = [
        {"chunk_id": 1, "text": "This is the first sentence. This is the second. This is the third.", "category": "Policy", "source": "test.pdf"}
    ]
    
    instructions = generate_instructions(chunks)
    assert 3 <= len(instructions) <= 5
    for inst in instructions:
        assert "instruction" in inst
        assert "output" in inst
        assert inst["category"] == "Policy"

def test_validate_dataset():
    dataset = [
        {"instruction": "Valid instruction?", "output": "Valid output that is long enough to pass." * 3},
        {"instruction": "Short", "output": "Valid output that is long enough to pass." * 3}, # Invalid instruction
        {"instruction": "Valid instruction?", "output": "Short"}, # Invalid output
        {"instruction": "", "output": "Valid output that is long enough to pass." * 3} # Empty instruction
    ]
    
    valid = validate_dataset(dataset)
    assert len(valid) == 1

def test_deduplicate_dataset():
    dataset = [
        {"instruction": "Same instruction", "output": "Output A " * 10},
        {"instruction": "Same instruction", "output": "Output B " * 10}, # Duplicate instruction
        {"instruction": "Unique 1", "output": "Same output " * 10},
        {"instruction": "Unique 2", "output": "Same output " * 10} # Duplicate output
    ]
    
    dedup = deduplicate_dataset(dataset)
    assert len(dedup) == 2
