import re
from typing import Dict, Any, List

def is_mostly_numbers(text: str) -> bool:
    """Check if the text is mostly numbers."""
    words = text.split()
    if not words:
        return True
    num_count = sum(1 for w in words if any(c.isdigit() for c in w))
    return num_count / len(words) > 0.6

def is_table_of_contents(text: str) -> bool:
    """Check if the text looks like a table of contents."""
    # Look for repeating dots or multiple lines ending in numbers
    lines = text.split('\n')
    toc_lines = [line for line in lines if re.search(r'\.{3,}\s*\d+\s*$', line) or re.search(r'\s\d+\s*$', line)]
    if len(lines) > 2 and len(toc_lines) / len(lines) > 0.5:
        return True
    return bool(re.search(r'(?i)\btable of contents\b', text[:100]))

def is_page_numbers_only(text: str) -> bool:
    """Check if the text is just page numbers or very short."""
    return bool(re.match(r'^\s*(page\s*)?\d+\s*$', text, re.IGNORECASE))

def is_references(text: str) -> bool:
    """Check if the text is a references or bibliography section."""
    return bool(re.search(r'(?i)\b(references|bibliography)\b', text[:100])) and len(text) < 500

def is_acknowledgements(text: str) -> bool:
    """Check if the text is an acknowledgements section."""
    return bool(re.search(r'(?i)\b(acknowledgements|acknowledgment)\b', text[:100]))

def has_repeated_punctuation(text: str) -> bool:
    """Check if the text has excessive repeated punctuation (e.g., .........)."""
    return bool(re.search(r'[\.\-\_]{10,}', text))

def is_ocr_garbage(text: str) -> bool:
    """Check for high non-alphanumeric ratio typical of OCR garbage."""
    if not text:
        return True
    non_alpha = len(re.findall(r'[^a-zA-Z0-9\s.,;:\'\"]', text))
    return (non_alpha / len(text)) > 0.2

def is_only_headings(text: str) -> bool:
    """Check if the text consists mostly of short lines indicating headings."""
    lines = [line.strip() for line in text.split('\n') if line.strip()]
    if not lines:
        return True
    # If all lines are very short
    return all(len(line.split()) < 8 for line in lines)

def filter_chunks(chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Filter bad chunks based on multiple criteria.

    Args:
        chunks: List of chunk dictionaries.

    Returns:
        List of retained chunk dictionaries.
    """
    retained_chunks = []
    
    for chunk in chunks:
        text = chunk.get('text', '')
        
        if not text:
            continue
        
        if len(text) < 200:
            continue
            
        if is_mostly_numbers(text):
            continue
            
        if is_table_of_contents(text):
            continue
            
        if is_page_numbers_only(text):
            continue
            
        if is_references(text):
            continue
            
        if is_acknowledgements(text):
            continue
            
        if has_repeated_punctuation(text):
            continue
            
        if is_ocr_garbage(text):
            continue
            
        if is_only_headings(text):
            continue
            
        retained_chunks.append(chunk)
        
    return retained_chunks
