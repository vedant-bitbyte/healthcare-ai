import re
from typing import Tuple

def compute_similarity(text1: str, text2: str) -> float:
    """Compute basic word overlap similarity between two strings."""
    words1 = set(text1.lower().split())
    words2 = set(text2.lower().split())
    
    if not words1 or not words2:
        return 0.0
        
    intersection = words1.intersection(words2)
    # Return percentage of original words that are in the new text
    return len(intersection) / max(len(words1), len(words2))

def has_repeated_words(text: str) -> bool:
    """Check if the text has consecutive repeated words."""
    words = text.lower().split()
    for i in range(len(words) - 1):
        if words[i] == words[i+1] and len(words[i]) > 1: # Ignore repeated single chars if any
            return True
    return False

def is_ocr_garbage(text: str) -> bool:
    """Check for high non-alphanumeric ratio typical of OCR garbage."""
    if not text:
        return True
    non_alpha = len(re.findall(r'[^a-zA-Z0-9\s.,;:\'\"]', text))
    return (non_alpha / len(text)) > 0.15

def contains_page_numbers(text: str) -> bool:
    """Check if the text contains isolated page number patterns."""
    return bool(re.search(r'(?i)\b(page\s*\d+|p\.\s*\d+)\b', text))

def is_only_numbers(text: str) -> bool:
    """Check if the text is primarily just numbers."""
    words = text.split()
    if not words:
        return False
    num_count = sum(1 for w in words if any(c.isdigit() for c in w))
    return num_count / len(words) > 0.7

def evaluate_rewrite(original: str, rewritten: str) -> Tuple[int, str]:
    """
    Evaluate the rewritten text against the quality rules.
    Returns (score, rejection_reason).
    Score: 1 (reject), 2 (weak), 3 (acceptable), 4 (good), 5 (excellent).
    We only keep >= 4.
    """
    if not rewritten:
        return 1, "Empty output"
        
    words = rewritten.split()
    word_count = len(words)
    
    # Check length
    if word_count < 20:
        return 1, "Too short (<20 words)"
    if word_count > 180:
        return 1, "Too long (>180 words)"
        
    # Starts with punctuation
    if re.match(r'^[^\w\s]', rewritten):
        return 1, "Starts with punctuation"
        
    # Starts mid-sentence (lowercase first letter of first word)
    if words and words[0][0].islower():
        return 1, "Starts mid-sentence (lowercase)"
        
    # OCR garbage
    if is_ocr_garbage(rewritten):
        return 1, "Contains OCR garbage"
        
    # Repeated words
    if has_repeated_words(rewritten):
        return 1, "Contains repeated words"
        
    # Forbidden words
    if re.search(r'\bTable\b', rewritten, re.IGNORECASE):
        return 1, "Contains 'Table'"
    if re.search(r'\bFigure\b', rewritten, re.IGNORECASE):
        return 1, "Contains 'Figure'"
        
    # Page numbers
    if contains_page_numbers(rewritten):
        return 1, "Contains page numbers"
        
    # Only numbers
    if is_only_numbers(rewritten):
        return 1, "Contains only numbers"
        
    # Similarity check
    sim = compute_similarity(original, rewritten)
    if sim > 0.30:
        return 1, f"Too similar to original ({sim:.1%} overlap)"
        
    # Assign score based on length and similarity (for valid ones)
    # 5: very different from original and good length (50-150 words)
    # 4: passes all, maybe slightly similar (20-30%) or slightly short/long
    
    if 50 <= word_count <= 150 and sim < 0.20:
        return 5, ""
    
    return 4, ""
