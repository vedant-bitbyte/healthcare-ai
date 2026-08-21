import pytest
from src.dataset.quality_checker import evaluate_rewrite

def test_too_short():
    original = "This is a long original answer. " * 10
    rewritten = "This is too short."
    score, reason = evaluate_rewrite(original, rewritten)
    assert score == 1
    assert "Too short" in reason

def test_too_long():
    original = "This is a long original answer. " * 10
    rewritten = "This is a very long text. " * 50 # 250 words
    score, reason = evaluate_rewrite(original, rewritten)
    assert score == 1
    assert "Too long" in reason

def test_starts_with_punctuation():
    original = "Some text."
    rewritten = "- This starts with punctuation. " + "word " * 30
    score, reason = evaluate_rewrite(original, rewritten)
    assert score == 1
    assert "Starts with punctuation" in reason

def test_starts_mid_sentence():
    original = "Some text."
    rewritten = "because it starts with lowercase. " + "word " * 30
    score, reason = evaluate_rewrite(original, rewritten)
    assert score == 1
    assert "Starts mid-sentence" in reason

def test_contains_table():
    original = "Some text."
    rewritten = "This is a good summary, but as seen in Table 1. " + "a b c d e f g h i j k l m n o p q r s t u v w x y z aa bb cc dd"
    score, reason = evaluate_rewrite(original, rewritten)
    assert score == 1
    assert "Contains 'Table'" in reason

def test_too_similar():
    original = "This is exactly the same sentence structure as the original output. And here are a few more words."
    # 25 words total, intersection is 12. 12/25 = 48% overlap
    rewritten = "This is exactly the same sentence structure as the original output. a b c d e f g h i j k l m"
    score, reason = evaluate_rewrite(original, rewritten)
    assert score == 1
    assert "Too similar" in reason

def test_excellent_rewrite():
    original = "The NFHS-5 report states that 67 percent of children have anaemia."
    rewritten = "According to the provided document, approximately two-thirds of children suffer from some form of anaemia, indicating a significant public health challenge that requires immediate nutritional intervention and policy changes."
    # Wait, the rewritten shouldn't have "According to the provided document" if we want a clean output, but the rules didn't hard-forbid that exact phrase, only "the document says". Let's use a cleaner one.
    rewritten = "Approximately two-thirds of children suffer from some form of anaemia, indicating a significant public health challenge that requires immediate nutritional intervention and policy changes across the nation to address this widespread issue effectively."
    # ensure it's >= 20 words
    score, reason = evaluate_rewrite(original, rewritten)
    assert score >= 4 # Should pass
