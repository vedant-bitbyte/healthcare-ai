import logging
from typing import List, Dict, Any

from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer

logger = logging.getLogger(__name__)

# Initialize evaluate modules conditionally if needed, but BERTScore is easier via evaluate library or bert_score package
# We added bert-score to pip, so we can use the evaluate library wrapper or raw bert_score
import bert_score

def compute_bleu(prediction: str, reference: str) -> float:
    """Computes BLEU score with smoothing."""
    # Simple whitespace tokenization for BLEU
    ref_tokens = reference.split()
    pred_tokens = prediction.split()
    
    # If prediction is empty, BLEU is 0
    if not pred_tokens:
        return 0.0
        
    smooth = SmoothingFunction().method1
    return sentence_bleu([ref_tokens], pred_tokens, smoothing_function=smooth)

def compute_rouge(prediction: str, reference: str) -> float:
    """Computes ROUGE-L f-measure."""
    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    scores = scorer.score(reference, prediction)
    return scores['rougeL'].fmeasure

def compute_exact_match(prediction: str, reference: str) -> int:
    """Computes whether prediction exactly matches the reference (1 or 0)."""
    return int(prediction.strip().lower() == reference.strip().lower())

def compute_bertscore(predictions: List[str], references: List[str]) -> List[float]:
    """
    Computes BERTScore f1-measures for a batch of strings.
    We batch this because loading the model for each string is inefficient.
    """
    logger.info("Computing BERTScore...")
    # lang="en" uses roberta-large by default
    _, _, f1 = bert_score.score(predictions, references, lang="en", verbose=True)
    return f1.tolist()

def calculate_all_metrics(results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Takes a list of dictionaries with 'prediction' and 'ground_truth'.
    Adds 'bleu', 'rouge_l', 'exact_match', and 'bert_score' to each dictionary.
    """
    predictions = [res["prediction"] for res in results]
    references = [res["ground_truth"] for res in results]
    
    # Compute batched metrics
    bert_scores = compute_bertscore(predictions, references)
    
    # Compute per-sample metrics
    for idx, res in enumerate(results):
        pred = res["prediction"]
        ref = res["ground_truth"]
        
        res["bleu"] = compute_bleu(pred, ref)
        res["rouge_l"] = compute_rouge(pred, ref)
        res["exact_match"] = compute_exact_match(pred, ref)
        res["bert_score"] = bert_scores[idx]
        
    return results

def aggregate_metrics(results: List[Dict[str, Any]]) -> Dict[str, float]:
    """
    Computes the average across all metrics.
    """
    total = len(results)
    if total == 0:
        return {}
        
    avg = {
        "avg_bleu": sum(r["bleu"] for r in results) / total,
        "avg_rouge_l": sum(r["rouge_l"] for r in results) / total,
        "avg_exact_match": sum(r["exact_match"] for r in results) / total,
        "avg_bert_score": sum(r["bert_score"] for r in results) / total,
        "avg_latency_sec": sum(r.get("latency_sec", 0.0) for r in results) / total,
        "avg_response_length": sum(r.get("response_length_tokens", 0) for r in results) / total,
    }
    
    return avg
