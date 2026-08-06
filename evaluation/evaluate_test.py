import os
import sys
import json
import logging
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Add project root to python path to import modules if necessary
sys.path.append(str(Path(__file__).parent.parent))

from evaluation.inference import load_model_for_inference, generate_responses
from evaluation.metrics import calculate_all_metrics, aggregate_metrics

# Configure basic logging for evaluation
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def load_test_data(data_path: str) -> list:
    """Loads the test_chat.jsonl dataset."""
    logger.info(f"Loading test data from {data_path}")
    data = []
    with open(data_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data

def plot_distribution(data: list, metric: str, title: str, xlabel: str, output_path: str):
    """Generates a distribution plot for a specific metric and saves it."""
    plt.figure(figsize=(8, 5))
    sns.histplot([d[metric] for d in data if metric in d], bins=20, kde=True, color='skyblue')
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()

def main():
    # Setup paths
    base_model_name = "microsoft/Phi-3-mini-4k-instruct"
    adapter_path = "outputs"
    test_data_path = "data/chat/test_chat.jsonl"
    
    results_dir = Path("evaluation/results")
    plots_dir = results_dir / "plots"
    
    results_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)
    
    # Load dataset
    test_data = load_test_data(test_data_path)
    
    # Ensure test data exists
    if not test_data:
        logger.error("Test data is empty. Exiting.")
        return
        
    # Load model and tokenizer
    model, tokenizer = load_model_for_inference(base_model_name, adapter_path)
    
    # Generate responses
    logger.info("Generating responses on test data...")
    # NOTE: Set max_new_tokens according to the expected response length.
    raw_results = generate_responses(model, tokenizer, test_data, max_new_tokens=512)
    
    # Calculate metrics
    logger.info("Calculating metrics...")
    final_results = calculate_all_metrics(raw_results)
    
    # Save per-sample results to CSV
    predictions_csv = results_dir / "test_predictions.csv"
    df = pd.DataFrame(final_results)
    df.to_csv(predictions_csv, index=False)
    logger.info(f"Saved per-sample predictions and metrics to {predictions_csv}")
    
    # Aggregate and save metrics to JSON
    metrics = aggregate_metrics(final_results)
    metrics_json = results_dir / "test_metrics.json"
    with open(metrics_json, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=4)
    logger.info(f"Saved aggregated metrics to {metrics_json}")
    
    # Generate Plots
    logger.info("Generating publication-quality plots...")
    plot_distribution(final_results, "bleu", "BLEU Score Distribution", "BLEU", plots_dir / "bleu_distribution.png")
    plot_distribution(final_results, "rouge_l", "ROUGE-L Score Distribution", "ROUGE-L", plots_dir / "rouge_l_distribution.png")
    plot_distribution(final_results, "bert_score", "BERTScore Distribution", "BERTScore F1", plots_dir / "bertscore_distribution.png")
    plot_distribution(final_results, "latency_sec", "Inference Latency Distribution", "Seconds per sample", plots_dir / "latency_distribution.png")
    plot_distribution(final_results, "response_length_tokens", "Response Length Distribution", "Number of Tokens", plots_dir / "response_length_distribution.png")
    
    logger.info("Evaluation completed successfully.")

if __name__ == "__main__":
    main()
