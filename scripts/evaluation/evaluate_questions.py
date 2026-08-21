import os
import sys
import logging
from pathlib import Path
import pandas as pd

# Add project root to python path
sys.path.append(str(Path(__file__).parent.parent))

from src.evaluation.inference import load_model_for_inference, generate_answers

# Configure basic logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    # Setup paths
    base_model_name = "microsoft/Phi-3-mini-4k-instruct"
    adapter_path = "outputs"
    input_csv = "evaluation/evaluation_questions.csv"
    
    results_dir = Path("evaluation/results")
    output_csv = results_dir / "finetuned_phi3_results.csv"
    
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Ensure input data exists
    if not os.path.exists(input_csv):
        logger.error(f"Input file {input_csv} does not exist.")
        return
        
    logger.info(f"Loading questions from {input_csv}")
    df_questions = pd.read_csv(input_csv)
    
    if "question" not in df_questions.columns or "id" not in df_questions.columns:
        logger.error("Input CSV must contain 'id' and 'question' columns.")
        return
        
    questions = df_questions["question"].tolist()
    ids = df_questions["id"].tolist()
    
    # Load model and tokenizer
    logger.info("Loading model and LoRA adapter...")
    model, tokenizer = load_model_for_inference(base_model_name, adapter_path)
    
    # Generate answers
    logger.info(f"Generating answers for {len(questions)} questions...")
    # The generation parameters (temperature=0, do_sample=False) are handled inside generate_answers
    raw_results = generate_answers(model, tokenizer, questions, max_new_tokens=512)
    
    # Combine IDs with results
    final_results = []
    for q_id, res in zip(ids, raw_results):
        final_results.append({
            "id": q_id,
            "question": res["question"],
            "answer": res["answer"],
            "latency_ms": res["latency_ms"]
        })
        
    # Save to CSV
    df_output = pd.DataFrame(final_results)
    # Ensure column order matches requirement
    df_output = df_output[["id", "question", "answer", "latency_ms"]]
    df_output.to_csv(output_csv, index=False)
    
    logger.info(f"Saved results to {output_csv}")
    logger.info("Inference pipeline completed successfully.")

if __name__ == "__main__":
    main()
