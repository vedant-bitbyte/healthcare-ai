import argparse
import logging
import sys
from pathlib import Path

# Add src to python path so we can import dataset_cleaner
sys.path.append(str(Path(__file__).parent.parent / "src"))

from dataset_cleaner import DatasetCleaner

def setup_logging():
    log_dir = Path(__file__).parent.parent / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    
    log_file = log_dir / "rewrite.log"
    
    # Configure logging to file and a simpler format for console
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file, encoding='utf-8'),
            # We don't want to clutter the console too much since tqdm is running
            # so we only log warnings/errors to console, or write to a file
        ]
    )
    
    # Optionally add a console handler if desired, but tqdm handles console output beautifully
    console = logging.StreamHandler()
    console.setLevel(logging.WARNING)
    formatter = logging.Formatter('%(levelname)s: %(message)s')
    console.setFormatter(formatter)
    logging.getLogger('').addHandler(console)

def main():
    parser = argparse.ArgumentParser(description="Rewrite dataset examples using local Ollama model.")
    parser.add_argument("--input", required=True, help="Input JSONL file to process.")
    parser.add_argument("--output", required=True, help="Output JSONL file for accepted clean samples.")
    parser.add_argument("--rejected", default="data/instructions/rejected.jsonl", help="Output JSONL file for rejected samples.")
    parser.add_argument("--workers", type=int, default=4, help="Number of concurrent workers for Ollama API.")
    
    args = parser.parse_args()
    
    setup_logging()
    
    cleaner = DatasetCleaner(
        input_file=args.input,
        output_file=args.output,
        rejected_file=args.rejected,
        workers=args.workers
    )
    
    cleaner.run()

if __name__ == "__main__":
    main()
