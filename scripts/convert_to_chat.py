import sys
import logging
from pathlib import Path
import argparse

# Add src to Python path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from src.dataset import DatasetConverter

def setup_logging():
    log_dir = Path("logs")
    log_dir.mkdir(parents=True, exist_ok=True)
    
    log_file = log_dir / "chat_conversion.log"
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file, encoding='utf-8')
        ]
    )

def main():
    parser = argparse.ArgumentParser(description="Convert cleaned datasets into Phi-3 chat format.")
    parser.add_argument("--train", default="output/train_clean.jsonl", help="Train dataset path")
    parser.add_argument("--val", default="output/validation_clean.jsonl", help="Validation dataset path")
    parser.add_argument("--test", default="output/test_clean.jsonl", help="Test dataset path")
    parser.add_argument("--outdir", default="data/chat", help="Output directory for chat datasets")
    
    args = parser.parse_args()
    
    setup_logging()
    logger = logging.getLogger("convert_to_chat")
    logger.info("Starting dataset conversion to chat format...")
    
    converter = DatasetConverter(output_dir=args.outdir)
    
    # Process files
    train_conv, train_rej = converter.convert_file(Path(args.train), "train_chat.jsonl")
    val_conv, val_rej = converter.convert_file(Path(args.val), "validation_chat.jsonl")
    test_conv, test_rej = converter.convert_file(Path(args.test), "test_chat.jsonl")
    
    total_rejected = train_rej + val_rej + test_rej
    
    # Print Console Summary
    print("-" * 40)
    print(f"Train converted     : {train_conv}")
    print(f"Validation converted: {val_conv}")
    print(f"Test converted      : {test_conv}")
    print(f"Rejected            : {total_rejected}")
    print(f"Output directory    : {args.outdir}")
    print("-" * 40)
    
    logger.info("Dataset conversion completed.")

if __name__ == "__main__":
    main()
