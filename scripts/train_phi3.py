import sys
import logging
import argparse
from pathlib import Path

# Add project root to Python path so we can import 'training' package
sys.path.append(str(Path(__file__).parent.parent))

from src.model.training import TrainingConfig, setup_logging, train

logger = logging.getLogger(__name__)

def main():
    """
    Entry point for running the Phi-3 Mini QLoRA fine-tuning process.
    """
    parser = argparse.ArgumentParser(description="Run Phi-3 Mini QLoRA Fine-tuning")
    parser.add_argument("--resume", action="store_true", help="Resume training from the latest checkpoint")
    args = parser.parse_args()

    # 1. Setup logging
    setup_logging("train_phi3.log")
    logger.info("Initializing Phi-3 Mini QLoRA training script...")
    
    # 2. Load Configuration
    config = TrainingConfig()
    if args.resume:
        config.resume_from_checkpoint = True
        logger.info("Resume flag detected. Training will resume from the latest checkpoint.")
    
    # 3. Start Training
    try:
        train(config)
        logger.info("Training script completed successfully.")
    except Exception as e:
        logger.error(f"Training failed with error: {e}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main()
