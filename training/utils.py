import logging
from pathlib import Path

def setup_logging(log_file_name: str = "train_phi3.log") -> None:
    """
    Sets up central logging for the training pipeline.
    
    Creates a 'logs' directory if it doesn't exist and configures
    both console and file logging.
    
    Args:
        log_file_name (str): The name of the log file to create.
    """
    log_dir = Path("logs")
    log_dir.mkdir(parents=True, exist_ok=True)
    
    log_file = log_dir / log_file_name
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file, encoding='utf-8'),
            logging.StreamHandler()
        ]
    )
