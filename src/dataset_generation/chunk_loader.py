import json
import logging
from pathlib import Path
from typing import List, Dict, Any

logger = logging.getLogger(__name__)

def load_chunks(data_dir: str | Path) -> List[Dict[str, Any]]:
    """
    Load every JSON file from the given directory and merge chunks into one list.

    Args:
        data_dir: The path to the directory containing processed JSON files.

    Returns:
        A list of dictionaries representing the chunks.
    """
    data_dir_path = Path(data_dir)
    if not data_dir_path.exists() or not data_dir_path.is_dir():
        logger.error(f"Directory not found: {data_dir_path}")
        raise NotADirectoryError(f"{data_dir_path} is not a valid directory.")
    
    all_chunks = []
    
    for json_file in data_dir_path.glob("*.json"):
        logger.info(f"Loading chunks from {json_file.name}")
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                chunks = json.load(f)
                if isinstance(chunks, list):
                    all_chunks.extend(chunks)
                else:
                    logger.warning(f"File {json_file.name} does not contain a JSON list. Skipping.")
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse {json_file.name}: {e}")
        except Exception as e:
            logger.error(f"Error reading {json_file.name}: {e}")
            
    logger.info(f"Loaded a total of {len(all_chunks)} chunks.")
    return all_chunks
