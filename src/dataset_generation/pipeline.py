import logging
from collections import Counter
from pathlib import Path

from .chunk_loader import load_chunks
from .chunk_filter import filter_chunks
from .classifier import classify_chunks
from .instruction_generator import generate_instructions
from .validator import validate_dataset
from .deduplicator import deduplicate_dataset
from .exporter import shuffle_and_split, export_jsonl, generate_report

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DatasetGenerationPipeline:
    def __init__(self, data_dir: str, output_dir: str):
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        
    def run(self):
        logger.info("Starting Dataset Generation Pipeline...")
        
        # 1. Load chunks
        all_chunks = load_chunks(self.data_dir)
        total_loaded = len(all_chunks)
        
        # 2. Filter bad chunks
        retained_chunks = filter_chunks(all_chunks)
        total_filtered = total_loaded - len(retained_chunks)
        total_retained = len(retained_chunks)
        logger.info(f"Filtered {total_filtered} bad chunks. Retained {total_retained}.")
        
        # 3. Classify chunks
        classified_chunks = classify_chunks(retained_chunks)
        logger.info("Classified chunks into categories.")
        
        # 4 & 5. Instruction Generation
        instructions = generate_instructions(classified_chunks)
        total_generated = len(instructions)
        logger.info(f"Generated {total_generated} instructions.")
        
        # 7. Validation
        valid_instructions = validate_dataset(instructions)
        total_removed_validation = total_generated - len(valid_instructions)
        logger.info(f"Removed {total_removed_validation} instructions during validation.")
        
        # 8. Deduplication
        final_dataset = deduplicate_dataset(valid_instructions)
        total_removed_duplicates = len(valid_instructions) - len(final_dataset)
        logger.info(f"Removed {total_removed_duplicates} duplicate instructions.")
        
        final_size = len(final_dataset)
        logger.info(f"Final dataset size: {final_size}")
        
        if final_size == 0:
            logger.error("Dataset is empty after processing. Exiting.")
            return
            
        # 9 & 10. Shuffle and Split
        train, val, test = shuffle_and_split(final_dataset, seed=42)
        
        # Export
        export_jsonl(train, self.output_dir / "instruction_train.jsonl")
        export_jsonl(val, self.output_dir / "instruction_validation.jsonl")
        export_jsonl(test, self.output_dir / "instruction_test.jsonl")
        export_jsonl(final_dataset, self.output_dir / "instruction_dataset.jsonl")
        
        # 12. Generate Report
        generate_report(final_dataset, self.output_dir / "report.md")
        
        # 11. Statistics Report to Console
        categories = Counter(r.get('category') for r in final_dataset)
        difficulties = Counter(r.get('difficulty') for r in final_dataset)
        avg_inst_len = sum(len(r.get('instruction', '').split()) for r in final_dataset) / final_size
        avg_ans_len = sum(len(r.get('output', '').split()) for r in final_dataset) / final_size
        
        print("\n" + "="*50)
        print(" Dataset Generation Statistics Report ")
        print("="*50)
        print(f"Total chunks loaded     : {total_loaded}")
        print(f"Chunks filtered         : {total_filtered}")
        print(f"Chunks retained         : {total_retained}")
        print(f"Instructions generated  : {total_generated}")
        print(f"Instructions removed    : {total_removed_validation}")
        print(f"Duplicates removed      : {total_removed_duplicates}")
        print(f"Final dataset size      : {final_size}")
        print("\nCategory Distribution:")
        for cat, count in categories.most_common():
            print(f"  - {cat:<25}: {count}")
        print("\nDifficulty Distribution:")
        for diff, count in difficulties.most_common():
            print(f"  - {diff:<25}: {count}")
        print(f"\nAverage instruction len : {avg_inst_len:.2f} words")
        print(f"Average answer len      : {avg_ans_len:.2f} words")
        print("="*50 + "\n")
