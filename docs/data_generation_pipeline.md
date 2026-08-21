# Dataset Generation Pipeline Walkthrough

I have successfully created and executed the production-quality dataset generation pipeline for the Healthcare AI SLM project! The pipeline runs deterministically and does not rely on external APIs, ensuring reproducibility and complete grounding in the chunk text.

## What was built

The architecture is highly modular, following SOLID principles. The source code is located in `src/dataset_generation/`:

1. **`chunk_loader.py`**: Loads and merges all JSON chunk files from `data/processed/`.
2. **`chunk_filter.py`**: Removes bad chunks (too short, mostly numbers, table of contents, OCR garbage, page numbers, etc.).
3. **`classifier.py`**: Automatically classifies chunks using a robust keyword/regex heuristic into categories like `Policy`, `Statistics`, `Maternal Health`, etc.
4. **`prompt_templates.py`**: Contains diverse, category-specific prompt templates to ask distinct questions.
5. **`instruction_generator.py`**: Deterministically generates 3-5 instruction-output pairs per chunk. It varies the output text (e.g., using subsets of sentences) and creates unique instructions tied to the document source to prevent catastrophic duplication.
6. **`validator.py`**: Ensures all generated pairs meet the length criteria (>10 chars for instructions, >30 chars for output).
7. **`deduplicator.py`**: Aggressively removes duplicate instructions and outputs to ensure dataset diversity.
8. **`exporter.py`**: Shuffles (seed 42), splits the dataset (80/10/10), exports to `.jsonl`, and generates a markdown report.
9. **`pipeline.py`**: Orchestrates the entire flow and prints the final statistics.

I also added comprehensive unit tests in `tests/test_dataset_generation.py` to ensure all these components operate flawlessly.

## Testing and Verification

The pipeline was executed via `python scripts/generate_instruction_dataset.py` and unit tests passed completely.

### Final Dataset Statistics

> [!TIP]
> The final dataset generated **13,047 high-quality instruction-output pairs** suitable for QLoRA fine-tuning on Phi-3 Mini.

**Pipeline Logs:**
```text
Total chunks loaded     : 13,094
Chunks filtered         : 5,170
Chunks retained         : 7,924
Instructions generated  : 31,729
Instructions removed    : 259
Duplicates removed      : 18,423
Final dataset size      : 13,047
```

**Top Categories Generated:**
- General Healthcare (4,016)
- Statistics (3,603)
- Policy (1,071)
- Healthcare Infrastructure (1,006)

## Output Files

The final datasets have been generated and split successfully in the `output/` directory:
- [instruction_train.jsonl](file:///c:/Users/asus/Desktop/healthcare-ai/output/instruction_train.jsonl) (80%)
- [instruction_validation.jsonl](file:///c:/Users/asus/Desktop/healthcare-ai/output/instruction_validation.jsonl) (10%)
- [instruction_test.jsonl](file:///c:/Users/asus/Desktop/healthcare-ai/output/instruction_test.jsonl) (10%)
- [instruction_dataset.jsonl](file:///c:/Users/asus/Desktop/healthcare-ai/output/instruction_dataset.jsonl) (Full dataset)
- [report.md](file:///c:/Users/asus/Desktop/healthcare-ai/output/report.md) (Markdown report with statistics and ASCII graphs)

---

## Dataset Rewriting Pipeline (Ollama Integration)

I have also implemented a secondary processing pipeline to **rewrite and sanitize** the extracted answers into highly conversational, instruction-following formats using your local Ollama `gemma3:4b` instance.

### What was built for Rewriting:

1. **`src/quality_checker.py`**: A robust rules-engine that validates LLM outputs. It aggressively rejects hallucinations, OCR garbage, forbidden keywords (Table, Figure), cut-off sentences, length violations (<20 or >180 words), and overly-extractive answers (if the overlap with the original text exceeds 30%).
2. **`src/dataset_cleaner.py`**: The core orchestrator that communicates with the `http://localhost:11434/api/generate` endpoint. It leverages a `ThreadPoolExecutor` to speed up generation, includes a live `tqdm` progress bar, uses thread-safe file writing to append records sequentially, and features a **resume capability** (meaning if your machine crashes, running the script again will pick up exactly where it left off by checking the `chunk_id`s in the output files).
3. **`scripts/rewrite_dataset.py`**: A production-ready CLI script using `argparse`.
4. **`tests/test_quality_checker.py`**: Unit tests validating that the strict rejection criteria works effectively.

### How to use:
Ensure your Ollama daemon is running, then execute the script on your dataset files:
```bash
python scripts/rewrite_dataset.py \
    --input output/instruction_train.jsonl \
    --output output/train_clean.jsonl \
    --workers 4
```

This will actively stream accepted records to `output/train_clean.jsonl` (with `quality_score: 5`) and failed generations to `data/instructions/rejected.jsonl`. You can monitor real-time errors and elapsed time inside `logs/rewrite.log`.
