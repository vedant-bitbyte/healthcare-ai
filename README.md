# Healthcare AI SLM

AI-powered Healthcare Budget Recommendation & Insights System for Indian Healthcare Administration.

A research project focused on developing a domain-specific Small Language Model (SLM) for healthcare using fine-tuning (Phi-3 Mini / Gemma) and Retrieval-Augmented Generation (RAG).

---

## System Architecture

<p align="center">
  <img src="assets/diagram/Main_Architecture.svg" alt="Main Architecture" width="1121">
</p>

---

## Fine-Tuning Architecture

<p align="center">
  <img src="assets/diagram/Fine-Tuning_Architecture.svg" alt="Fine-Tuning Architecture" width="191">
</p>

---

## Project Modules & Sub-Projects

### 1. Document Ingestion & Vector Storage (`src/ingestion`, `src/embeddings`)
- **PDF & CSV Loaders**: Automated parsing and cleaning of Indian healthcare budget documents and unstructured text.
- **Text Chunking**: Configurable recursive chunking for semantic preservation.
- **Vector Embeddings**: ChromaDB vector store integration with huggingface embeddings for fast similarity retrieval.

### 2. Retrieval-Augmented Generation (RAG) (`src/rag`, `src/retrieval`)
- **Query Router**: Classifies incoming query intent (budget, policy, general) to route to specialized retrieval pipelines.
- **Hybrid Retriever**: Retrieves context from vector store with relevance scoring and fallback handling.
- **RAG Pipeline**: Dynamic prompt construction and LLM inference integration.

### 3. Synthetic Dataset Generation & Cleaning (`src/dataset_generation`, `src/dataset_converter`)
- **Instruction Generator**: Automated generation of domain-specific Q&A instruction pairs from raw chunks using LLM prompts.
- **Deduplication & Quality Checker**: Filtering low-quality pairs, removing exact/near duplicate samples.
- **Format Converter**: Utility scripts converting instruction datasets into Chat, Alpaca, and fine-tuning standard formats.

### 4. Dataset Analysis & Reporting (`src/dataset_analysis`)
- **Dataset Analyzer**: Computes token counts, message length distributions, and vocabulary statistics.
- **Visualizer**: Generates plots for dataset length distribution and key metrics.
- **Report Generator**: Exports HTML/Markdown analytical reports for dataset health inspection.

### 5. SLM Fine-Tuning Engine (`training/`, `scripts/`)
- **LoRA / QLoRA Fine-Tuning**: Parameter-efficient fine-tuning scripts optimized for Phi-3 Mini and Gemma models.
- **Trainer Setup**: Custom dataset formatting, tokenization, dynamic batching, and checkpoint management.

### 6. Model Evaluation & Benchmarking (`evaluation/`, `evaluate_model.py`)
- **Automated Batch Evaluator**: Runs RAG evaluation pipelines against predefined test question sets (`evaluation_questions.csv`).
- **Model Comparison**: Benchmarks performance metrics across fine-tuned model variants (Phi-3 vs Gemma 3).
- **Manual Evaluation Suite**: Generates scoring templates for human feedback and quality review.

---

## Core Features

- 📄 **Healthcare Document Ingestion**: Ingest PDF/CSV budget and policy documents.
- 🎯 **Domain-Specific RAG**: Retrieval-augmented system tailored for Indian healthcare context.
- 🤖 **Fine-Tuned Healthcare SLM**: Lightweight models (Phi-3 Mini) fine-tuned on instruction datasets.
- 📊 **Dataset Analytics**: Built-in analysis, validation, and visual reporting for datasets.
- 🧪 **Comprehensive Evaluation**: Automated metric tracking and comparative model testing.

---

## Tech Stack

- **Languages & Frameworks**: Python, PyTorch, Hugging Face Transformers
- **RAG & Vectors**: LangChain, ChromaDB
- **Fine-Tuning**: PEFT (LoRA / QLoRA), TRL (SFTTrainer), BitsAndBytes
- **Models**: Phi-3 Mini, Gemma 3
- **Analysis & Visualization**: Pandas, Matplotlib, Seaborn

---

## Project Structure

```text
healthcare-ai/
├── assets/             # Architecture diagrams and static visual assets
├── configs/            # Configuration files for training and RAG
├── data/               # Raw and processed healthcare datasets
├── evaluation/         # Evaluation scripts, question sets, and result metrics
├── reports/            # Generated analysis and dataset reports
├── scripts/            # Helper CLI scripts (dataset analysis, chat format conversion)
├── src/                # Core source modules
│   ├── dataset_analysis/   # Dataset stats, validation, and visual reports
│   ├── dataset_converter/  # Dataset format converters (Alpaca/Chat)
│   ├── dataset_generation/ # Instruction generation and cleaning pipeline
│   ├── embeddings/         # Vector store and embedding generation
│   ├── ingestion/          # Document loading (PDF/CSV) and text chunking
│   ├── rag/                # RAG pipeline and prompt builders
│   └── retrieval/          # Query router and document retriever
├── training/           # Fine-tuning codebase (Phi-3 / Gemma, LoRA configs)
├── evaluate_model.py   # Main evaluation runner script
└── requirements.txt    # Python dependencies
```

---

## Getting Started

### 1. Installation

```bash
git clone https://github.com/vedant-bitbyte/healthcare-ai.git
cd healthcare-ai
pip install -r requirements.txt
```

### 2. Running Model Evaluation

```bash
python evaluate_model.py --model phi3:mini --questions evaluation/evaluation_questions.csv
```

### 3. Dataset Generation & Analysis

```bash
# Run dataset analysis
python scripts/analyze_dataset.py

# Convert dataset to chat format
python scripts/convert_to_chat.py
```

### 4. Fine-Tuning the Model

```bash
python scripts/train_phi3.py
```

---

## Project Goal

To build a lightweight, domain-specific Healthcare Small Language Model (SLM) capable of providing accurate healthcare insights, policy answers, and budget recommendations for Indian healthcare administration.

---

## License

This project is intended for academic and research purposes.