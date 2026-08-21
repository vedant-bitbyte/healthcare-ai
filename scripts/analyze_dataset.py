import sys
import logging
from pathlib import Path
import argparse

# Add src to Python path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from src.dataset import DatasetValidator, DatasetAnalyzer, DatasetVisualizer, ReportGenerator

def setup_logging():
    log_dir = Path("logs")
    log_dir.mkdir(parents=True, exist_ok=True)
    
    log_file = log_dir / "dataset_analysis.log"
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file, encoding='utf-8'),
            # logging.StreamHandler(sys.stdout) # Will not log to console to keep output clean as requested
        ]
    )

def main():
    parser = argparse.ArgumentParser(description="Run complete validation and generate statistics/reports for dataset.")
    parser.add_argument("--train", default="output/train_clean.jsonl", help="Train dataset path")
    parser.add_argument("--val", default="output/validation_clean.jsonl", help="Validation dataset path")
    parser.add_argument("--test", default="output/test_clean.jsonl", help="Test dataset path")
    
    args = parser.parse_args()
    
    setup_logging()
    logger = logging.getLogger("analyze_dataset")
    logger.info("Starting dataset analysis...")
    
    # 1. Validation
    validator = DatasetValidator()
    
    train_records = validator.validate_file(Path(args.train))
    val_records = validator.validate_file(Path(args.val))
    test_records = validator.validate_file(Path(args.test))
    
    val_summary = validator.get_summary()
    
    # 2. Analysis
    analyzer = DatasetAnalyzer()
    analyzer.load_records(train_records, "train")
    analyzer.load_records(val_records, "validation")
    analyzer.load_records(test_records, "test")
    
    stats = analyzer.analyze()
    df = analyzer.get_dataframe()
    
    # 3. Visualization
    visualizer = DatasetVisualizer(output_dir="plots")
    visualizer.generate_all_plots(df, analyzer.split_counts)
    
    # 4. Report Generation
    report_gen = ReportGenerator(output_path="reports/dataset_report.md")
    report_gen.generate(val_summary, stats, df)
    
    # 5. Print Console Summary
    print("-" * 38)
    print("Dataset Analysis Complete\n")
    print(f"Train Samples              : {stats.get('Total train samples', 0)}")
    print(f"Validation Samples         : {stats.get('Total validation samples', 0)}")
    print(f"Test Samples               : {stats.get('Total test samples', 0)}")
    print(f"Duplicates                 : {val_summary.get('Duplicates', 0)}")
    print(f"Average Output Length      : {stats.get('Average output length', 0)} words")
    print(f"Average Instruction Length : {stats.get('Average instruction length', 0)} words")
    print(f"Average Quality Score      : {stats.get('Average quality score', 0)}")
    print("\nPlots Saved                : 8")
    print("Report Saved               : reports/dataset_report.md")
    print("-" * 38)
    
    logger.info("Dataset analysis script completed.")

if __name__ == "__main__":
    main()
