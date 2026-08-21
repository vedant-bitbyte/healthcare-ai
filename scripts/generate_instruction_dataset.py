import sys
from pathlib import Path

# Add src to python path so we can import dataset_generation
sys.path.append(str(Path(__file__).parent.parent / "src"))

from src.dataset.pipeline import DatasetGenerationPipeline

def main():
    base_dir = Path(__file__).parent.parent
    data_dir = base_dir / "data" / "processed"
    output_dir = base_dir / "output"
    
    pipeline = DatasetGenerationPipeline(
        data_dir=str(data_dir),
        output_dir=str(output_dir)
    )
    
    pipeline.run()

if __name__ == "__main__":
    main()
