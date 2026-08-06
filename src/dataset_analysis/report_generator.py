import logging
from pathlib import Path
from typing import Dict, Any
import pandas as pd

logger = logging.getLogger(__name__)

class ReportGenerator:
    """Generates a comprehensive Markdown report based on dataset analysis."""
    
    def __init__(self, output_path: str = "reports/dataset_report.md"):
        self.output_path = Path(output_path)
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        
    def generate(self, validation_summary: Dict[str, int], stats: Dict[str, Any], df: pd.DataFrame):
        logger.info(f"Generating research report at {self.output_path}...")
        
        report_lines = []
        report_lines.append("# Dataset Overview\n")
        report_lines.append("This report provides a comprehensive analysis of the generated instruction tuning dataset for fine-tuning Phi-3 Mini.\n")
        
        report_lines.append("## Validation Summary\n")
        for key, value in validation_summary.items():
            report_lines.append(f"- **{key}**: {value}")
        report_lines.append("\n")
        
        report_lines.append("## Dataset Statistics\n")
        for key, value in stats.items():
            report_lines.append(f"- **{key}**: {value}")
        report_lines.append("\n")
        
        report_lines.append("## Category Distribution\n")
        report_lines.append("Please refer to `plots/category_distribution.png` for a visual distribution.\n")
        
        report_lines.append("## Difficulty Distribution\n")
        report_lines.append("Please refer to `plots/difficulty_distribution.png` for a visual distribution.\n")
        
        report_lines.append("## Source Distribution\n")
        report_lines.append("Please refer to `plots/source_distribution.png` for a visual distribution.\n")
        
        report_lines.append("## Output Length Analysis\n")
        report_lines.append("Please refer to `plots/output_length_histogram.png` and `plots/instruction_length_histogram.png` for length distributions.\n")
        
        report_lines.append("## Quality Analysis\n")
        report_lines.append("Please refer to `plots/quality_distribution.png` and `plots/quality_vs_output_length.png`.\n")
        
        report_lines.append("## Observations\n")
        observations = self._generate_observations(stats, df)
        for obs in observations:
            report_lines.append(f"- {obs}")
            
        with open(self.output_path, 'w', encoding='utf-8') as f:
            f.write("\n".join(report_lines))
            
        logger.info("Report generated successfully.")
        
    def _generate_observations(self, stats: Dict[str, Any], df: pd.DataFrame) -> list:
        observations = []
        
        # 1. Class Imbalance (Categories)
        if not df.empty:
            cat_counts = df['category'].value_counts()
            if not cat_counts.empty:
                top_cat = cat_counts.index[0]
                top_cat_pct = cat_counts.iloc[0] / len(df) * 100
                if top_cat_pct > 50:
                    observations.append(f"**Potential class imbalance**: The category '{top_cat}' dominates the dataset ({top_cat_pct:.1f}%). Consider balancing categories before training.")
                else:
                    observations.append(f"Category distribution appears reasonably balanced. The top category is '{top_cat}' ({top_cat_pct:.1f}%).")
                    
        # 2. Potential Bias (Source representation)
        if not df.empty:
            source_counts = df['source'].value_counts()
            if not source_counts.empty:
                top_source = source_counts.index[0]
                top_source_pct = source_counts.iloc[0] / len(df) * 100
                if top_source_pct > 60:
                    observations.append(f"**Potential bias**: {top_source_pct:.1f}% of the dataset comes from a single source ('{top_source}'). Model might overfit to this document's style/domain.")
                    
        # 3. Answer Lengths
        min_len = stats.get("Minimum output length", 0)
        max_len = stats.get("Maximum output length", 0)
        avg_len = stats.get("Average output length", 0)
        
        if min_len < 5:
            observations.append("**Very short answers detected**: Some outputs are extremely short. Ensure these are valid and not truncated.")
        if max_len > 500:
            observations.append(f"**Very long answers detected**: The maximum output length is {max_len} words, which might exceed model context window constraints if not handled.")
            
        # 4. Quality Observations
        avg_quality = stats.get("Average quality score", 0)
        if avg_quality >= 4.5:
            observations.append(f"**Quality**: The dataset has a high average quality score ({avg_quality}/5), indicating strong rewriting performance.")
        elif avg_quality < 3.0:
            observations.append(f"**Quality Warning**: The average quality score is low ({avg_quality}/5). Substantial filtering might be needed.")
            
        # 5. Recommendations
        observations.append("**Recommendations before training**: Verify the dataset split ratios (train/val/test) are optimal for your task and ensure that the 'quality_score' threshold for training inclusion is set appropriately.")
        
        return observations
