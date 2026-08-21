import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

class DatasetVisualizer:
    """Generates publication-quality plots for the dataset analysis."""
    
    def __init__(self, output_dir: str = "plots"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Set clean research-ready theme
        sns.set_theme(style="whitegrid", palette="muted")
        
    def generate_all_plots(self, df: pd.DataFrame, split_counts: dict):
        """Generates all required plots."""
        if df.empty:
            logger.error("Empty DataFrame provided to visualizer. Skipping plots.")
            return

        logger.info(f"Generating plots in {self.output_dir}...")
        
        self.plot_category_distribution(df)
        self.plot_difficulty_distribution(df)
        self.plot_source_distribution(df)
        self.plot_quality_distribution(df)
        self.plot_instruction_length_histogram(df)
        self.plot_output_length_histogram(df)
        self.plot_quality_vs_output_length(df)
        self.plot_dataset_split(split_counts)
        
        logger.info("All plots generated successfully.")
        
    def plot_category_distribution(self, df: pd.DataFrame):
        plt.figure(figsize=(10, 6))
        order = df['category'].value_counts().index
        sns.countplot(data=df, x='category', order=order)
        plt.title('Category Distribution', fontsize=14, pad=15)
        plt.xlabel('Category', fontsize=12)
        plt.ylabel('Count', fontsize=12)
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(self.output_dir / 'category_distribution.png', dpi=300)
        plt.close()
        
    def plot_difficulty_distribution(self, df: pd.DataFrame):
        plt.figure(figsize=(8, 6))
        order = ['Easy', 'Medium', 'Hard'] # Try to enforce logical order if present
        existing_diffs = df['difficulty'].unique()
        plot_order = [d for d in order if d in existing_diffs] + [d for d in existing_diffs if d not in order]
        
        sns.countplot(data=df, x='difficulty', order=plot_order)
        plt.title('Difficulty Distribution', fontsize=14, pad=15)
        plt.xlabel('Difficulty Level', fontsize=12)
        plt.ylabel('Count', fontsize=12)
        plt.tight_layout()
        plt.savefig(self.output_dir / 'difficulty_distribution.png', dpi=300)
        plt.close()
        
    def plot_source_distribution(self, df: pd.DataFrame):
        plt.figure(figsize=(10, 8))
        order = df['source'].value_counts().index
        sns.countplot(data=df, y='source', order=order)
        plt.title('Source Document Distribution', fontsize=14, pad=15)
        plt.xlabel('Count', fontsize=12)
        plt.ylabel('Source', fontsize=12)
        plt.tight_layout()
        plt.savefig(self.output_dir / 'source_distribution.png', dpi=300)
        plt.close()
        
    def plot_quality_distribution(self, df: pd.DataFrame):
        plt.figure(figsize=(8, 6))
        # Ensure quality is treated as discrete
        quality_counts = df['quality_score'].astype(float).value_counts().sort_index()
        sns.barplot(x=quality_counts.index, y=quality_counts.values, color=sns.color_palette()[0])
        plt.title('Quality Score Distribution', fontsize=14, pad=15)
        plt.xlabel('Quality Score', fontsize=12)
        plt.ylabel('Count', fontsize=12)
        plt.tight_layout()
        plt.savefig(self.output_dir / 'quality_distribution.png', dpi=300)
        plt.close()
        
    def plot_instruction_length_histogram(self, df: pd.DataFrame):
        plt.figure(figsize=(10, 6))
        sns.histplot(df['instruction_length'], bins=30, kde=True)
        plt.title('Instruction Length Distribution', fontsize=14, pad=15)
        plt.xlabel('Length (Words)', fontsize=12)
        plt.ylabel('Frequency', fontsize=12)
        plt.tight_layout()
        plt.savefig(self.output_dir / 'instruction_length_histogram.png', dpi=300)
        plt.close()
        
    def plot_output_length_histogram(self, df: pd.DataFrame):
        plt.figure(figsize=(10, 6))
        sns.histplot(df['output_length'], bins=50, kde=True)
        plt.title('Output Length Distribution', fontsize=14, pad=15)
        plt.xlabel('Length (Words)', fontsize=12)
        plt.ylabel('Frequency', fontsize=12)
        plt.tight_layout()
        plt.savefig(self.output_dir / 'output_length_histogram.png', dpi=300)
        plt.close()
        
    def plot_quality_vs_output_length(self, df: pd.DataFrame):
        plt.figure(figsize=(10, 6))
        # Add some jitter to quality score for better visualization
        sns.stripplot(data=df, x='quality_score', y='output_length', alpha=0.3, jitter=True)
        plt.title('Quality Score vs Output Length', fontsize=14, pad=15)
        plt.xlabel('Quality Score', fontsize=12)
        plt.ylabel('Output Length (Words)', fontsize=12)
        plt.tight_layout()
        plt.savefig(self.output_dir / 'quality_vs_output_length.png', dpi=300)
        plt.close()
        
    def plot_dataset_split(self, split_counts: dict):
        plt.figure(figsize=(8, 8))
        labels = []
        sizes = []
        for split, count in split_counts.items():
            if count > 0:
                labels.append(split.capitalize())
                sizes.append(count)
                
        colors = sns.color_palette("pastel")[0:len(labels)]
        plt.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', 
                startangle=90, wedgeprops={'edgecolor': 'white'})
        plt.title('Dataset Split (Train / Validation / Test)', fontsize=14, pad=15)
        plt.tight_layout()
        plt.savefig(self.output_dir / 'dataset_split.png', dpi=300)
        plt.close()
