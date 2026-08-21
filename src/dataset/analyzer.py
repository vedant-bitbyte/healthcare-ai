import pandas as pd
import numpy as np
import logging
from typing import Dict, Any, List

logger = logging.getLogger(__name__)

class DatasetAnalyzer:
    """Analyzes the instruction tuning datasets to generate statistics and distributions."""
    
    def __init__(self):
        self.df = pd.DataFrame()
        self.stats = {}
        self.split_counts = {"train": 0, "validation": 0, "test": 0}

    def load_records(self, records: List[Dict[str, Any]], split_name: str):
        """Loads records into the analyzer, tracking the dataset split."""
        if not records:
            logger.warning(f"No records to load for split: {split_name}")
            return
            
        self.split_counts[split_name] = len(records)
        
        # Add a split column to track the source
        df_split = pd.DataFrame(records)
        df_split['split'] = split_name
        
        if self.df.empty:
            self.df = df_split
        else:
            self.df = pd.concat([self.df, df_split], ignore_index=True)
            
        logger.info(f"Loaded {len(records)} records for split: {split_name}")

    def analyze(self) -> Dict[str, Any]:
        """Computes all required dataset statistics."""
        if self.df.empty:
            logger.error("Cannot analyze: DataFrame is empty.")
            return {}

        logger.info("Computing dataset statistics...")
        
        # Compute word counts
        self.df['instruction_length'] = self.df['instruction'].apply(lambda x: len(str(x).split()))
        self.df['output_length'] = self.df['output'].apply(lambda x: len(str(x).split()))
        
        # Calculate vocabulary size (naïve space-split approach for both instruction and output)
        vocab = set()
        for text in self.df['instruction'].astype(str):
            vocab.update(text.lower().split())
        for text in self.df['output'].astype(str):
            vocab.update(text.lower().split())
            
        # Basic Stats
        self.stats = {
            "Total train samples": self.split_counts.get("train", 0),
            "Total validation samples": self.split_counts.get("validation", 0),
            "Total test samples": self.split_counts.get("test", 0),
            "Average instruction length": round(self.df['instruction_length'].mean(), 2),
            "Average output length": round(self.df['output_length'].mean(), 2),
            "Median output length": round(self.df['output_length'].median(), 2),
            "Maximum output length": int(self.df['output_length'].max()),
            "Minimum output length": int(self.df['output_length'].min()),
            "Vocabulary size": len(vocab),
            "Unique instructions": self.df['instruction'].nunique(),
            "Unique outputs": self.df['output'].nunique(),
            "Unique source documents": self.df['source'].nunique(),
            "Unique categories": self.df['category'].nunique(),
            "Unique difficulty labels": self.df['difficulty'].nunique(),
            "Average quality score": round(self.df['quality_score'].astype(float).mean(), 2)
        }
        
        logger.info("Analysis complete.")
        return self.stats
        
    def get_category_counts(self) -> pd.Series:
        return self.df['category'].value_counts()
        
    def get_difficulty_counts(self) -> pd.Series:
        return self.df['difficulty'].value_counts()
        
    def get_source_counts(self) -> pd.Series:
        return self.df['source'].value_counts()
        
    def get_quality_counts(self) -> pd.Series:
        return self.df['quality_score'].value_counts().sort_index()

    def get_dataframe(self) -> pd.DataFrame:
        """Returns the internal pandas DataFrame for visualization purposes."""
        return self.df
