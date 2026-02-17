"""
Utility functions for Credit Risk project.
Handles file I/O, object persistence, and logging helpers.
"""

import joblib
import pandas as pd
from pathlib import Path
import os
from typing import Any

def save_joblib(obj: Any, filepath: Path) -> None:
    """
    Save an object to a joblib file.
    
    Args:
        obj: Object to save
        filepath: Path to save the file
    """
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(obj, filepath)
    print(f"Artifact saved to {filepath}")

def load_joblib(filepath: Path) -> Any:
    """
    Load an object from a joblib file.
    
    Args:
        filepath: Path to load the file from
        
    Returns:
        Loaded object
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")
    return joblib.load(filepath)

def save_dataframe(df: pd.DataFrame, filepath: Path) -> None:
    """
    Save DataFrame to CSV.
    """
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(filepath, index=False)
    print(f"Data saved to {filepath}")

def load_dataframe(filepath: Path) -> pd.DataFrame:
    """
    Load DataFrame from CSV.
    """
    return pd.read_csv(filepath)
