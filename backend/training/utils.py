# File I/O helpers — nothing clever, just keeping the rest of the code
# from having to care about joblib vs pandas vs path creation.

import joblib
import pandas as pd
from pathlib import Path
import os
from typing import Any


def save_joblib(obj: Any, filepath: Path) -> None:
    """Dump any object to a joblib file. Creates parent dirs if needed."""
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(obj, filepath)
    print(f"Artifact saved to {filepath}")


def load_joblib(filepath: Path) -> Any:
    """Load a joblib file. Gives a clear error if the file doesn't exist yet."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")
    return joblib.load(filepath)


def save_dataframe(df: pd.DataFrame, filepath: Path) -> None:
    """Save a DataFrame to CSV without row index."""
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(filepath, index=False)
    print(f"Data saved to {filepath}")


def load_dataframe(filepath: Path) -> pd.DataFrame:
    """Read a CSV into a DataFrame."""
    return pd.read_csv(filepath)
