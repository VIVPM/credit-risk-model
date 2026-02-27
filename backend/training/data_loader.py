# Loads the three raw CSVs (customers, loans, bureau) and merges them
# on cust_id. Everything downstream expects this single merged DataFrame.

import pandas as pd
import os
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

from config import FILE_CUSTOMERS, FILE_LOANS, FILE_BUREAU


class DataLoader:
    def __init__(self, data_dir: str = "dataset"):
        self.data_dir = data_dir

    def load_raw_data(self):
        """Read the three source CSVs. Fails loudly if any file is missing."""
        try:
            customers = pd.read_csv(os.path.join(self.data_dir, FILE_CUSTOMERS))
            loans     = pd.read_csv(os.path.join(self.data_dir, FILE_LOANS))
            bureau    = pd.read_csv(os.path.join(self.data_dir, FILE_BUREAU))
            return customers, loans, bureau
        except FileNotFoundError as e:
            raise FileNotFoundError(
                f"Could not find dataset files in {self.data_dir}. Error: {e}"
            )

    def merge_data(self, customers, loans, bureau):
        """
        Inner join customers → loans → bureau on cust_id.
        Inner join intentionally drops customers who have no loan or bureau record.
        """
        df = pd.merge(customers, loans,   on="cust_id", how="inner")
        df = pd.merge(df,        bureau,  on="cust_id", how="inner")
        return df

    def get_data(self):
        """One-call interface for the rest of the pipeline."""
        customers, loans, bureau = self.load_raw_data()
        return self.merge_data(customers, loans, bureau)


if __name__ == "__main__":
    try:
        loader = DataLoader(data_dir=os.path.join(os.path.dirname(__file__), "..", "dataset"))
        df = loader.get_data()
        print(f"Data loaded successfully. Shape: {df.shape}")
        print(df.head())
    except Exception as e:
        print(f"Error loading data: {e}")
