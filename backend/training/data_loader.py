import pandas as pd
import os
import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from config import FILE_CUSTOMERS, FILE_LOANS, FILE_BUREAU

class DataLoader:
    """
    Handles loading and merging of Credit Risk datasets.
    """
    def __init__(self, data_dir: str = "dataset"):
        self.data_dir = data_dir

    def load_raw_data(self):
        """Loads raw DataFrames from CSVs."""
        try:
            customers = pd.read_csv(os.path.join(self.data_dir, FILE_CUSTOMERS))
            loans = pd.read_csv(os.path.join(self.data_dir, FILE_LOANS))
            bureau = pd.read_csv(os.path.join(self.data_dir, FILE_BUREAU))
            return customers, loans, bureau
        except FileNotFoundError as e:
            raise FileNotFoundError(f"Could not find dataset files in {self.data_dir}. Error: {e}")

    def merge_data(self, customers, loans, bureau):
        """Merges the three dataframes into a single view."""
        # Merge customers and loans
        # Note: Depending on data quality, validate 'cust_id' uniqueness
        df = pd.merge(customers, loans, on="cust_id", how="inner")
        
        # Merge bureau data
        df = pd.merge(df, bureau, on="cust_id", how="inner")
        
        return df

    def get_data(self):
        """Orchestrates loading and merging."""
        customers, loans, bureau = self.load_raw_data()
        return self.merge_data(customers, loans, bureau)

if __name__ == "__main__":
    # Test execution
    try:
        loader = DataLoader(data_dir=os.path.join(os.path.dirname(__file__), "..", "dataset"))
        df = loader.get_data()
        print(f"Data loaded successfully. Shape: {df.shape}")
        print(df.head())
    except Exception as e:
        print(f"Error loading data: {e}")
