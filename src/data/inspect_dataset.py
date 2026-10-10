import pandas as pd
from pathlib import Path

data_path = Path("data/raw/banking77")

train_file = data_path / "train.csv"
test_file = data_path / "test.csv"

train_df = pd.read_csv(train_file)
test_df = pd.read_csv(test_file)

print("Train shape:", train_df.shape)
print("Test shape:", test_df.shape)

print("\nColumns:")
print(train_df.columns.tolist())

print("\nFirst 5 training records:")
print(train_df.head())

print("\nData types:")
print(train_df.dtypes)