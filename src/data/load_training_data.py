import pandas as pd


def load_dataset(file_path: str) -> pd.DataFrame:
    """Load the training dataset."""
    return pd.read_csv(file_path)


if __name__ == "__main__":
    df = load_dataset("/Users/joshbolton/ml_final_project/data/raw/train.csv")
    print(df.head())
