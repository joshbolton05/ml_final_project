print('hi')
import os
import pandas as pd


def clean_test_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """Clean the training dataset by dropping rows with missing values."""
    df_cleaned = df.copy()

    #fill null values
    df_cleaned['Tenure'] = df_cleaned['Tenure'].fillna(df_cleaned['Tenure'].median())
    df_cleaned['Payment Delay'] = df_cleaned['Payment Delay'].fillna(df_cleaned['Payment Delay'].median())
    df_cleaned['Last Interaction'] = df_cleaned['Last Interaction'].fillna(df_cleaned['Last Interaction'].median())

    df_cleaned['Support Calls'] = df_cleaned['Support Calls'].replace({'None' : 0,'none' : 0})
    df_cleaned['Support Calls'] = pd.to_numeric(df_cleaned['Support Calls']).fillna(0).astype(int)


    #Give replacing non-numeric data with numeric replacements

    df_cleaned['Gender'] = df_cleaned['Gender'].replace({'Male' : 0, 'Female' : 1})
    df_cleaned['Customer Status'] = df_cleaned['Customer Status'].replace({'inactive' : 0, 'active' : 1})
    df_cleaned['Subscription Type'] = df_cleaned['Subscription Type'].replace({'Basic' : 0, 'Standard' : 1, 'Premium' : 2})
    df_cleaned['Contract Length'] = df_cleaned['Contract Length'].replace({'Annual' : 1, 'Quarterly' : 4, 'Monthly' : 12})

    #changing date of month-day to only month

    df_cleaned['Last Due Date'] = df_cleaned['Last Due Date'].astype(str).str[0:2].astype(int)
    df_cleaned['Last Payment Date'] = df_cleaned['Last Payment Date'].astype(str).str[0:2].astype(int)
    return df_cleaned


if __name__ == "__main__":
    # Load the raw dataset
    raw = pd.read_csv("/Users/joshbolton/ml_final_project/data/raw/test.csv")
    # Clean the dataset
    cleaned = clean_test_dataset(raw)
    # Ensure the processed directory exists
    os.makedirs("data/processed", exist_ok=True)
    # Save the cleaned data
    processed_path = "/Users/joshbolton/ml_final_project/data/processed/cleaned_test.csv"
    cleaned.to_csv(processed_path, index=False)
    print(f"Cleaned data saved to {processed_path}")
