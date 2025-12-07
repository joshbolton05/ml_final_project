import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def plot_eda(df: pd.DataFrame) -> None:
    print(df['Churn'].value_counts())

    number_cols = ['Age', 'Tenure', 'Usage Frequency', 'Support Calls', 'Payment Delay', 'Total Spend']
    category_cols = ['Gender', 'Subscription Type', 'Contract Length', 'Customer Status', 'Last Due Date', 'Last Payment Date']

    for col in number_cols:
        sns.histplot(data=df, x=col, hue='Churn')
        plt.title(f'Count of {col} by Churn')
        plt.show()
    for col in category_cols:
        sns.countplot(data=df, x=col, hue='Churn')
        plt.title(f'Count of {col} by Churn')
        plt.show()


if __name__ == "__main__":
    from src.data.load_training_data import load_dataset as load_training_dataset
    from src.data.preprocess_training import clean_training_dataset

    raw = load_training_dataset("/Users/joshbolton/ml_final_project/data/raw/train.csv")
    print('i am human')
    clean_train = clean_training_dataset(raw)
    plot_eda(clean_train)
    print('hello')
