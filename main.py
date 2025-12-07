import sys
print(sys.path)
from src.data.load_training_data import load_dataset as load_training_dataset
from src.data.load_test_data import load_dataset as load_test_dataset
from src.data.preprocess_training import clean_training_dataset
from src.data.preprocess_test import clean_test_dataset
from src.visualization.ml_eda import plot_eda
from src.data.split_data import split_dataset, plot_roc_curve
from src.models.my_knn_model import train_knn_model
from src.models.my_classifier_model import train_my_classifier_model
from src.models.my_decision_tree_model import train_decision_tree_model
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from src.visualization.performance import (
    plot_confusion_matrices,
    plot_performance_comparison,
)

def main() -> None:
    print("---Loading training data...")
    raw_train_df = load_training_dataset("/Users/joshbolton/ml_final_project/data/raw/train.csv")
    
    print("---Cleaning training data...")
    clean_train_df = clean_training_dataset(raw_train_df)

    print("---Loading test data...")
    raw_test_df = load_test_dataset("/Users/joshbolton/ml_final_project/data/raw/test.csv")

    print("---Cleaning test data...")
    clean_test_df = clean_test_dataset(raw_test_df)

    print('hi')

    print(f"Cleaned training dataset shape: {clean_train_df.shape}")

    print("---Creating EDA visuals...")
    print(clean_train_df.head())
    plot_eda(clean_train_df)

    print("---Splitting data...")
    X_train, X_val, y_train, y_val = split_dataset(clean_train_df)

    print("---Training models...")
    knn_model = train_knn_model(X_train, y_train)
    my_decision_tree = train_decision_tree_model(X_train, y_train)

    print("---Evaluating on validation set...")
    y_pred_my_classifier, yproba_my_classifier = train_my_classifier_model(X_val)
    y_pred_knn = knn_model.predict(X_val)
    y_pred_decision_tree = my_decision_tree.predict(X_val)

    y_proba_knn = knn_model.predict_proba(X_val)[:, 1]
    y_proba_decision_tree = my_decision_tree.predict_proba(X_val)[:, 1]

    plot_confusion_matrices(y_val, y_pred_my_classifier, y_pred_knn, y_pred_decision_tree)
    plot_performance_comparison(y_val, y_pred_my_classifier, y_pred_knn, y_pred_decision_tree)

    auc_my_classifier = plot_roc_curve(y_val, yproba_my_classifier, "My Classifier")
    auc_knn = plot_roc_curve(y_val, y_proba_knn, "KNN")
    auc_decision_tree = plot_roc_curve(y_val,y_proba_decision_tree, "Decision Tree")

    print("My Classifer is Best Model")
    X_test = clean_test_df
    y_test_pred_my_classifier, yproba_test_my_classifier = train_my_classifier_model(X_test)

    print("Done.")




if __name__ == "__main__":
    main()