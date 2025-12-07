import sys
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score


def plot_confusion_matrices(y_val, y_pred_my_classifier, y_pred_knn, y_pred_decision_tree) -> None:
    """Plot confusion matrices for both models."""
    conf_my_classifier = confusion_matrix(y_val, y_pred_my_classifier)
    conf_knn = confusion_matrix(y_val, y_pred_knn)
    conf_decision_tree = confusion_matrix(y_val, y_pred_decision_tree)

    fig, axes = plt.subplots(1, 3, figsize=(10, 4))
    sns.heatmap(conf_my_classifier, annot=True, fmt='d', cmap='Reds', ax=axes[0])
    axes[0].set_title('My Classifier')
    sns.heatmap(conf_knn, annot=True, fmt='d', cmap='Blues', ax=axes[1])
    axes[1].set_title('KNN')
    sns.heatmap(conf_decision_tree, annot=True, fmt='d', cmap='Greens', ax=axes[2])
    axes[2].set_title('Decision Tree')
    plt.tight_layout()
    plt.show()


def plot_performance_comparison(y_val, y_pred_my_classifier, y_pred_knn, y_pred_decision_tree) -> None:
    """Create a bar chart comparing model metrics."""
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1 Score']
    my_classifier_scores = [
        accuracy_score(y_val, y_pred_my_classifier),
        precision_score(y_val, y_pred_my_classifier, zero_division=0),
        recall_score(y_val, y_pred_my_classifier),
        f1_score(y_val, y_pred_my_classifier)
    ]
    knn_scores = [
        accuracy_score(y_val, y_pred_knn),
        precision_score(y_val, y_pred_knn),
        recall_score(y_val, y_pred_knn),
        f1_score(y_val, y_pred_knn)
    ]

    decision_tree_scores = [
        accuracy_score(y_val, y_pred_decision_tree),
        precision_score(y_val, y_pred_decision_tree),
        recall_score(y_val, y_pred_decision_tree),
        f1_score(y_val, y_pred_decision_tree)
    ]
    df = pd.DataFrame({'Metric': metrics, "My Classifier": my_classifier_scores, 'KNN': knn_scores, 'Decision Tree': decision_tree_scores})
    df.plot(x='Metric', kind='bar', figsize=(8, 5))
    plt.ylim(0, 1)
    plt.title('Model Performance Comparison')
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    sys.path.append('/Users/joshbolton/ml_final_project/')
    from src.data.load_training_data import load_dataset
    from src.data.preprocess_training import clean_training_dataset
    from src.data.split_data import split_dataset
    from src.models.my_knn_model import train_knn_model
    from src.models.my_classifier_model import train_my_classifier_model
    from src.models.my_decision_tree_model import train_decision_tree_model

    raw_df = load_dataset("/Users/joshbolton/ml_final_project/data/raw/train.csv")
    clean_df = clean_training_dataset(raw_df)
    X_train, X_val, y_train, y_val = split_dataset(clean_df)
    my_knn_model = train_knn_model(X_train, y_train)
    y_pred_my_classifier = train_my_classifier_model(X_train, y_train)
    my_decision_tree = train_decision_tree_model(X_train, y_train)
    y_pred_knn = my_knn_model.predict(X_val)
    y_pred_decision_tree = my_decision_tree.predict(X_val)
    plot_confusion_matrices(y_val, y_pred_my_classifier, y_pred_knn, y_pred_decision_tree)
    plot_performance_comparison(y_val, y_pred_my_classifier, y_pred_knn, y_pred_decision_tree)
