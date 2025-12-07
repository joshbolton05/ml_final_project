import pandas as pd
from sklearn.tree import DecisionTreeClassifier


def train_decision_tree_model(X_train: pd.DataFrame, y_train: pd.Series) -> DecisionTreeClassifier:
    """Train and return a Decision Tree classifier."""
    my_decision_tree = DecisionTreeClassifier(criterion='entropy', max_depth=10, min_samples_split=20, min_samples_leaf=8, random_state=123)
    my_decision_tree.fit(X_train, y_train)
    return my_decision_tree
