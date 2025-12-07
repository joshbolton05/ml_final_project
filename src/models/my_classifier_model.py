import pandas as pd


def train_my_classifier_model(X):
    def my_classifier(row):
        if (row['Payment Delay'] == 10 or row['Total Spend'] < 500 or row['Contract Length'] == 12):
            return 1
        else:
            return 0
        
    def my_classifier_proba(row):
        x = 0.1
        if (row['Payment Delay'] == 10):
            x+=0.75
        if (row['Total Spend'] < 500):
            x+=0.6
        if (row['Contract Length'] == 12):
            x+=0.6
        if (row['Customer Status'] == 0):
            x+=0.85
        if (row['Age'] > 50):
            x+=0.6
        if (row['Support Calls'] > 5):
            x+=0.6
        if (row['Last Payment Date'] == 7):
            x+=0.3
        if (row['Gender'] == 1):
            x+=0.2
        return min(x, 0.99)

    y_pred = X.apply(my_classifier, axis=1).tolist()
    yproba_my_classifier = X.apply(my_classifier_proba, axis=1).tolist()
    return y_pred, yproba_my_classifier