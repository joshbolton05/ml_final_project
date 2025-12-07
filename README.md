# ML Customer Churn Final Project

This repository is an example template that demonstrates how to structure a machine learning project for reproducibility. This uses basic machine learning principles and learning models to determine Customer Churn

## Purpose

This project uses basic machine learning principles and learning models to determine Customer Churn.

## Project layout

```
.
├── main.py                 # Entry point that runs the entire pipeline
├── requirements.txt        # Python dependencies
├── data/
│   ├── processed/
|   |     cleaned_test.csv  # cleaned test data
|   |     cleaned_train.csv # cleaned training data
│   └── raw/
│       └── test.csv # original testing data file
|           train.csv # original training data file
├── notebooks/
│   └── ml_final_project.ipynb # project notebook
└── src/
    ├── data/ # all data processing related files
    │   ├── load_training_data.py
            load_test_data.py
    │   ├── preprocess_training.py
            preprocess_test.py
    │   └── split_data.py
    ├── models/ # my three models
    │   ├── my_classifier_model.py
    │   ├── my_knn_model.py
    │   └── my_decision_tree_model.py
    └── visualization/ # all visualization files
        ├── ml_eda.py
        └── performance.py
```

`main.py` imports the modules inside `src/` and executes them to reproduce the analysis and results.

## Running the example

Install the dependencies and run the pipeline. You should use the versions of the dependencies as specified by the requirements file:

```bash
conda create -n ml_final_project_env --file requirements.txt
conda activate ml_final_project_env
python main.py
```

This will load the dataset, perform basic feature engineering, train my few models and produce some visualizations similar to those in the notebook.
The cleaned data will be written to `data/processed/` and all plots will be displayed interactively as you run through the main.py file.
