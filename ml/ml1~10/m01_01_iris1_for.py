import numpy as np
from sklearn.datasets import load_iris,load_breast_cancer,load_digits,fetch_covtype,load_wine
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import RobustScaler
# Define a list of models
models = [LinearSVC(max_iter=10000), RandomForestClassifier()]

# Loop through each dataset and model, fit the model and print the score

datasets = [(load_iris(), 'Iris'), 
            (load_breast_cancer(), 'Breast Cancer'), 
            (load_digits(), 'Digits'), 
            (fetch_covtype(), 'Covtype'), 
            (load_wine(), 'Wine')]

scaler = RobustScaler()

for dataset, name in datasets:
    x, y = dataset['data'], dataset['target']  # unpack dataset object from tuple
    x = scaler.fit_transform(x)
    for model in models:
        model.fit(x, y)
        score = model.score(x, y)
        print(f"Dataset: {name}, Model: {type(model).__name__}, Score: {score}")
# Dataset: Iris, Model: LinearSVC, Score: 0.9466666666666667
# Dataset: Iris, Model: RandomForestClassifier, Score: 1.0
# Dataset: Breast Cancer, Model: LinearSVC, Score: 0.9876977152899824
# Dataset: Breast Cancer, Model: RandomForestClassifier, Score: 1.0
# Dataset: Digits, Model: LinearSVC, Score: 0.9938786867000556
# Dataset: Digits, Model: RandomForestClassifier, Score: 1.0
# Dataset: Covtype, Model: LinearSVC, Score: 0.7127150557991918
# Dataset: Covtype, Model: RandomForestClassifier, Score: 1.0
# Dataset: Wine, Model: LinearSVC, Score: 1.0
# Dataset: Wine, Model: RandomForestClassifier, Score: 1.0