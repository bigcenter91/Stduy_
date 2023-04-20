import warnings
warnings.filterwarnings('ignore') 

import numpy as np
from sklearn.datasets import load_iris, load_breast_cancer, load_wine, fetch_covtype, load_digits
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import MaxAbsScaler, MinMaxScaler, RobustScaler, StandardScaler

#1. 데이터 
datasets = [load_iris(return_X_y=True),load_breast_cancer(return_X_y=True),load_wine(return_X_y=True),
            load_digits(return_X_y=True),fetch_covtype(return_X_y=True)]

#2. 모델구성
models = [RandomForestRegressor(),DecisionTreeRegressor(),
          LogisticRegression(),LinearSVC()]

scaler = MinMaxScaler()

# Loop through the datasets and models, fit the models, and print the results
for i, dataset in enumerate(datasets):
    x, y = dataset
    x = scaler.fit_transform(x)
    print(f"\nResults for dataset {i+1}:")
    for j, model in enumerate(models):
        model.fit(x, y)
        score = model.score(x, y)
        print(f"  Model {j+1}: {score:.3f}")
