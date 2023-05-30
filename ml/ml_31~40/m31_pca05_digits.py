import numpy as np
import pandas as pd
from sklearn.datasets import load_iris, load_diabetes, load_breast_cancer, load_wine, load_digits
from sklearn.ensemble import RandomForestRegressor
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split

#1. 데이터 
datasets = load_digits()
print(datasets.feature_names) 

x = datasets['data']
y = datasets.target

print(x.shape, y.shape)    # (1797, 64) (1797,)

for i in range(64, 0, -1):
    pca=PCA(n_components=i)
    x = pca.fit_transform(x)
    
    x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.8, random_state=123, shuffle=True,)
    
    model = RandomForestRegressor(random_state=123)
    
    model.fit(x_train, y_train)
    
    result = model.score(x_test, y_test)
    print(f"n_coponets={i},  결과: {result} ")