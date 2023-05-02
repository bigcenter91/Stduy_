import numpy as np
import pandas as pd
from sklearn.datasets import load_iris, load_breast_cancer
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestRegressor
# Y는 압축할 필요없겠지? // X만 압축을 한다
# target 값은 통상 y라고 할 수 있겠지?
# 차라리 차원축소라고 하면 다른 사람이 이해한다
# target값이 없다 비지도 target값을 생성하지 그럼 비지도

#1. 데이터 
datasets = load_breast_cancer()
print(datasets.feature_names) 

x = datasets['data']
y = datasets.target

print(x.shape, y.shape)    # (569, 30) (569,)

for i in range(30, 0, -1):
    pca=PCA(n_components=i) # 압축 횟수
    x = pca.fit_transform(x)
    x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.8, random_state=123, shuffle=True,)
    model = RandomForestRegressor(random_state=123)
    model.fit(x_train, y_train)
    result = model.score(x_test, y_test)
    print(f"n_coponets={i},  결과: {result} ")