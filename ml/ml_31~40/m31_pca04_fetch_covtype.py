import numpy as np
import pandas as pd
from sklearn.datasets import fetch_covtype
from sklearn.decomposition import PCA #분해 #1.비지도, 2.전처리
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
#pca로 차원축소(컬럼압축)을 할 때 필요없는 데이터를 압축함으로써 성능이 좋아 질 수도 있음. (y를 압축하지 않음)

#1. 데이터

datasets = fetch_covtype()

x = datasets['data']
y = datasets.target

print(x.shape, y.shape) #(581012, 54) (581012,)
for i in range(54, 0, -3):
    pca = PCA(n_components=i)
    x = pca.fit_transform(x)
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, shuffle=True, random_state=123, train_size=0.8
    )
    # 2. 모델
    model = RandomForestClassifier(random_state=123)

    # 3. 훈련
    model.fit(x_train, y_train)

    # 4. 결과
    results = model.score(x_test, y_test)
    print("n_components={}: 결과는 {}".format(i, results))