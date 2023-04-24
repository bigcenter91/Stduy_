import numpy as np
import pandas as pd
from sklearn.datasets import load_iris, load_breast_cancer
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
# Y는 압축할 필요없겠지? // X만 압축을 한다
# target 값은 통상 y라고 할 수 있겠지?
# 차라리 차원축소라고 하면 다른 사람이 이해한다
# target값이 없다 비지도 target값을 생성하지 그럼 비지도


#1. 데이터
datasets = load_breast_cancer()

x = datasets['data']
y = datasets.target

print(x.shape, y.shape)
# (442, 10) (442,)

pca = PCA(n_components=5) # 차원을 줄이는 갯수가 된다_n_components
x = pca.fit_transform(x)
print(x.shape) # 442, 5


x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.8, random_state=1234, shuffle=True,
)

#2. 모델
from sklearn.ensemble import RandomForestRegressor
model = RandomForestRegressor(random_state=123) # 훈련할 때마다 바뀌니까 고정해놓는다

#3. 훈련
model.fit(x_train, y_train)

#4. 평가, 예측
results = model.score(x_test, y_test)
print("결과 : ", results) #r2겠지?
# 결과 :  0.42816745372688414

