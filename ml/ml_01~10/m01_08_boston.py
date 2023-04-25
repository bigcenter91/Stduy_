### 회귀 데이터들을 모아서



import numpy as np
from sklearn.datasets import load_breast_cancer, fetch_california_housing, load_boston


#1. 데이터
# datasets = load_iris()
# x = datasets.data
# y = datasets['target']

x, y = fetch_california_housing(return_X_y=True) # 이것도 가능해?

print(x.shape, y.shape) # (150, 4) (150,)

#2. 모델
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Dense
from sklearn.svm import LinearSVC # 리폼할거다
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier


# model = Sequential()
# model.add(Dense (10, activation='relu', input_shape=(4,)))
# model.add(Dense (10))
# model.add(Dense (10))
# model.add(Dense (10))
# model.add(Dense (3, activation='softmax'))
# model = LinearSVC(C=0.3) 
# model = LogisticRegression() # 분류모델이다// 한마디로 sigmoid
# model = DecisionTreeRegressor()
# model = DecisionTreeClassifier()
model = RandomForestRegressor()
# model = RandomForestClassifier()
# ValueError: Unknown label type: 'continuous'
# 위에 애를 판단할 수 없는거다


#3. 컴파일, 훈련
# model.compile(loss='sparse_categorical_crossentropy', # 원핫까지 포함되어있는거다_sparse_categorical_crossentropy
#               optimizer='adam',
#               metrics=['acc'])
model.fit(x,y)

# model.fit(x,y, epochs=100, validation_split=0.2)

#4. 평가, 예측
# results = model.evaluate(x, y) # results 출력하면 loss, acc 나오지

results = model.score(x,y)
print(results)
# 0.9666666666666667

# 머신러닝 안에 딥러닝이 들어간다
# 딥하지 않은 러닝 해보자
