import numpy as np
from sklearn.datasets import load_breast_cancer


#1. 데이터
# datasets = load_iris()
# x = datasets.data
# y = datasets['target']

x, y = load_breast_cancer(return_X_y=True) # 이것도 가능해?

print(x.shape, y.shape) # (150, 4) (150,)

#2. 모델
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Dense
from sklearn.svm import LinearSVC # 리폼할거다
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.ensemble import RandomForestRegressor


# model = Sequential()
# model.add(Dense (10, activation='relu', input_shape=(4,)))
# model.add(Dense (10))
# model.add(Dense (10))
# model.add(Dense (10))
# model.add(Dense (3, activation='softmax'))
# model = LinearSVC(C=0.3) 
# model = DecisionTreeRegressor()
# model = DecisionTreeClassifier()
model = RandomForestRegressor()
# 모델 안에 대부분 알고리즘이 포함되어있다
# C가 작으면 작을 수록 직선이다 크면 곡선?
# 그렇게 중요한건 아니고 선 그어서 찾는거다 
# 한마디로 뭐야? 머신러닝도 선 긋는다


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
