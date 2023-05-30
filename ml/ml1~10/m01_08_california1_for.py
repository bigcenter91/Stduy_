import numpy as np
from sklearn.datasets import load_boston,fetch_california_housing

#1. data
# datasets = load_iris()
# x = datasets.data
# y = datasets['target']
x,y = load_boston(return_X_y=True)

print(x.shape,y.shape) #(20640, 8) (20640,)

#2. model

from sklearn.svm import LinearSVC # 이 모델로 리폼.
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier,DecisionTreeRegressor #분류모델 DecisionTreeRegressor 회귀모델
from sklearn.ensemble import RandomForestRegressor,RandomForestClassifier
#tree모델 randomforest 는 숲이여서 나무들이 모여있음
#model =LinearSVC(C=10) #-> 알고리즘연산이 다 포함되어있음. = 비지도학습의 kmean과 비슷.
#model = LogisticRegression() # -> 얘는 원래 이진분류
#model = DecisionTreeRegressor() #-> 원래 안되는게 정상
#model = DecisionTreeClassifier() #분류모델
#model = RandomForestClassifier()
model = RandomForestRegressor() # 실수가 나오는 모델
#3. compile, fit

model.fit(x,y)

#4. evaluate , predict

results = model.score(x,y)


print(results) #0.9739684003422027
