########회귀 데이터 싹 모아서 테스트############

import numpy as np
from sklearn.datasets import fetch_california_housing

#1. data
# datasets = load_iris()
# x = datasets.data
# y = datasets['target']
x,y = fetch_california_housing(return_X_y=True)

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

#Traceback (most recent call last):
#   File "c:\Users\bitcamp\Documents\GitHub\aia_study\ml\m01_08_boston.py", line 27, in <module>
#     model.fit(x,y)
#   File "C:\Users\bitcamp\anaconda3\envs\tf274gpu\lib\site-packages\sklearn\ensemble\_forest.py", line 367, in fit
#     y, expanded_class_weight = self._validate_y_class_weight(y)
#   File "C:\Users\bitcamp\anaconda3\envs\tf274gpu\lib\site-packages\sklearn\ensemble\_forest.py", line 734, in _validate_y_class_weight
#     check_classification_targets(y)
#   File "C:\Users\bitcamp\anaconda3\envs\tf274gpu\lib\site-packages\sklearn\utils\multiclass.py", line 197, in check_classification_targets
#     raise ValueError("Unknown label type: %r" % y_type)
# ValueError: Unknown label type: 'continuous
# -> 회귀평가에 분류평가를 사용하면 이런 오류가뜸