#교차검증

#1)데이터
import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_score, KFold #전처리
from sklearn.metrics import accuracy_score

#2)모델
from sklearn.ensemble import RandomForestRegressor

#1. 데이터
x,y = fetch_california_housing(return_X_y=True)

# 원래는 이렇게
# x_train, x_test, y_train, y_test = train_test_split(
#     x,y, shuffle=True, random_state= 224242, test_size= 0.2
# )

n_splits=5
kfold = KFold(n_splits = n_splits, shuffle = True, random_state = 413) #n_splits 는 분할갯수 5개면 20%
# kfold는 cross validation을 쓸 애
#2.모델구성
model = RandomForestRegressor()

#3,4. 컴파일, 훈련, 예측
scores = cross_val_score(model, x, y, cv = kfold)
#scores = cross_val_score(model, x, y, cv = 5)
#print(scores) #[0.93333333 0.93333333 0.93333333 1.         0.96666667]

print('ACC :', scores, 
      '\n cross_val_score average : ', round(np.mean(scores),4)) #리스트이므로 np로 바꿈.

# ACC : [0.80373487 0.81312403 0.81593601 0.81271135 0.80403083] 
#  cross_val_score average :  0.8099