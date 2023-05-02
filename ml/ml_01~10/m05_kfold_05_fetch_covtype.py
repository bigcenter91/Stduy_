import numpy as np

from sklearn.datasets import fetch_covtype
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold, cross_val_score
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score
import time

# 설명했던 교차검증(cross_val)
# 데이터를 n빵친다

#1. 데이터
x, y = fetch_covtype(return_X_y=True)

# x_train, x_test, y_train, y_test = train_test_split(
#     x, y, shuffle=True, random_state=333, test_size=0.2
# )

n_splits = 5
kfold = KFold(n_splits=n_splits, shuffle=True, random_state=123)


#2. 모델구성
model = LinearSVC()

#3, 4. 컴파일, 훈련, 평가, 예측
start = time.time()
scores = cross_val_score(model, x, y, cv=kfold)
end= time.time()
print(scores)

# [0.90350877 0.92105263 0.93859649 0.93859649 0.7079646 ]

print ('ACC :', scores, 
       '\n cross_val_score 평균 : ', round(np.mean(scores), 4))
print('걸린시간 :', round(end-start,2),'초')
# ACC : [0.53508772 0.85087719 0.8245614  0.75438596 0.85840708]
#  cross_val_score 평균 :  0.7647