import numpy as np

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold, cross_val_score
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score

# 설명했던 교차검증(cross_val)
# 데이터를 n빵친다

#1. 데이터
x, y = load_iris(return_X_y=True)

# x_train, x_test, y_train, y_test = train_test_split(
#     x, y, shuffle=True, random_state=333, test_size=0.2
# )

n_splits = 5 # 몇일 때가 좋을까? 모른다 통상적으로 5를 쓴다 데이터 양에 따라 다르다 // 디폴트가 5
kfold = KFold(n_splits=n_splits, shuffle=True, random_state=123)
# kfold = KFold()
# shuffle은 한번만 한다
#import 했으면 정의하고 정의한걸 사용한다

#2. 모델구성
model = LinearSVC()

#3, 4. 컴파일, 훈련, 평가, 예측
# scores = cross_val_score(model, x, y, cv=kfold)
scores = cross_val_score(model, x, y, cv=5)
print(scores)

# [0.96666667 1.         0.93333333 0.93333333 0.9       ]

print ('ACC :', scores, 
       '\n cross_val_score 평균 : ', round(np.mean(scores), 4))


# cross_validation 개념, 사용법 정리 