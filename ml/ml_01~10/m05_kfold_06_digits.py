import numpy as np

from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold, cross_val_score
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score

#1. 데이터
x, y = load_digits(return_X_y=True)

# x_train, x_test, y_train, y_test = train_test_split(
#     x, y, shuffle=True, random_state=333, test_size=0.2
# )

n_splits = 5
kfold = KFold(n_splits=n_splits, shuffle=True, random_state=123)


#2. 모델구성
model = LinearSVC()

#3, 4. 컴파일, 훈련, 평가, 예측
scores = cross_val_score(model, x, y, cv=kfold)
print(scores)


print ('ACC :', scores, 
       '\n cross_val_score 평균 : ', round(np.mean(scores), 4))

