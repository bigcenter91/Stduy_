import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import cross_val_predict, cross_val_score
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier


#1. 데이터
x, y = load_iris(return_X_y=True)

x_train, x_test, y_train, y_test = train_test_split(
    x, y, shuffle=True, random_state=337, test_size=0.2
)

n_splits=5
kfold = KFold(n_splits=n_splits, shuffle=True, random_state=337)

#2. 모델
model = SVC()
# model = RandomForestClassifier()

#3. 4. 컴파일, 훈련, 평가, 예측
score = cross_val_score(model, x_train, y_train, cv=kfold)
print('cross_val_score : ', score, '\n 교차검증평균점수 : ', round(np.mean(score), 4))

y_predict = cross_val_predict(model, x_test, y_test, cv=kfold)
print ('cross_val_predict ACC : ', accuracy_score(y_test, y_predict))