from sklearn.model_selection import train_test_split, cross_val_score, cross_val_predict, KFold
from sklearn.metrics import r2_score, accuracy_score
from sklearn.svm import SVR
import numpy as np
from sklearn.datasets import load_boston


#1. 데이터
x, y = load_boston(return_X_y=True)

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, random_state=9)
                      
n_splits = 5              
kfold = KFold(n_splits=n_splits, shuffle=True, random_state=123)
                    
                                      
#2. 모델구성
model = SVR()


#3. 컴파일, 훈련, 평가, 예측
scores = cross_val_score(model, x_train, y_train, cv=kfold)               # cv=5 라면 kfold를 5로 쓴다
y_predict = cross_val_predict(model, x_test, y_test, cv=kfold)
r2 = r2_score(y_test, y_predict)

print('r2 :', scores, '\ncross_val_score :' , round(np.mean(scores), 4)) # 4번째까지 출력 (반올림을 5번째 자리에서)
print(y_predict)
print('cross_val_predict r2 : ', r2)