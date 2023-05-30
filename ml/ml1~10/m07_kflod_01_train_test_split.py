import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import cross_val_predict,cross_val_score
from sklearn.model_selection import train_test_split,KFold
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
#1. 데이터
x,y = load_iris(return_X_y= True)

x_train, x_test, y_train, y_test = train_test_split(
    x,y, shuffle=True, random_state=337,train_size= 0.7)

n_splits = 5 #-> 70% * 5 = 350 훈련
kfold = KFold(n_splits= n_splits, shuffle= True, random_state= 337) #위의 split random_state와 무조건 같은 필요는 없음.

#2. 모델

model = RandomForestClassifier()

#3. 컴파일, 훈련, 평가, 예측
score = cross_val_score(model, x_train,y_train, cv= kfold)
print('cross_val_score :', score, 
      '\n 교차검증평균점수 :', round(np.mean(score), 4))

y_predict = cross_val_predict(model, x_test,y_test, cv= kfold)
print('cross_val_predict ACC :', accuracy_score(y_test, y_predict))

# cross_val_score : [1.         0.85714286 1.         0.95238095 0.9047619 ] 
#  교차검증평균점수 : 0.9429
# cross_val_predict ACC : 0.9555555555555556
