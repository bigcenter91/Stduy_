import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import cross_val_predict,cross_val_score
from sklearn.model_selection import train_test_split,KFold, StratifiedKFold #stratify=y 얘랑 동일함.
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
#1. 데이터
x,y = load_iris(return_X_y= True)

x_train, x_test, y_train, y_test = train_test_split(
    x,y, shuffle=True, random_state=337,train_size= 0.7, #stratify=y
)

n_splits = 5
#kfold = StratifiedKFold(n_splits= n_splits, shuffle= True, random_state= 337) #위의 split random_state와 무조건 같은 필요는 없음.
kfold = KFold(n_splits= n_splits, shuffle= True, random_state= 337)
#

#2. 모델

model = RandomForestClassifier()

#3. 컴파일, 훈련, 평가, 예측
score = cross_val_score(model, x_train,y_train, cv= kfold)
print('cross_val_score :', score, 
      '\n 교차검증평균점수 :', round(np.mean(score), 4))

y_predict = cross_val_predict(model, x_test,y_test, cv= kfold)
print('cross_val_predict ACC :', accuracy_score(y_test, y_predict))

# print("==============================================================")
#print(np.unique(y_train, return_counts=True))
#print(np.unique(y_test, return_counts=True))
#모델이 한쪽에 치우치게 되면 성능이 좋아지지않음.

# kfold: cross_val_predict ACC : 1.0
# StratifiedKFold :cross_val_predict ACC : 1.0