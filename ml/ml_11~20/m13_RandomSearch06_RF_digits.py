import numpy as np
import pandas as pd
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold, cross_val_score
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV # 크로스발리데이션이 엮여있다
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC, LinearSVC
import time


#1. 데이터
x, y = load_digits(return_X_y=True)

x_train, x_test, y_train, y_test = train_test_split(
    x, y, shuffle=True, random_state=337, test_size=0.2, #stratify=y
)

n_splits = 5
kfold = KFold(n_splits=n_splits, shuffle=True, random_state=337)

parameters = [
    {"C":[1,10,100,1000], "kernel": ['linear'], 'degree':[3,4,5]}, # 12 
    {"C":[1,10,100], 'kernel':['rbf'], 'gamma':[0.001, 0.0001]},  # 12
    {"C":[1,10,100,1000], "kernel": ['sigmoid'],
     'gamma':[0.01, 0.001, 0.001, 0.0001], 'degree':[3, 4]}, # 24
    {"C" : [0.1,1],'gamma':[1,10]},
    # 총 52번 돌겠지// 리스트 안에 딕셔너리 형태 안에 다시 리스트
    # 그리드 써치는 그냥 무조건 다 돈다 파라미터 계수만큼
]

#2. 모델 구성



# model = GridSearchCV(SVC(),parameters,cv=kfold,verbose=1,
#                      refit=True,n_jobs=-1,) 


model = RandomizedSearchCV(SVC(),parameters,
                           cv=5,
                           verbose=1,
                           refit=True,
                           n_iter=10,
                           n_jobs=-1,) 

#3. 컴파일,훈련
import time
start_time = time.time()
model.fit(x_train,y_train) 
end_time = time.time()

# #4.  평가,예측

print("최적의 매개변수 : ", model.best_estimator_)
print("최적의 파라미터 : ", model.best_params_)
print('best_score_ : ',model.best_score_) # train best score
print('model_score : ',model.score(x_test, y_test)) # test best score

y_predict = model.predict(x_test)
print('accuracy_score : ', accuracy_score(y_test, y_predict))
# accuracy_score :  1.0

y_pred_best = model.best_estimator_.predict(x_test)
print("최적의 튠 ACC", accuracy_score(y_test, y_pred_best))
# 최적의 튠 ACC 1.0

print("걸린시간 : ", round(end_time - start_time, 2),'초')