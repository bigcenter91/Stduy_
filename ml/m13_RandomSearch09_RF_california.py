from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_california_housing
from tensorflow.python.keras.callbacks import EarlyStopping
from sklearn.metrics import accuracy_score,r2_score
from sklearn.preprocessing import MinMaxScaler,StandardScaler
from sklearn.model_selection import KFold,cross_val_score,GridSearchCV,StratifiedKFold
from sklearn.ensemble import RandomForestRegressor 
import numpy as np
from sklearn.model_selection import RandomizedSearchCV

#1. 데이터
datasets = fetch_california_housing()
x = datasets.data
y = datasets.target


x_train, x_test ,y_train, y_test = train_test_split(
          x, y, train_size=0.8,shuffle=True,random_state=100)

kfold = KFold(n_splits=5, shuffle=True, random_state=66)


scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)


parameters = [
    {'n_estimators':[100, 200],'max_depth':[6, 8],'min_samples_leaf':[3,5],
     'min_samples_split':[2, 3],'n_jobs':[-1, 2]},
    {'n_estimators':[300, 400],'max_depth':[6, 8],'min_samples_leaf':[7, 10],
     'min_samples_split':[4, 7],'n_jobs':[-1, 4]}
    ]     

#2. 모델 구성
model = RandomizedSearchCV(RandomForestRegressor(),
                           parameters,
                           cv=kfold,
                           verbose=1,
                           refit=True,
                           n_jobs=-1) 

#3. 컴파일,훈련
import time
start = time.time()
model.fit(x_train,y_train) 
end = time.time()- start

print("최적의 매개변수 :",model.best_estimator_)
print("최적의 파라미터 :",model.best_params_)
print("best_score :",model.best_score_)
print("model_score :",model.score(x_test,y_test))
y_predict = model.predict(x_test)
print('accuracy_score :',r2_score(y_test,y_predict))
y_pred_best = model.best_estimator_.predict(x_test)
print('최적 튠  ACC :',r2_score(y_test,y_predict))
print("걸린 시간 :",round(end,2),"초")

# 최적의 매개변수 : RandomForestRegressor(max_depth=8, min_samples_leaf=7, min_samples_split=4,
#                       n_estimators=300, n_jobs=-1)
# 최적의 파라미터 : {'n_jobs': -1, 'n_estimators': 300, 'min_samples_split': 4, 'min_samples_leaf': 7, 'max_depth': 8}
# best_score : 0.7479570749269806
# model_score : 0.7622703601668357
# accuracy_score : 0.7622703601668358
# 최적 튠  ACC : 0.7622703601668358
# 걸린 시간 : 112.87 초