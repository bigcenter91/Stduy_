# all_estimators : 모든 모델에 대한 평가 (회귀 55개 모델)

import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.metrics import r2_score
from sklearn.utils import all_estimators

import sklearn as sk
print(sk.__version__)    #1.0.2
import warnings
warnings.filterwarnings('ignore')



#1. 데이터 
x,y = fetch_california_housing(return_X_y=True)

x_train, x_test, y_train, y_test=train_test_split(
    x, y, shuffle=True, random_state=123, test_size=0.2)

scaler = RobustScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

#2. 모델구성 
# model = RandomForestRegressor(n_estimators=120, n_jobs=4) #n_estimators :트리 개수 = epochs/  #n_jobs : cpu사용 개수(4개 모두사용하겠다) 
allAlgorithms = all_estimators(type_filter='regressor')    #회귀모델
# allAlgorithms = all_estimators(type_filter='classifier')  #분류모델

print('allAlgorithms:', allAlgorithms)
print('모델의 개수 :', len(allAlgorithms)) #55
max_r2 = 0
max_name = 'max_model'

###for문 단짝 = 예외처리(try, except)###
for (name, algorithm) in allAlgorithms:
    try: #예외처리
        model = algorithm()
        #3. 컴파일, 훈련 
        model.fit(x_train, y_train)
        #4. 평가, 예측 
        results = model.score(x_test, y_test)
        print(name, '의 정답률:', results )

        if max_r2 < results: 
            max_r2 = results
            max_name = name

    except: 
        # print('error') # error : 모델 내부의 파라미터를 조정해줘야함 (디폴트로는 안됨)
        print(name, '[error]')

print('===========================')        
print('최고모델:', max_name, max_r2)
print('===========================')  


#참고
# print(y_test.dtype)    #float64 #메모리 터질때, y_test.astype(float32)로 변경하면 메모리 반으로줄면서 해결가능 
# print(y_predict.dtype) #float64
# y_predict = model.predict(x_test)
# r2 = r2_score(y_test, y_predict)
# print("r2_score: ", r2)
