import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler,StandardScaler
from sklearn.metrics import r2_score
import warnings
warnings.filterwarnings("ignore")
from sklearn.ensemble import RandomForestRegressor
from sklearn.utils import all_estimators
import sklearn as sk
#print(sk.__version__) #1.0.2

#1. 데이터
x,y = fetch_california_housing(return_X_y=True)

x_train, x_test, y_train, y_test = train_test_split(x,y,
                        shuffle=True, random_state= 1543, train_size= 0.7)

scaler = StandardScaler()
x_train =scaler.fit_transform(x_train)
x_test =scaler.transform(x_test)

#2. 모델구성

#model = RandomForestRegressor(n_jobs= 4, n_estimators= 100) # n_estimators = epochs 랑같은 의미, n_jobs 는 코어 4개 돌린다는 것.
allAlgorithms = all_estimators(type_filter='regressor')
# allAlgorithms = all_estimators(type_filter='classifier') 분류
i = 0
max_r2 = 0
max_name = '최대값'
for (name , algorithm) in allAlgorithms :
    i+= 1
    try: #예외 처리
        model =  algorithm()
        print("================================================")
        #3. 훈련
        model.fit(x_train, y_train)
        print(i,'번')
        #4. 평가, 예측

        results = model.score(x_test, y_test)
        print(name, '의 정답률 :',results)
        
        if max_r2 < results :
            max_r2 = results #earlystopping이랑 비슷함(계속 최댓값만 저장.)
            max_name = name
        # y_predict = model.predict(x_test)
        # # print(y_test.dtype)     #float64
        # # print(y_predict.dtype)  #float64

        # r2 =r2_score(y_test, y_predict)
        # print("r2_score :", r2)    
    except:
        print(name, "에러")
        #에러가 뜨는 모델들은 파라미터 수정해줘야됨.
print("================================================")
print('최고모델 :', max_name, max_r2)
print("================================================")
# print('allAlgorithms :', allAlgorithms) #모든 모델들이 들어가있음. Regressor
# print('모델의 개수 :', len(allAlgorithms)) #모델의 개수 : 55

# ================================================
# 최고모델 : HistGradientBoostingRegressor 0.8355493131175084
# ================================================