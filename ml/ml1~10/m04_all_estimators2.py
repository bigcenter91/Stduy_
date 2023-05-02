# 분류 만들어!!!
import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler,StandardScaler
from sklearn.metrics import r2_score
import warnings
warnings.filterwarnings("ignore")
from sklearn.ensemble import RandomForestRegressor
from sklearn.utils import all_estimators

#1. 데이터
x,y = load_digits(return_X_y=True)

x_train, x_test, y_train, y_test = train_test_split(x,y,
                        shuffle=True, random_state= 1543, train_size= 0.7)

scaler = RobustScaler()
x_train =scaler.fit_transform(x_train)
x_test =scaler.transform(x_test)

#2. 모델구성

allAlgorithms = all_estimators(type_filter='classifier')

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
    
    except:
        print(name, "에러")

print("================================================")
print('최고모델 :', max_name, max_r2)
print("================================================")

# 39 번
# SVC 의 정답률 : 0.9648148148148148
# StackingClassifier 에러
# VotingClassifier 에러
# ================================================
# 최고모델 : ExtraTreesClassifier 0.9814814814814815
# ================================================

