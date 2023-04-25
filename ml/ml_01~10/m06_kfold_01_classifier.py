import numpy as np
from sklearn.datasets import load_iris, load_breast_cancer, load_wine, fetch_covtype, load_digits, load_diabetes,fetch_california_housing
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import r2_score, accuracy_score
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.preprocessing import MinMaxScaler, MaxAbsScaler
from sklearn.model_selection import KFold, cross_val_score, cross_val_predict
import warnings
warnings.filterwarnings(action='ignore')
from sklearn.utils import all_estimators



#1. 데이터
datasets = [load_iris(return_X_y = True),
             load_breast_cancer(return_X_y = True),
             load_wine(return_X_y = True),
             load_digits(return_X_y = True)]
             #fetch_covtype(return_X_y = True)

data_name_list = ['아이리스 : ',
                  '브레스트 캔서 : ',
                  '와인 : ',
                  '디지트 : ']
                #   '패치코브타입 : '

n_splits = 5
kfold = KFold(n_splits=n_splits, shuffle=True, random_state=123)


model_name_list = ['LinearSVC : ',
'LogisticRegression:: ',
'DecisionTreeClassifier : ',
'RandomForestClassifier : ']


#2. 데이터
for i, v in enumerate(datasets): # i = 인덱스값 순서, v = 데이터값, enumerate
    x, y = v
        
        
# all_estimators : 모든 모델에 대한 평가 (회귀 55개 모델)

#2. 모델구성 
allAlgorithms = all_estimators(type_filter='classifier')    #회귀모델

max_score = 0
max_name = '바보'

###for문 단짝 = 예외처리(try, except)###
for (name, algorithm) in allAlgorithms:
    try: #예외처리
        model = algorithm()
        
        scores = cross_val_score(model, x, y, cv=kfold,)
        results = round(np.mean(scores), 4)
        

        if max_score < results: 
            max_score = results
            max_name = name
        

    except: 
        # print('error') # error : 모델 내부의 파라미터를 조정해줘야함 (디폴트로는 안됨)
        # print(name, '[error]')
        continue

print("=========", data_name[index], "===========")        
print('최고모델:', max_name, max_score)
print("===========================")  


#참고
# print(y_test.dtype)    #float64 #메모리 터질때, y_test.astype(float32)로 변경하면 메모리 반으로줄면서 해결가능 
# print(y_predict.dtype) #float64
# y_predict = model.predict(x_test)
# r2 = r2_score(y_test, y_predict)
# print("r2_score: ", r2)
