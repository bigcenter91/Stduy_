import numpy as np
from sklearn.datasets import load_boston,fetch_california_housing,load_diabetes
from sklearn.svm import LinearSVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
import warnings
from sklearn.metrics import r2_score, accuracy_score
from sklearn.model_selection import cross_val_score, KFold
warnings.filterwarnings(action= 'ignore') #경고 무시

data_list = [load_boston(return_X_y=True),
             fetch_california_housing(return_X_y=True),
             load_diabetes(return_X_y=True),] #->매우중요!!!!!!!!! for문을 쓰기위해 list형태로 만들어 놓음. 

model_list = [LinearSVR(),
              DecisionTreeRegressor(),
              RandomForestRegressor()]

data_name_list = ['boston : ',
                  'california :',
                  'diabetes :']

model_name_list = ['LinearSVR',
              'DecisionTreeRegressor',
              'RandomForestRegressor']

n_splits=5
kfold = KFold(n_splits = n_splits, shuffle = True, random_state = 413)

#2. 모델
for  i, value in enumerate(data_list): #enumerate수치와 순서를 나타내주는 함수.
    x, y = value #첫번째 iris들어가고 두번째 cancer가 들어감
    #print(x.shape,y.shape)
    print("=============================")
    print(data_name_list[i])
    
    for j, value2 in enumerate(model_list):
        model = value2 #j가 들어가면 데이터값이 나와버림

        #4. 평가, 예측
        scores = cross_val_score(model, x, y, cv = kfold)
        print(model_name_list[j], 'ACC :' ,scores,
              '\n cross_val_score average : ', round(np.mean(scores),4))
