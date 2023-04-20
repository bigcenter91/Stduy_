### 분류데이터들을 싹 모아서

import numpy as np
from sklearn.datasets import load_iris, load_breast_cancer, load_wine, fetch_covtype, load_digits
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import r2_score, accuracy_score
import warnings
warnings.filterwarnings(action='ignore')



#1. 데이터
data_list = [load_iris(return_X_y = True),
             load_breast_cancer(return_X_y = True),
             load_wine(return_X_y = True)]

model_list = [LinearSVC(),
              LogisticRegression(),
              DecisionTreeClassifier(),
              RandomForestClassifier()]

data_name_list = ['아이리스 : ',
                  '브레스트 캔서 : ',
                  '와인 : ']

model_name_list = ['LinearSVC : ',
'LogisticRegression:: ',
'DecisionTreeClassifier : ',
'RandomForestClassifier : ']


#2. 모델
for i, v in enumerate(data_list): # i = 인덱스값 순서, v = 데이터값, enumerate
    x, y = v
    # print(x.shape, y.shape)
    print("===================================")
    print(data_name_list[i])
    for j , v2 in enumerate(model_list):
        model = v2
        
        #3. 컴파일, 훈련
        model.fit(x, y)
        
        #4. 평가, 예측
        results = model.score(x, y) # evaluate 대신 분류에선 당연히 acc 회귀에선 r2_score
        print(model_name_list[j], results)
        y_predict = model.predict(x)
        acc = accuracy_score(y, y_predict)
        print(model_name_list[j], "accuracy_score:", acc)
        
        
'''
# 모델 안에 대부분 알고리즘이 포함되어있다
# C가 작으면 작을 수록 직선이다 크면 곡선?
# 그렇게 중요한건 아니고 선 그어서 찾는거다 
# 한마디로 뭐야? 머신러닝도 선 긋는다


#3. 컴파일, 훈련
# model.compile(loss='sparse_categorical_crossentropy', # 원핫까지 포함되어있는거다_sparse_categorical_crossentropy
#               optimizer='adam',
#               metrics=['acc'])
model.fit(x,y)

# model.fit(x,y, epochs=100, validation_split=0.2)

#4. 평가, 예측
# results = model.evaluate(x, y) # results 출력하면 loss, acc 나오지

results = model.score(x,y)
print(results)
'''
# iris
# DecisionTreeRegressor : 1.0
# DecisionTreeClassifier : 1.0
# RandomForestRegressor : 0.99281
# RandomForestClassifier : 1.0

# load_breast_cancer
# DecisionTreeRegressor : 1.0
# DecisionTreeClassifier : 1.0
# RandomForestRegressor : 0.9795003924211194
# RandomForestClassifier : 1.0

# load_wine
# DecisionTreeRegressor : 1.0
# DecisionTreeClassifier : 1.0
# RandomForestRegressor : 0.9923626948480846
# RandomForestClassifier : 1.0

# fetch_covtype
# DecisionTreeRegressor : 1.0
# DecisionTreeClassifier : 1.0
# RandomForestRegressor : 0.9911982879788639
# RandomForestClassifier : 1.0
# '''

# 머신러닝 안에 딥러닝이 들어간다
# 딥하지 않은 러닝 해보자


