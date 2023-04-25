# 10개 데이터셋
# 10개의 파일을 만든다.
# [실습/과제] 피처를 한개씩 삭제하고 성능 비교
# 모델은 RF로만 한다.
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, MaxAbsScaler, StandardScaler, RobustScaler
from sklearn.datasets import load_iris, load_breast_cancer, load_digits, load_wine
from sklearn.metrics import accuracy_score

#1 데이터
dacon_diabetes_path = 'c:/Stduy_/_data/dacon_diabetes/'

dacon_diabetes = pd.read_csv(dacon_diabetes_path + 'train.csv', index_col = 0).dropna()

x = dacon_diabetes.drop(['Outcome'], axis = 1)
y = dacon_diabetes['Outcome']


data_list = [load_iris(return_X_y = True),
             load_breast_cancer(return_X_y = True),
             load_digits(return_X_y = True),
             load_wine(return_X_y = True),
             (x, y)]

data_list_name = ['아이리스',
                  '캔서',
                  '디지트',
                  '와인',
                  '데이콘']

for i in range(len(data_list)):
    x, y = data_list[i]
    x_train, x_test, y_train, y_test = train_test_split(x, y, train_size = 0.7, shuffle = True, random_state = 123)
    
    scaler = MinMaxScaler()
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)
    
    #2 모델
    model = RandomForestClassifier()

    #3 컴파일, 훈련
    model.fit(x_train, y_train)

    #4 평가, 예측
    result = model.score(x_test, y_test)
    print(data_list_name[i], 'result : ', result)

    y_predict = model.predict(x_test)

    acc = accuracy_score(y_test, y_predict)
    print(data_list_name[i], 'acc : ', acc)
    print(model, ':', model.feature_importances_) # x 열 중요도
    argmin = np.argmin(model.feature_importances_, axis = 0)
    
    x_drop = pd.DataFrame(x).drop(argmin, axis = 1)
    
    x_train1, x_test1, y_train1, y_test1 = train_test_split(x, y, train_size = 0.7, shuffle = True, random_state = 123)
    
    model.fit(x_train1, y_train1)
    
    result = model.score(x_test1, y_test1)
    print(data_list_name[i], 'result : ', result)
    
    y_predict1 = model.predict(x_test1)
    
    acc1 = accuracy_score(y_test1, y_predict1)
    print(data_list_name[i], 'acc1 : ', acc1)