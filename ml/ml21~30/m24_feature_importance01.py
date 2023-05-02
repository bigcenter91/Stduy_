#m17-1카피
import numpy as np
from sklearn.datasets import load_iris
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

#1.데이터
x,y = load_iris(return_X_y=True)

x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.8, shuffle= True, random_state=27
)

scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

#2. 모델
model_list = [DecisionTreeClassifier(),
RandomForestClassifier(),
GradientBoostingClassifier(),
XGBClassifier()]

# model_name_list = ['DecisionTreeClassifier',
# 'RandomForestClassifier',
# 'GradientBoostingClassifier',
# 'XGBClassifier']

for i, value in enumerate(model_list):
    model = value
    # 3. 훈련
    model.fit(x_train,y_train)

    #4. 평가, 예측
    result = model.score(x_test,y_test)
    print("acc : ", result)

    y_predict = model.predict(x_test)
    acc = accuracy_score(y_test,y_predict)
    print("accuracy_score : ", acc)
    print("=====================================================")
    print(type(model).__name__, ":", model.feature_importances_) #출력 : DecisionTreeClassifier() tree계열에게만 적용

# DecisionTreeClassifier() : [0.01669101 0.         0.55667565 0.42663335] 0.9
# RandomForestClassifier() : [0.08766767 0.02793797 0.37666757 0.50772679] 0.9
# GradientBoostingClassifier() : [0.00693138 0.01181204 0.24868812 0.73256847] 0.9333333333333333
# XGBClassifier():           [0.02545078 0.03227026 0.5598112  0.38246778] 0.9333333333333333

# feature 를 0.9의 정확도를 믿는 전재하에 가장큰 컬럼을 위주로 한다.
# 낮은놈부터 제거하고 큰놈을 나중에 제거한다.

# for i, value in enumerate(model_list):
#     model = value
#     # 3. 훈련
#     model.fit(x_train,y_train)

#     #4. 평가, 예측
#     result = model.score(x_test,y_test)
#     print("acc : ", result)

#     y_predict = model.predict(x_test)
#     acc = accuracy_score(y_test,y_predict)
#     print("accuracy_score : ", acc)
#     print("=====================================================")
#      if i !=3:
#        print(model, ":", model.feature_importances_)
    #  else:
    #     print('XGBClassifier():', model.feature_importances_)