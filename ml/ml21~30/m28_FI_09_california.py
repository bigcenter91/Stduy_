import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor,GradientBoostingRegressor
from sklearn.metrics import r2_score
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor
import pandas as pd
#1.데이터
x,y = fetch_california_housing(return_X_y=True)

x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.7, shuffle= True, random_state=27
)

#2. 모델
model_list = [RandomForestRegressor(),
GradientBoostingRegressor(),
DecisionTreeRegressor(),
XGBRegressor()]

# 3. 훈련
for model, value in enumerate(model_list) :
    model = value
    model.fit(x_train,y_train)

    #4. 평가, 예측
    result = model.score(x_test,y_test)
    print("acc : ", result)

    y_predict = model.predict(x_test)
    acc = r2_score(y_test,y_predict)
    
    print("r2_score : ", acc)
    print(type(model).__name__, ":", \
        model.feature_importances_)
    print("=====================================================")
    
    # 하위 20-25%의 피처 제거 후 재학습
    # idx = np.argsort(model.feature_importances_)[int(len(model.feature_importances_) * 0.2) : int(len(model.feature_importances_) * 0.25)]
    # #argmin = np.argpartition(model.feature_importances_, 4)[:4]
    # x_drop = pd.DataFrame(x).drop(idx, axis=1)
    
    n_drop = int(len(model.feature_importances_) * 0.25)
    idx = np.argsort(model.feature_importances_)[-n_drop:]
    x_drop = pd.DataFrame(x).drop(x.columns[idx], axis=1)
    x_train1, x_test1, y_train1, y_test1 = train_test_split(x_drop, y, train_size=0.7, shuffle=True, random_state=123)

    model.fit(x_train1, y_train1)

    result = model.score(x_test1, y_test1)
    print("acc after feature selection: ", result)

    y_predict1 = model.predict(x_test1)
    acc1 = r2_score(y_test1, y_predict1)

    print("r2_score after feature selection: ", acc1)
    print(type(model).__name__, ":", model.feature_importances_)
    print("=====================================================")
    
# acc :  0.7987187301421566
# r2_score :  0.7987187301421566
# RandomForestRegressor : [0.52418479 0.05739953 0.04460882 0.03001002 0.03026337 0.13926806
#  0.08738024 0.08688519]
# =====================================================
# acc after feature selection:  0.8130820224831881
# r2_score after feature selection:  0.8130820224831881
# RandomForestRegressor : [0.52829378 0.05505275 0.05306151 0.03650294 0.13697658 0.0960581
#  0.09405435]
# =====================================================
# acc :  0.7767912106865944
# r2_score :  0.7767912106865944
# GradientBoostingRegressor : [0.60299474 0.03785063 0.02168392 0.00469068 0.00172628 0.12482813
#  0.10626763 0.099958  ]
# =====================================================
# acc after feature selection:  0.7903612246018082
# r2_score after feature selection:  0.7903612246018082
# GradientBoostingRegressor : [0.60695676 0.03057774 0.02982919 0.00501702 0.12526781 0.08695081
#  0.11540066]
# =====================================================
# acc :  0.6069463913020826
# r2_score :  0.6069463913020826
# DecisionTreeRegressor : [0.52139492 0.05579104 0.04984824 0.02992774 0.02929258 0.13310546
#  0.0931548  0.08748523]
# =====================================================
# acc after feature selection:  0.6373572746347802
# r2_score after feature selection:  0.6373572746347802
# DecisionTreeRegressor : [0.52434176 0.04586192 0.056156   0.03610193 0.13066019 0.10673231
#  0.10014589]
# =====================================================
# acc :  0.828240073318117
# r2_score :  0.828240073318117
# XGBRegressor : [0.44796085 0.07454932 0.04646437 0.02581183 0.02213857 0.15033498
#  0.12199391 0.11074615]
# =====================================================
# acc after feature selection:  0.840472601726478
# r2_score after feature selection:  0.840472601726478
# XGBRegressor : [0.45826235 0.0746479  0.05282211 0.02830956 0.16106074 0.10626029
#  0.11863706]