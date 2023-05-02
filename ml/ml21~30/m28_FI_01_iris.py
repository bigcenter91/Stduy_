#실습
#피처임포턴스가 전체 중요도에서 하위 20~25퍼 컬럼들을 제거
#재구성후
#모델을 돌려서 결과도출

#기존모델들과 성능비교

#2. 모델구성


import numpy as np
from sklearn.datasets import load_iris
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
import pandas as pd
#1.데이터
x,y = load_iris(return_X_y=True)


x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.8, shuffle= True, random_state=27
)

#2. 모델
model_list = [RandomForestClassifier(),
GradientBoostingClassifier(),
DecisionTreeClassifier(),
XGBClassifier()]

# 3. 훈련
for model, value in enumerate(model_list) :
    model = value
    model.fit(x_train,y_train)

    #4. 평가, 예측
    result = model.score(x_test,y_test)
    print("acc : ", result)

    y_predict = model.predict(x_test)
    acc = accuracy_score(y_test,y_predict)
    
    print("accuracy_score : ", acc)
    print(type(model).__name__, ":", model.feature_importances_)
    print("=====================================================")
    
    # 하위 20-25%의 피처 제거 후 재학습
    # idx = np.argsort(model.feature_importances_)[int(len(model.feature_importances_) * 0.2) : int(len(model.feature_importances_) * 0.25)]
    # x_drop = pd.DataFrame(x).drop(idx, axis=1)
    n_drop = int(len(model.feature_importances_) * 0.25)
    idx = np.argsort(model.feature_importances_)[-n_drop:]
    x_drop = pd.DataFrame(x).drop(x.columns[idx], axis=1) 
    x_train1, x_test1, y_train1, y_test1 = train_test_split(x_drop, y, train_size=0.7, shuffle=True, random_state=123)

    model.fit(x_train1, y_train1)

    result = model.score(x_test1, y_test1)
    print("acc after feature selection: ", result)

    y_predict1 = model.predict(x_test1)
    acc1 = accuracy_score(y_test, y_predict1)

    print("accuracy_score after feature selection: ", acc1)
    print(type(model).__name__, ":", model.feature_importances_)
    print("=====================================================")


# acc :  0.8666666666666667
# accuracy_score :  0.8666666666666667
# RandomForestClassifier : [0.07230817 0.02371527 0.44654526 0.45743129]
# =====================================================
# acc after feature selection:  0.9333333333333333
# accuracy_score after feature selection:  0.9333333333333333
# RandomForestClassifier : [0.18258641 0.45661218 0.36080141]
# =====================================================
# acc :  0.9333333333333333
# accuracy_score :  0.9333333333333333
# GradientBoostingClassifier : [0.01282792 0.38406947 0.60310261]
# =====================================================
# acc after feature selection:  0.9111111111111111
# accuracy_score after feature selection:  0.9111111111111111
# GradientBoostingClassifier : [5.19695270e-04 2.37717609e-02 2.28524614e-01 7.47183930e-01]
# =====================================================
# acc :  0.9555555555555556
# accuracy_score :  0.9555555555555556
# DecisionTreeClassifier : [0.01364196 0.01435996 0.5461181  0.42587999]
# =====================================================
# acc after feature selection:  0.9555555555555556
# accuracy_score after feature selection:  0.9555555555555556
# DecisionTreeClassifier : [0.02871991 0.06543706 0.90584302]
# =====================================================
# acc :  0.9111111111111111
# accuracy_score :  0.9111111111111111
# XGBClassifier : [0.03066424 0.37356448 0.59577125]
# =====================================================
# acc after feature selection:  0.9333333333333333
# accuracy_score after feature selection:  0.9333333333333333
# XGBClassifier : [0.01228447 0.02874331 0.40544182 0.5535304 ]

