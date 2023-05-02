import numpy as np
from sklearn.datasets import load_wine
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
import pandas as pd
#1.데이터
x,y = load_wine(return_X_y=True)


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
    print(type(model).__name__, ":", \
        model.feature_importances_)
    print("=====================================================")
    
    # 하위 20-25%의 피처 제거 후 재학습
    #idx = np.argsort(model.feature_importances_)[int(len(model.feature_importances_) * 0.2) : int(len(model.feature_importances_) * 0.25)]
    # argmin = np.argpartition(model.feature_importances_, 4)[:4]
    # x_drop = pd.DataFrame(x).drop(argmin, axis=1)
    
    n_drop = int(len(model.feature_importances_) * 0.25)
    idx = np.argsort(model.feature_importances_)[-n_drop:]
    x_drop = pd.DataFrame(x).drop(x.columns[idx], axis=1) 
    x_train1, x_test1, y_train1, y_test1 = train_test_split(x_drop, y, train_size=0.7, shuffle=True, random_state=123)

    model.fit(x_train1, y_train1)

    result = model.score(x_test1, y_test1)
    print("acc after feature selection: ", result)

    y_predict1 = model.predict(x_test1)
    acc1 = accuracy_score(y_test1, y_predict1)

    print("accuracy_score after feature selection: ", acc1)
    print(type(model).__name__, ":", model.feature_importances_)
    print("=====================================================")
    
# acc :  1.0
# accuracy_score :  1.0    
# RandomForestClassifier : [0.12278688 0.031253   0.01129988 0.01847891 0.03613893 0.06624587
#  0.16207984 0.02049899 0.02004668 0.1768112  0.06719437 0.11055071
#  0.15661473]
# =====================================================
# acc after feature selection:  0.9814814814814815
# accuracy_score after feature selection:  0.9814814814814815
# RandomForestClassifier : [0.1379534  0.04058842 0.03265567 0.04202178 0.13811472 0.17862258
#  0.09639797 0.11545124 0.21819422]
# =====================================================
# acc :  0.9444444444444444
# accuracy_score :  0.9444444444444444
# GradientBoostingClassifier : [3.06756122e-06 3.80480229e-02 1.75655490e-02 2.29496697e-03
#  8.22450559e-03 6.73387696e-04 2.79689700e-01 2.25054664e-03
#  6.26263760e-07 2.97415260e-01 5.00543330e-02 1.41628361e-02
#  2.89617199e-01]
# =====================================================
# acc after feature selection:  0.8703703703703703
# accuracy_score after feature selection:  0.8703703703703703
# GradientBoostingClassifier : [0.06650164 0.02925908 0.00326042 0.01766975 0.07019548 0.26958641
#  0.21686278 0.0013643  0.32530016]
# =====================================================
# acc :  0.8611111111111112
# accuracy_score :  0.8611111111111112
# DecisionTreeClassifier : [0.         0.         0.         0.04025921 0.04990113 0.
#  0.17473837 0.         0.         0.32321345 0.         0.
#  0.41188784]
# =====================================================
# acc after feature selection:  0.9444444444444444
# accuracy_score after feature selection:  0.9444444444444444
# DecisionTreeClassifier : [0.02436339 0.04590204 0.01675789 0.02094736 0.         0.11491823
#  0.34844566 0.         0.42866542]
# =====================================================
# acc :  1.0
# accuracy_score :  1.0
# XGBClassifier : [0.0147584  0.04469927 0.00869847 0.00416202 0.02784223 0.02064051
#  0.14571643 0.01210101 0.01861065 0.14456782 0.03851257 0.39575252
#  0.12393806]
# =====================================================
# acc after feature selection:  0.9814814814814815
# accuracy_score after feature selection:  0.9814814814814815
# XGBClassifier : [0.0513827  0.01326958 0.01955407 0.06035079 0.03328762 0.17545512
#  0.29301172 0.19203249 0.16165592]
# =====================================================