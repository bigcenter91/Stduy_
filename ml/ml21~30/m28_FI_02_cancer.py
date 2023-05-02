import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
import pandas as pd
#1.데이터
x,y = load_breast_cancer(return_X_y=True)


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
    # idx = np.argsort(model.feature_importances_)[int(len(model.feature_importances_) * 0.2) : int(len(model.feature_importances_) * 0.25)]
    # x_drop = pd.DataFrame(x).drop(idx, axis=1)
    n_drop = int(len(model.feature_importances_) * 0.25)
    idx = np.argsort(model.feature_importances_)[-n_drop:]
    x_drop = pd.DataFrame(x).drop(x.columns[idx], axis=1) 
    x_train, x_test, y_train, y_test = train_test_split(x_drop, y, train_size=0.7, shuffle=True, random_state=123)

    model.fit(x_train, y_train)

    result = model.score(x_test, y_test)
    print("acc after feature selection: ", result)

    y_predict = model.predict(x_test)
    acc = accuracy_score(y_test, y_predict)

    print("accuracy_score after feature selection: ", acc)
    print(type(model).__name__, ":", model.feature_importances_)
    print("=====================================================")
    
# acc :  0.9736842105263158
# accuracy_score :  0.9736842105263158
# RandomForestClassifier : [0.05180307 0.01642365 0.03084502 0.03192475 0.00562553 0.02085028
#  0.02732273 0.07357097 0.00209574 0.00382761 0.0146101  0.0046943
#  0.03223421 0.0407038  0.00541583 0.00389028 0.00906574 0.00460133
#  0.00442166 0.00403416 0.11147162 0.017069   0.18611678 0.08903854
#  0.01171586 0.01891806 0.0380801  0.12223649 0.01037182 0.00702098]
# =====================================================
# acc after feature selection:  0.9824561403508771
# accuracy_score after feature selection:  0.9824561403508771
# RandomForestClassifier : [0.05134636 0.01636617 0.03880146 0.06300525 0.00329002 0.00597333
#  0.02717577 0.08493916 0.00325547 0.00545995 0.0238105  0.00539637
#  0.02536724 0.00521659 0.00404313 0.01193586 0.00724525 0.00647851
#  0.0042942  0.11543702 0.02517562 0.18228334 0.12357814 0.01546387
#  0.0129555  0.02531018 0.08838127 0.01015358 0.00786091]
#  1.47687622e-04]
# =====================================================
# acc after feature selection:  0.9649122807017544
# accuracy_score after feature selection:  0.9649122807017544
# GradientBoostingClassifier : [6.93335343e-04 5.84554715e-03 4.85018106e-03 2.07942976e-05
#  6.70721876e-04 1.36163032e-03 4.56037064e-02 2.65007505e-03
#  3.08640006e-03 6.61604646e-05 3.06606602e-03 4.58013611e-03
#  1.35460502e-03 1.49483809e-03 2.23008017e-04 4.26756967e-03
#  1.89350394e-03 2.27353904e-03 6.02062992e-01 8.61972915e-02
#  1.01648615e-01 2.14643889e-02 3.04109861e-03 2.93480510e-04
#  3.67523492e-03 9.51848986e-02 4.16637189e-04 2.01354453e-03]
# =====================================================
# acc :  0.9532163742690059
# accuracy_score :  0.9532163742690059
# DecisionTreeClassifier : [0.         0.         0.         0.00870516 0.         0.      
#  0.02149519 0.         0.         0.0099579  0.0420934  0.
#  0.01541539 0.         0.         0.         0.         0.
#  0.70814967 0.08469868 0.0072543  0.01160688 0.         0.
#  0.00262815 0.08799528 0.         0.        ]
# =====================================================
# acc after feature selection:  0.9590643274853801
# accuracy_score after feature selection:  0.9590643274853801
# DecisionTreeClassifier : [0.         0.         0.         0.         0.02804997 0.      
#  0.00940468 0.         0.         0.         0.01056046 0.
#  0.00185625 0.         0.02967669 0.00816109 0.         0.0172122
#  0.70814967 0.08469868 0.         0.01160688 0.         0.
#  0.00262815 0.08799528 0.         0.        ]
# =====================================================
# acc :  0.9707602339181286
# accuracy_score :  0.9707602339181286
# XGBClassifier : [1.3591385e-02 9.4278529e-03 3.9507940e-02 6.4421524e-03 6.3358927e-03
#  6.0578710e-03 2.9251541e-04 6.3979410e-02 7.8695687e-03 5.3178249e-03
#  1.7666128e-03 0.0000000e+00 9.7680492e-03 9.2516718e-03 1.0789652e-02
#  0.0000000e+00 5.7887365e-03 5.1565571e-03 4.6656245e-01 2.7369564e-02
#  2.0768178e-01 1.4816604e-02 8.7390179e-03 5.2482262e-03 1.4277947e-02
#  5.1601216e-02 1.1807767e-03 1.1787076e-03]
# =====================================================
# acc after feature selection:  0.9707602339181286
# accuracy_score after feature selection:  0.9707602339181286
# XGBClassifier : [7.0818882e-03 6.0684178e-03 4.1456446e-02 8.9651877e-03 8.0117760e-03
#  5.9159165e-03 0.0000000e+00 5.6270268e-02 2.8272675e-04 9.6123982e-03
#  0.0000000e+00 0.0000000e+00 5.6823944e-03 9.1377348e-03 1.0797856e-02
#  0.0000000e+00 9.6503098e-04 3.1478459e-03 5.5018383e-01 2.5124246e-02
#  1.5504247e-01 1.4744631e-02 4.6909908e-03 0.0000000e+00 1.4885771e-02
#  5.8954716e-02 1.3803941e-03 1.5970445e-03]
# =====================================================