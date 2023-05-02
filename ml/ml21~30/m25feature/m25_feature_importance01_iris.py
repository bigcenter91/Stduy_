
# 10개 데이터셋
# 10개의 파일을 만든다.
#[실습/과제] 피처를 한개씩 삭제하고 성능비교.
# 모델은 RF로만 한다.
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

# 필요 없는 특성 삭제
x= np.delete(x, 1, axis=1)

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
    print("=====================================================")
    print(type(model).__name__, ":", model.feature_importances_)

# 지우기전
# RandomForestClassifier : [0.10431864 0.02218342 0.35323494 0.52026301] 0.9

#지운후
#RandomForestClassifier : [0.1942312  0.44400177 0.36176703] 0.933333333333333

