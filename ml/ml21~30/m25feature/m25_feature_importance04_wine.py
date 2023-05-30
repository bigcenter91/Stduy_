import numpy as np
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

#1.데이터
x,y = load_wine(return_X_y=True)

# 필요 없는 특성 삭제
x= np.delete(x, 12, axis=1)

x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.8, shuffle= True, random_state=27
)

#2. 모델
model = RandomForestClassifier()

# 3. 훈련
model.fit(x_train,y_train)

#4. 평가, 예측
result = model.score(x_test,y_test)
print("acc : ", result)

y_predict = model.predict(x_test)
acc = accuracy_score(y_test,y_predict)
print("accuracy_score : ", acc)
print("=====================================================")
print(type(model).__name__, ":", model.feature_importances_)

#변경전
# [0.09774044 0.03428093 0.01457356 0.02424595 0.03659235 0.08318594
#  0.16759454 0.00893174 0.02012132 0.175029   0.07406631 0.08913255
#  0.17450536] 1.0

#변경후(가장큰값을 없애버림)
# [0.14990909 0.04643237 0.01593242 0.02866187 0.07212741 0.07574111
#  0.2004331  0.01257181 0.02110152 0.1755642  0.06446329 0.13706182] acc: 0.9722222222222222