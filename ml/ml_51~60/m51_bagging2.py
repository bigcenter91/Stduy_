#[실습] 각종 모델 10개 넣어서 확인해볼것!!!

import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

#1. 데이터
x, y = load_breast_cancer(return_X_y=True)

x_train, x_test, y_train, y_test = train_test_split(
    x, y, random_state=123, train_size=0.8, shuffle=True,
    stratify=y
)

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

#2. 모델
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier
from sklearn.linear_model import LogisticRegression # 분류

# model = DecisionTreeClassifier()
# model = RandomForestClassifier()
aaaa = LogisticRegression()
model = BaggingClassifier(aaaa,
                          n_estimators=10,
                          n_jobs=-1,
                          random_state=337,
                          bootstrap=True, # 디폴트, 중복허용
                          
                          )


#3. 훈련
model.fit(x_train, y_train)

#4. 평가, 예측
y_pred = model.predict(x_test)
print('score : ', model.score(x_test, y_test))
print('acc : ', accuracy_score(y_test, y_pred))
