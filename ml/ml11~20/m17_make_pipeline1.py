#소문자 함수
import numpy as np
from sklearn.datasets import load_iris
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVC
#1.데이터
x,y = load_iris(return_X_y=True)

x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.8, shuffle= True, random_state=27
)

# scaler = MinMaxScaler()
# x_train = scaler.fit_transform(x_train)
# x_test = scaler.transform(x_test)

# n_split = 5
# kfold = KFold(n_split=n_split, shuffle=True, random_state=27)

#2. 모델
#model = RandomForestClassifier()
#model = make_pipeline(MinMaxScaler(),RandomForestClassifier()) #파이프라인에 스케일러랑, 모델을 넣으면됨.
#model = make_pipeline(StandardScaler(),RandomForestClassifier()) #파이프라인에 스케일러랑, 모델을 넣으면됨.
model = make_pipeline(StandardScaler(),SVC()) 

# 3. 훈련
model.fit(x_train,y_train)

#4. 평가, 예측
result = model.score(x_test,y_test)
print("acc : ", result)

y_predict = model.predict(x_test)
acc = accuracy_score(y_test,y_predict)
print("accuracy_score : ", acc)