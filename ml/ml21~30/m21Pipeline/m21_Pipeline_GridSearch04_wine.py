import numpy as np
from sklearn.datasets import load_wine
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.pipeline import make_pipeline,Pipeline
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV

#1.데이터
x,y = load_wine(return_X_y=True)

x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.8, shuffle= True, random_state=27
)

parameters = [{'rf__n_estimators' : [100, 200, 300]}, {'rf__max_depth' : [6, 10, 15, 12]}, 
            {'rf__min_samples_leaf' : [3, 10]},
    {'rf__min_samples_split' : [2, 3, 10]}, 
    {'rf__max_depth' : [6, 8, 12]}, 
    {'rf__min_samples_leaf' : [3, 5, 7, 10]},
    {'rf__n_estimators' : [100, 200, 400]},
    {'rf__min_samples_split' : [2, 3, 10]},
]

#2. 모델
pipe = Pipeline([("std", StandardScaler()),("rf", RandomForestClassifier())])  #->얘는 리스트로 받아놈.

model = GridSearchCV(pipe, parameters, cv =5, verbose=1, n_jobs= 3)
#pipe랑 parameters는 정상적인 파라미터가 아님.
#pipeline에 파라미터를 넣어야하는데 랜포 파라미터를 넣어서 오류가뜸. 해결책 rf__(정의한 것)을 넣으면됨.

# 3. 훈련
model.fit(x_train,y_train)

#4. 평가, 예측
result = np.round(model.score(x_test,y_test),4)
print("acc : ", result)

y_predict = model.predict(x_test)
acc = np.round(accuracy_score(y_test,y_predict),4)
print("accuracy_score : ", acc)

# acc :  1.0
# accuracy_score :  1.0