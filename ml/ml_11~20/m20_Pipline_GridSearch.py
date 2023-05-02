import numpy as np
from sklearn.datasets import load_iris
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.svm import SVC
# make_pipeline, Pipeline 동일한 놈이야
from sklearn.model_selection import GridSearchCV

#1. 데이터
x, y = load_iris(return_X_y=True)

x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.8, random_state=123, shuffle=True
)

# scaler = MinMaxScaler()
# x_train = scaler.fit_transform(x_train)
# x_test = scaler.transform(x_test)

parameters = [
    {'rf__n_estimators':[100,200], 'rf__max_depth':[6,8,10,12], 'rf__n_jobs':[-1,2,4]},
    {'rf__max_depth':[6,8,10,12], 'rf__min_samples_leaf':[3,5,7,10], 'rf__n_jobs':[-1,2,4]},
    {'rf__min_samples_leaf':[3,5,7,10], 'rf__min_samples_split':[2,3,5,10], 'rf__n_jobs':[-1,2,4]},
    {'rf__n_estimators':[100,200], 'rf__max_depth':[6,8,10,12], 'rf__min_samples_split':[2,3,5,10]},
    {'rf__min_samples_split':[2,3,5,10]},
    ]                                                                                       

# ValueError: Invalid parameter max_depth for estimator Pipeline(steps=[('std', StandardScaler()), ('rf', RandomForestClassifier())]). Check the list of available parameters with `estimator.get_params().keys()`.
# 위 파라미터는 randomforest 파라미터니까 pipeline의 파라미터 형태로 바꿔줘야한다

#2. 모델
# model = RandomForestClassifier()

# model = make_pipeline(MinMaxScaler(), RandomForestClassifier())
# model = make_pipeline(StandardScaler(), RandomForestClassifier())
# model = make_pipeline(StandardScaler(), SVC())
# pipe = Pipeline([("std", StandardScaler()), ("svc", SVC())])
pipe = Pipeline([("std", StandardScaler()), ("rf", RandomForestClassifier())])

# pipeline 설계 됐을 때 리스트 안에 튜플형태로 됐다
model = GridSearchCV(pipe, parameters, cv=5, verbose=1)



#3. 훈련
model.fit(x_train, y_train)
# fit 시킨 놈이 문제다 value 에러

#4. 평가, 예측
result = model.score(x_test, y_test)
print("model.score : ", result)

y_predict = model.predict(x_test)
acc = accuracy_score(y_test, y_predict)
print("accuracy_score : ", acc)