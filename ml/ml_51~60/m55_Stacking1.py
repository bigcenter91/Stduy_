import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import VotingClassifier, StackingClassifier
from sklearn.ensemble import RandomForestClassifier
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
lr = LogisticRegression()
knn = KNeighborsClassifier()
dt = DecisionTreeClassifier()

model = StackingClassifier(
    estimators=[('LR',lr), ('KNN', knn), ('DT',dt)],
    # final_estimator=DecisionTreeClassifier()
    # final_estimator=LogisticRegression()
    # final_estimator=KNeighborsClassifier()
    final_estimator=RandomForestClassifier()
    # final_estimator=VotingClassifier()
    # 스태킹 안에 보팅 넣을 수 있다
)

#3. 훈련
model.fit(x_train, y_train)

#4. 평가, 예측
y_pred = model.predict(x_test)
print('score : ', model.score(x_test, y_test))
print('stacking Acc : ', accuracy_score(y_test, y_pred))

# hard voting 결과
# score :  0.9736842105263158
# acc :  0.9736842105263158 

# soft voting 결과
# score :  0.9736842105263158
# acc :  0.9736842105263158

classfiers = [lr, knn, dt]
for model2 in classfiers:
    model2.fit(x_train, y_train)
    y_predict = model2.predict(x_test)
    score2 = accuracy_score(y_test, y_predict)
    class_name = model2.__class__.__name__
    print("{0}정확도 : {1:.4f}".format(class_name, score2))
# print(model.__class__.__name__)

# 과적합의 문제성이 있다
# 항상 신경을 써야한다 과적합 

# score :  0.9824561403508771
# voting Acc :  0.9824561403508771
# LogisticRegression정확도 : 0.9737
# KNeighborsClassifier정확도 : 0.9737
# DecisionTreeClassifier정확도 : 0.9474